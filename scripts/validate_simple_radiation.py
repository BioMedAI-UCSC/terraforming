"""Check simple-radiation columns against unchanged pinned native routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.simple_radiation import column_fluxes,dust_depth,scatter
from native_reference import compile_routines,extract_routines

DRIVER_BLOB='3e17830d742bcec90d00165db8c8037876d65f5a'
SCATTER_BLOB='09de59736cfefdc3011be89ef5ae81ceaced98d5'


def compile_reference(source,directory,compiler):
    path=source/'atmos_param_mars/Rad'
    routines,_=extract_routines(path/'radiation_util.f90',SCATTER_BLOB,['scatter_calc','ecoeffs'])
    header='''module reference_mod
use iso_c_binding
implicit none
real,parameter :: gcp=3.72d0/735.94d0,stefan=5.6734d-8
real :: conrath,reference_press,tauir,alfir
logical :: use_forget_swheat,do_lat_vary_dust,use_newton_damping
contains
'''+ '\n'.join(routines)+'\n'
    adapter='''
subroutine simple(n,ncol,flags,options,p,ph,t,ts,emiss,lat,mu,alb,tau,heating,sw,lw,outgoing) bind(C)
integer(c_int),value :: n,ncol
integer(c_int),intent(in) :: flags(4)
real(c_double),intent(in) :: options(8),p(ncol,1,n),ph(ncol,1,n+1),t(ncol,1,n)
real(c_double),intent(in) :: ts(ncol,1),emiss(ncol,1),lat(ncol,1),mu(ncol,1),alb(ncol,1)
real(c_double),intent(out) :: tau(ncol,1,n),heating(ncol,1,n),sw(ncol,1),lw(ncol,1),outgoing(ncol,1)
real :: delp(ncol,1,n),sscat(ncol,1,n),gfac(ncol,1,n),tcol(ncol,1,n+1)
real :: hsw(ncol,1,n),heatra(ncol,1,n),trans(ncol,1),outflx(ncol,1),sfc_ir_flx(ncol,1)
real :: radin(ncol,1),sols,tlen,ramp
use_forget_swheat=flags(2)/=0
do_lat_vary_dust=flags(1)/=0
use_newton_damping=flags(3)/=0
reference_press=options(2)
conrath=options(3)
tauir=options(6)
alfir=options(7)
sols=options(8)
delp=ph(:,:,2:n+1)-ph(:,:,1:n)
call dust_distrib_fixed(1,1,0.d0,lat,lat,options(1),reference_press,p,ph,delp,tau)
sscat=options(4)
gfac=options(5)
radin=610.d0*.75d0*mu
call swheat(p,ph,radin,alb,mu,tau,sscat,gfac,hsw,trans,outflx)
tcol(:,:,1:n)=t
tcol(:,:,n+1)=ts
call lwheat(p,ph,tcol,ts,emiss,heatra,sfc_ir_flx)
tlen=5.0
if(sols.lt.tlen) then
    ramp=0.5*(1.0-cos(sols*acos(-1.d0)/tlen))
else
    ramp=1.0
endif
hsw=ramp*hsw
trans=ramp*trans
heating=heatra+hsw
sw=trans
lw=sfc_ir_flx
if(flags(4)/=0) lw=0.
outgoing=outflx
end subroutine
subroutine scattering(n,ncol,tau,w,g,f,mu,alb,heat,ground,outgoing) bind(C)
integer(c_int),value :: n,ncol
real(c_double),intent(in) :: tau(ncol,n),w(ncol,n),g(ncol,n),f(ncol,n),mu(ncol),alb(ncol)
real(c_double),intent(out) :: heat(ncol,n),ground(ncol),outgoing(ncol)
call scatter_calc(ncol,n,tau,w,g,f,mu,alb,heat,ground,outgoing)
end subroutine
end module
'''
    library,provenance=compile_routines(path/'simple_rad_driver.F90',DRIVER_BLOB,
        ['dust_distrib_fixed','swheat','lwheat'],header,adapter,directory,compiler)
    provenance['scattering_source_git_blob']=SCATTER_BLOB
    provenance['scattering_routine_sha256']={name:hashlib.sha256(body.encode()).hexdigest()
        for name,body in zip(['scatter_calc','ecoeffs'],routines,strict=True)}
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.simple.argtypes=[ctypes.c_int]*2+[integers]+[array]*14
    native.simple.restype=None
    native.scattering.argtypes=[ctypes.c_int]*2+[array]*9
    native.scattering.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    f=lambda value:np.asfortranarray(value,dtype=np.float64)
    layout=lambda value:f(np.asarray(value).T[:,None,:])
    cases=[]
    with tempfile.TemporaryDirectory(prefix='simple-radiation-native-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (4,40,56):
            ph=np.linspace(0.,1.,n+1)[:,None]**2*np.array([300.,610.,705.,1000.,1500.])
            p=.5*(ph[1:]+ph[:-1]);t=160.+50.*np.cos(p/1000.)
            ts=np.array([150.,170.,200.,250.,300.]);emiss=np.array([0.,.3,.7,.9,1.])
            lat=np.array([-1.57,-.7,0.,.5,1.57]);mu=np.array([0.,.001,.3,.6,.99])
            alb=np.array([0.,.1,.3,.6,.9])
            for index in range(8):
                flags=np.array([index&1,(index>>1)&1,(index>>2)&1,(index>>1)&1],dtype=np.int32)
                optical=[0.,.2,2.,10.][index%4];ssa=[0.,.92,.999][index%3]
                options=f([optical,610.,.001,ssa,.55,[0.,1.,5.][index%3],.3 if index&1 else 0.,[0.,2.5,5.,10.][index%4]])
                tau=np.empty((5,1,n),order='F');heating=np.empty_like(tau,order='F')
                surface=[np.empty((5,1),order='F') for _ in range(3)]
                native.simple(n,5,flags,options,layout(p),layout(ph),layout(t),f(ts[:,None]),
                    f(emiss[:,None]),f(lat[:,None]),f(mu[:,None]),f(alb[:,None]),tau,heating,*surface)
                flux=column_fluxes(t,ts,ph,p,mu,610.,alb,emiss,lat,options[7],daylight_fraction=.75,
                    optical_depth=optical,single_scattering_albedo=ssa,pressure_exponent=options[6],
                    tau_infrared=options[5],latitude_variation=bool(flags[0]),
                    forget_shortwave_heating=bool(flags[1]),newton_damping=bool(flags[2]),
                    no_downward_infrared=bool(flags[3]))
                actual_heat=flux.atmospheric_convergence_w_m2/(735.94*np.diff(ph,axis=0)/3.72)+flux.nonlte_heating_k_s
                actual_tau=dust_depth(ph,p,lat,optical,latitude_variation=bool(flags[0]))
                values=[actual_tau,actual_heat,flux.shortwave_down_w_m2[-1]-flux.shortwave_up_w_m2[-1],flux.longwave_down_w_m2[-1]]
                refs=[tau[:,0,:].T,heating[:,0,:].T,surface[0][:,0],surface[1][:,0]]
                errors=[]
                for actual,expected in zip(values,refs,strict=True):
                    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12)
                    errors.append(float(np.max(np.abs(np.asarray(actual)-expected))))
                # Also check arbitrary layer optics against the unchanged scatter driver.
                w=np.broadcast_to(ssa+.0001*np.arange(n)[:,None]/n,p.shape).copy()
                g=np.broadcast_to(np.linspace(.1,.8,n)[:,None],p.shape).copy();forward=g*g
                expected_heat=np.empty((5,n),order='F');ground=np.empty(5);outgoing=np.empty(5)
                native.scattering(n,5,f(actual_tau.T),f(w.T),f(g.T),f(forward.T),f(mu),f(alb),expected_heat,ground,outgoing)
                result=scatter(actual_tau,w,g,forward,mu,alb)
                for actual,expected in zip(result[:3],(expected_heat.T,ground,outgoing),strict=True):
                    np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-12)
                cases.append(dict(layers=n,flags=flags.tolist(),options=options.tolist(),max_errors=errors))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_simple_radiation.py','package/src/framework/physics/simple_radiation.py']
    return dict(status='pass',reference=provenance,cases=cases,scattering_cases=len(cases),
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Unchanged dust profile, ecoeffs/scatter_calc, swheat and lwheat; source CUBE_CORE ramp/composition. Astronomy is checked separately. SPEC_CORE-only heating modulation is excluded by the released compile flags.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),scattering_cases=report['scattering_cases'])))
