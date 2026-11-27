"""Check released bulk-water roots and settling against unchanged Fortran."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from scipy.optimize import brentq
from src.framework.gcm._dinosaur import jax
from src.framework.physics.water_bulk import cloud_step,phase_change,_findq,find_condensed_mass,saturation_ratio
from native_reference import compile_routines,extract_routines

WATER_BLOB='92667ec8575160bc8e6a68e88cd0a93397d3d450'
RADIATION_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler):
    solver,_=extract_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,['dtridgl'])
    header="""module constants_mod
implicit none
real*8,parameter :: grav=3.72d0,rdgas=189.02d0,cp_air=735.94d0,pi=acos(-1.d0),kboltz=1.38064852d-23
end module
module reference_mod
use iso_c_binding
use constants_mod
implicit none
type time_type
integer :: unused=0
end type
integer,parameter :: model_atmos=1
integer,parameter :: id_blkh2ocld=0,id_blkh2ocld_col=0,id_blkh2ocld_r=0,id_blkvap_col=0
integer,parameter :: id_cprecip=0,id_cprecip_rain=0,id_cprecip_snow=0
integer,parameter :: id_cprecip_thres=0,id_cprecip_thres_rain=0,id_cprecip_thres_snow=0
integer,parameter :: id_cprecip_sedim=0,id_cprecip_sedim_rain=0,id_cprecip_sedim_snow=0
real*8,parameter :: aerdensh2o=917.d0,lw=2.8d6
real*8 :: ccn_blkh2o=1.d5,prec_threshold=.001d0
logical :: mcpu0
real*8,allocatable :: sfc_frost_blk(:,:,:),cumulative_prec_blk(:,:,:)
real*8,allocatable :: cumulative_prec_thres(:,:,:),cumulative_prec_sedim(:,:,:)
interface send_data
module procedure send2,send3
end interface
contains
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
integer function find_field_index(model,name)
integer :: model
character(len=*) :: name
find_field_index=2
if(name=='vap_mass_blk') find_field_index=1
end function
logical function send2(id,values,clock,is,js)
integer :: id,is,js
type(time_type) :: clock
real*8 :: values(:,:)
send2=.true.
end function
logical function send3(id,values,clock,is,js)
integer :: id,is,js
type(time_type) :: clock
real*8 :: values(:,:,:)
send3=.true.
end function
"""
    adapter='\n'+solver[0]+"""
subroutine water(n,dt,ts,ccn,threshold,half,full,t,vapor,ice,kh,out_t,out_vapor,out_ice,deposit) bind(C)
integer(c_int),value :: n
real(c_double),value :: dt,ts,ccn,threshold
real(c_double),intent(in) :: half(1,1,n+1),full(1,1,n),t(1,1,n),vapor(n),ice(n),kh(n-1)
real(c_double),intent(out) :: out_t(n),out_vapor(n),out_ice(n),deposit(3)
real*8 :: r(1,1,n,2),rdt(1,1,n,2),rdt_out(1,1,n,2),tdt(1,1,n),tdt_out(1,1,n)
real*8 :: tg(1,1),mixing(1,1,2*n+1)
type(time_type) :: clock
integer :: i
if(allocated(sfc_frost_blk)) then
 deallocate(sfc_frost_blk,cumulative_prec_blk,cumulative_prec_thres,cumulative_prec_sedim)
endif
allocate(sfc_frost_blk(1,1,3),cumulative_prec_blk(1,1,3))
allocate(cumulative_prec_thres(1,1,3),cumulative_prec_sedim(1,1,3))
sfc_frost_blk=0.d0
cumulative_prec_blk=0.d0
cumulative_prec_thres=0.d0
cumulative_prec_sedim=0.d0
ccn_blkh2o=ccn
prec_threshold=threshold
r(1,1,:,1)=vapor
r(1,1,:,2)=ice
rdt=0.d0
tdt=0.d0
tg=ts
mixing=0.d0
do i=1,n-1
 mixing(1,1,2*i+1)=kh(i)
enddo
call blkh2o_driver(1,1,1,1,n,2,clock,half,full,t,tdt,r,rdt,tg,dt,rdt_out,mixing,tdt_out,.false.)
out_t=t(1,1,:)+dt*tdt_out(1,1,:)
out_vapor=vapor+dt*rdt_out(1,1,:,1)
out_ice=ice+dt*rdt_out(1,1,:,2)
deposit=cumulative_prec_blk(1,1,:)
end subroutine
subroutine root(q,p,t,guess,result) bind(C)
real(c_double),value :: q,p,t,guess
real(c_double),intent(out) :: result(1)
real*8 :: vapor,pressure,temp
vapor=q
pressure=p
temp=t
result(1)=guess
call findq(vapor,pressure,temp,result(1),lw,cp_air)
end subroutine
subroutine settling(n,dt,ts,half,full,tpre,tpost,qice,kh,qout,deposit,radius) bind(C)
integer(c_int),value :: n
real(c_double),value :: dt,ts
real(c_double),intent(in) :: half(n+1),full(n),tpre(n),tpost(n),qice(n),kh(n-1)
real(c_double),intent(out) :: qout(n),deposit(1),radius(n)
real*8 :: p(2*n+3),tl(2*n+3),qpi(2*n+3),kd(2*n+1),rho(n),scale
integer :: l,k
p=0.d0
tl=0.d0
qpi=0.d0
kd=0.d0
p(3)=half(1)
tl(3)=tpre(1)
scale=3.d0*(kboltz/1.66054d-27)/44.d0
do l=1,n
 k=2*l+2
 p(k)=full(l)
 p(k+1)=half(l+1)
 tl(k)=tpre(l)
 qpi(k)=qice(l)
 rho(l)=full(l)/(rdgas*tpost(l))
 radius(l)=max((qice(l)/1.d5*.75d0/pi/917.d0)**(1.d0/3.d0),1.d-7)
 if(l<n) then
  tl(k+1)=.5d0*(tpre(l)+tpre(l+1))
  kd(2*l+1)=kh(l)
 else
  tl(k+1)=.5d0*(ts+tpre(l))
 endif
enddo
deposit=0.d0
call sedimh2o(dt,p,tl,rho,kd,qpi,radius,917.d0,deposit(1),scale,n)
do l=1,n
 qout(l)=qpi(2*l+2)
enddo
end subroutine
end module
"""
    library,provenance=compile_routines(source/'atmos_param_mars/blkh2omod_mgcm.F90',WATER_BLOB,
        ['blkh2o_driver','findq','newton','watsat','sedimh2o','fallvel'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.water.argtypes=[ctypes.c_int]+[ctypes.c_double]*4+[array]*10;native.water.restype=None
    native.root.argtypes=[ctypes.c_double]*4+[array];native.root.restype=None
    native.settling.argtypes=[ctypes.c_int]+[ctypes.c_double]*2+[array]*9;native.settling.restype=None
    provenance['solver_git_blob']=RADIATION_BLOB
    return native,provenance


def tight_root(q,p,t,guess):
    def residual(value):
        return q-value-max(float(saturation_ratio(max(t+value*2.8e6/735.94,60.),p)),1e-50)
    return brentq(residual,min(0.,guess),max(0.,guess),xtol=1e-15,rtol=1e-14)


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[];roots=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='bulk-water-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for p in (30.,610.,50000.):
            for t in (140.,200.,270.,320.):
                sat=float(saturation_ratio(t,p))
                for ratio in (.2,2.):
                    q=sat*ratio;guess=-2.*sat if ratio<1. else q
                    expected=np.zeros(1);native.root(q,p,t,guess,expected)
                    raw=float(_findq(q,p,t,guess,2.8e6,735.94))
                    np.testing.assert_allclose(raw,expected[0],rtol=3e-12,atol=3e-14)
                    polished=float(find_condensed_mass(q,p,t,guess))
                    exact=tight_root(q,p,t,guess)
                    np.testing.assert_allclose(polished,exact,rtol=3e-11,atol=3e-14)
                    roots.append(dict(pressure_pa=p,temperature_k=t,saturation_ratio=ratio,
                        native_root_error=abs(expected[0]-exact),polished_root_error=abs(polished-exact)))
        for n in (2,12,40):
            half=np.linspace(0.,610.,n+1);full=.5*(half[:-1]+half[1:])
            for mode in ('clear','condense','evaporate','mixed'):
                t=np.linspace(150.,285.,n)
                sat=np.asarray(saturation_ratio(t,full))
                vapor=sat*(2. if mode=='condense' else .2)
                ice=np.full(n,.002 if mode in ('evaporate','mixed') else 0.)
                if mode=='clear': vapor[:]=0.
                if mode=='mixed': vapor[::2]=2.*sat[::2]
                for dt in (5.,300.,30000.):
                    kh=np.geomspace(.001,10.,n-1)
                    expected=[np.zeros(n) for _ in range(3)]+[np.zeros(3)]
                    native.water(n,dt,250.,1e5,.001,f(half),f(full),f(t),f(vapor),f(ice),f(kh),*expected)
                    actual=cloud_step(t,half,full,vapor,ice,250.,kh,dt)
                    phase_t,phase_v,phase_i=phase_change(t,full,vapor,ice)
                    # The source root stops at |step|<1e-8. Keep that difference
                    # separate from the strict sedimentation and radius gates.
                    np.testing.assert_allclose(phase_v,expected[1],rtol=3e-12,atol=2e-8)
                    np.testing.assert_allclose(phase_t,expected[0],rtol=3e-12,atol=2e-8*2.8e6/735.94)
                    qsettle=np.zeros(n);deposit=np.zeros(1);radius=np.zeros(n)
                    formed=np.minimum(np.asarray(phase_i),.001)
                    native.settling(n,dt,250.,f(half),f(full),f(t),f(phase_t),f(formed),f(kh),
                        qsettle,deposit,radius)
                    np.testing.assert_allclose(actual.ice_ratio,qsettle,rtol=3e-12,atol=3e-14)
                    np.testing.assert_allclose(actual.radius_m,radius,rtol=3e-12,atol=1e-18)
                    threshold_mass=np.sum(np.diff(half)/3.72*np.maximum(np.asarray(phase_i)-.001,0.))
                    np.testing.assert_allclose(actual.deposition_kg_m2,deposit[0]+threshold_mass,rtol=3e-12,atol=3e-14)
                    gas=np.diff(half)/3.72
                    np.testing.assert_allclose(np.sum(gas*(vapor+ice)),
                        np.sum(gas*(actual.vapor_ratio+actual.ice_ratio))+actual.deposition_kg_m2,
                        rtol=3e-12,atol=3e-12)
                    assert np.isfinite(np.asarray(actual.temperature)).all()
                    assert np.min(actual.vapor_ratio)>=-1e-14 and np.min(actual.ice_ratio)>=-1e-14
                    cases.append(dict(layers=n,case=mode,dt_seconds=dt,
                        native_temperature_difference_k=float(np.max(np.abs(actual.temperature-expected[0]))),
                        native_vapor_difference=float(np.max(np.abs(actual.vapor_ratio-expected[1]))),
                        native_ice_difference=float(np.max(np.abs(actual.ice_ratio-expected[2]))),
                        native_deposition_difference_kg_m2=abs(float(actual.deposition_kg_m2)-expected[3][0])))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_bulk_water.py','package/src/framework/physics/water_bulk.py',
        'package/src/framework/physics/co2_microphysics.py','scripts/native_reference.py']
    return dict(status='pass',reference=provenance,roots=roots,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Original root iteration matches Fortran; polished saturation roots match independent Brent solves. Settling and radii match unchanged native routines. Whole-driver stopping differences are reported. Surface exchange and moment microphysics require separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),roots=len(report['roots']))))
