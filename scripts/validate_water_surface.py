"""Compare water surface exchange with the three unchanged release routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.water_surface import surface_exchange
from src.framework.physics.water_bulk import saturation_ratio
from native_reference import compile_routines

PHYSICS_BLOB='ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc'


def compile_reference(source,destination,compiler):
    header="""#define RELEASE
module constants_mod
implicit none
real*8,parameter :: grav=3.72d0
end module
module reference_mod
use iso_c_binding
implicit none
integer,parameter :: nice_mass=1
integer :: vapor_indx(1)=[1]
logical :: inj_vap_pbl,liquid_vap_pres,blk_latent_heat
real*8 :: facsubl
contains
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
"""
    adapter="""
subroutine exchange(n,kind,inject,liquid,latent,ktop,fac,dt,ts,drag,frost,ps,q,qout, &
 frostout,flux,heat,dheat) bind(C)
integer(c_int),value :: n,kind,inject,liquid,latent,ktop
real(c_double),value :: fac,dt,ts,drag,frost
real(c_double),intent(in) :: ps(1,1,n+1),q(n)
real(c_double),intent(out) :: qout(n),frostout(1),flux(1),heat(1),dheat(1)
real*8 :: r(1,1,n,1),rdt(1,1,n,1),out(1,1,n,1),water(1,1,1)
real :: lat(1,1),lon(1,1),tg(1,1),drg(1,1),wf(1,1),hl(1,1),dhl(1,1),dtm
integer :: kpbl(1,1)
lat=0.
lon=0.
tg=ts
drg=drag
kpbl=ktop
dtm=dt
water=frost
r(1,1,:,1)=q
rdt=0.d0
hl=0.
dhl=0.
facsubl=fac
inj_vap_pbl=inject/=0
liquid_vap_pres=liquid/=0
blk_latent_heat=latent/=0
if(kind==0) then
 call update_water_blk(1,1,1,1,n,1,lat,lon,dtm,ps,tg,drg,kpbl,r,rdt,1,water(:,:,1),wf,out,hl,dhl)
else if(kind==1) then
 call update_water(1,1,1,1,n,1,lat,lon,dtm,ps,tg,drg,kpbl,r,rdt,1,water,wf,out,0.)
else
 call update_water_fv3(1,1,1,1,n,1,lat,lon,dtm,ps,tg,drg,kpbl,r,rdt,1,water(:,:,1),wf,out)
endif
qout=q+dt*out(1,1,:,1)
frostout=water(1,1,1)
flux=wf(1,1)
heat=hl(1,1)
dheat=dhl(1,1)
end subroutine
end module
"""
    library,provenance=compile_routines(source/'atmos_param_mars/mars_physics.F90',PHYSICS_BLOB,
        ['update_water','update_water_fv3','update_water_blk'],header,adapter,destination,compiler)
    provenance['preprocessor_definitions']=['RELEASE']
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.exchange.argtypes=[ctypes.c_int]*6+[ctypes.c_double]*5+[array]*7
    native.exchange.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='water-surface-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (2,12,40):
            half=f(np.linspace(0.,610.,n+1))
            for kind,scheme in enumerate(('bulk','moment','bin')):
                for inject in (False,True):
                    for skin in (140.,240.,290.):
                        q=f(np.full(n,float(saturation_ratio(skin,610.))*.5))
                        for name,amount,drag,scale in (('wet',100.,.001,1.),('limited',1e-8,.1,2.),
                                ('deposition',0.,.01,.5),('invalid',-1.,.01,1.)):
                            if name=='deposition': vapor=q*4.
                            else: vapor=q
                            outputs=[np.zeros(n),*[np.zeros(1) for _ in range(4)]]
                            top=max(0,n-3)
                            native.exchange(n,kind,int(inject),1,1,top+1,scale,300.,skin,drag,amount,
                                half,f(vapor),*outputs)
                            actual=surface_exchange(vapor,half,skin,amount,drag,top,300.,
                                scheme=scheme,inject_through_pbl=inject,sublimation_factor=scale,
                                liquid_vapor_pressure=True,bulk_latent_heat=True)
                            errors=[]
                            for a,b in zip(actual[:5],(outputs[0],*(x[0] for x in outputs[1:])),strict=True):
                                np.testing.assert_allclose(a,b,rtol=3e-12,atol=3e-12)
                                errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                            before=np.sum(np.diff(half)/3.72*vapor)+amount
                            after=np.sum(np.diff(half)/3.72*actual.vapor_ratio)+actual.surface_water_kg_m2
                            np.testing.assert_allclose(after-before,actual.diagnostics['repaired_surface_mass_kg_m2'],
                                rtol=3e-12,atol=3e-12)
                            cases.append(dict(layers=n,scheme=scheme,inject_through_pbl=inject,
                                surface_temperature_k=skin,case=name,max_errors=errors))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_water_surface.py','package/src/framework/physics/water_surface.py',
        'package/src/framework/physics/water_bulk.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Released bulk/moment/bin water surface exchange; source latent formulas retained literally; full water driver coupling requires separate checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
