"""Compare CO2 cloud phase changes and settling with unchanged Fortran routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.co2_microphysics import cloud_step
from native_reference import compile_routines,extract_routines

CLOUD_BLOB='aa566e12d6e01afa346162385055f41d8d2bc18f'
RADIATION_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler):
    solver,_=extract_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',
        RADIATION_BLOB,['dtridgl'])
    header="""module constants_mod
implicit none
real*8, parameter :: grav=3.72d0,rdgas=189.02d0
end module
module reference_mod
use iso_c_binding
use constants_mod
implicit none
real*8, parameter :: scale_dt=3.d0*(1.38064852d-23/1.66054d-27)/44.d0
contains
"""
    adapter='\n'+solver[0]+"""
subroutine clouds(n,dt,ts,half,full,t,q,kh,out_t,out_q,loss,deposit,radius) bind(C)
integer(c_int), value :: n
real(c_double), value :: dt,ts
real(c_double), intent(in) :: half(n+1),full(n),t(n),q(n),kh(n-1)
real(c_double), intent(out) :: out_t(n),out_q(n),loss(n),deposit(1),radius(n)
real*8 :: p(2*n+3),tl(2*n+3),qpi(2*n+3),kd(2*n+1),rho(n)
real*8 :: dev,pi
integer :: l,k
p=0.d0
tl=0.d0
qpi=0.d0
kd=0.d0
pi=acos(-1.d0)
dev=.3087d0
p(3)=half(1)
do l=1,n
  k=2*l+2
  p(k)=full(l)
  p(k+1)=half(l+1)
  tl(k)=t(l)
  qpi(k)=max(q(l),0.d0)
  if(l<n) then
    tl(k+1)=.5d0*(t(l)+t(l+1))
    kd(2*l+1)=kh(l)
  else
    tl(k+1)=.5d0*(t(l)+ts)
  endif
  rho(l)=full(l)/(rdgas*t(l))
enddo
tl(3)=tl(4)
call co2condsub(dt,p,tl,qpi,590200.d0,735.94d0,n)
do l=1,n
  k=2*l+2
  out_t(l)=tl(k)
  loss(l)=(qpi(k)-max(q(l),0.d0))*(half(l+1)-half(l))/grav
  radius(l)=(qpi(k)/1.d5*.75d0/pi/1620.d0)**(1.d0/3.d0)*exp(-.5d0*dev**2)
  radius(l)=max(radius(l),1.d-7)*exp(1.5d0*dev**2)
  ! Restore pre-phase temperatures for the sedimentation stencil, as in driver.
  tl(k)=t(l)
enddo
deposit=0.d0
call sedimco2(dt,p,tl,rho,kd,qpi,radius,1620.d0,deposit(1),n)
do l=1,n
  out_q(l)=qpi(2*l+2)
enddo
end subroutine
end module
"""
    library,provenance=compile_routines(source/'atmos_param_mars/co2micromod_mgcm.F90',
        CLOUD_BLOB,['co2condsub','sedimco2','fallvel'],header,adapter,destination,compiler)
    provenance['solver_source_git_blob']=RADIATION_BLOB
    provenance['solver_routine_sha256']=hashlib.sha256(solver[0].encode()).hexdigest()
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.clouds.argtypes=[ctypes.c_int]+[ctypes.c_double]*2+[array]*10
    native.clouds.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='co2-cloud-reference-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (2,4,12,40):
            for surface_pressure in (610.,50000.):
                half=np.concatenate([[0.],np.geomspace(.1,surface_pressure,n)])
                full=.5*(half[:-1]+half[1:])
                frost=3182.48/(23.3494-np.log(full/100.))
                for mode in ('clear','condense','evaporate','mixed'):
                    t=f(frost+(10. if mode in ('clear','evaporate') else -5.))
                    q=f(np.zeros(n) if mode in ('clear','condense') else np.geomspace(.0001,.05,n))
                    if mode=='mixed': t[::2]+=10.
                    for kh in (np.zeros(n-1),np.full(n-1,.001),np.geomspace(.01,100.,n-1)):
                        for dt in (5.,300.,30000.):
                            expected=[f(np.zeros(n)) for _ in range(3)]+[f(np.zeros(1)),f(np.zeros(n))]
                            native.clouds(n,dt,150.,f(half),f(full),t,q,f(kh),*expected)
                            result=cloud_step(t,half,full,q,150.,kh,dt)
                            actual=result[:5]
                            errors=[]
                            for name,value,ref in zip(('temperature','ice','mass_loss','deposit','radius'),actual,expected,strict=True):
                                np.testing.assert_allclose(value,np.squeeze(ref) if name=='deposit' else ref,
                                    rtol=2e-10,atol=2e-13,err_msg=f'{n=} {mode=} {dt=} {name=}')
                                errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                            assert np.min(result.ice_ratio)>=-1e-14
                            gas=np.diff(half)/3.72
                            np.testing.assert_allclose(np.sum(gas*q),
                                np.sum(gas*result.ice_ratio-result.gas_mass_loss_kg_m2)+result.deposition_kg_m2,
                                rtol=3e-12,atol=3e-13)
                            cases.append(dict(layers=n,surface_pressure_pa=surface_pressure,mode=mode,dt_seconds=dt,
                                max_diffusivity=float(np.max(kh)),implicit=bool(result.diagnostics['implicit']),
                                max_errors=errors))
    repo=Path(__file__).parents[1]
    paths=['scripts/validate_co2_microphysics.py','scripts/native_reference.py',
        'package/src/framework/physics/co2_microphysics.py','package/src/framework/physics/vertical_solve.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
        limits='Bulk cloud phase and settling only; optical properties and Dinosaur coupling are separate checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
