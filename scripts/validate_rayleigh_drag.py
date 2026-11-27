"""Compare Rayleigh drag switches against the original reference routine."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.rayleigh_drag import rayleigh_tendencies
from native_reference import compile_routines

PHYSICS_BLOB='ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc'


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
real :: sigma_b=.7,vkf
real, parameter :: radian=180.d0/acos(-1.d0)
logical :: rayleighModelTop_flag=.false.,rayleighTropical_flag=.false.
real :: rayleighModelTop_tau_days=1.,rayleighModelTop_pres_inflex=.1,rayleighModelTop_pres_cutoff=1.
real :: rayleighTropical_tau_days=1.,rayleighTropical_pres_inflex=.1,rayleighTropical_pres_cutoff=1.
real :: rayleighTropical_lat_width=30.
contains
"""
    adapter="""
subroutine drag(n,ncol,top,trop,tau,ps,p,lat,u,v,mask,du,dv) bind(C)
integer(c_int), value :: n,ncol,top,trop
real(c_double), value :: tau
real(c_double), intent(in) :: ps(ncol,1),p(ncol,1,n),lat(ncol,1)
real(c_double), intent(in) :: u(ncol,1,n),v(ncol,1,n),mask(ncol,1,n)
real(c_double), intent(out) :: du(ncol,1,n),dv(ncol,1,n)
real :: kf
kf=tau
if(kf<0.) kf=-86400.*kf
vkf=0.
if(kf>0.) vkf=1./kf
rayleighModelTop_flag=top/=0
rayleighTropical_flag=trop/=0
call rayleigh_damping(ncol,1,n,ps,p,lat,u,v,du,dv,mask)
end subroutine
end module
"""
    lib,provenance=compile_routines(source/'atmos_param_mars/mars_physics.F90',PHYSICS_BLOB,
        ['rayleigh_damping'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(lib))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.drag.argtypes=[ctypes.c_int]*4+[ctypes.c_double]+[array]*8
    native.drag.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    layout=lambda x:f(np.asarray(x).T[:,None,:])
    cases=[]
    with tempfile.TemporaryDirectory(prefix='rayleigh-drag-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (4,40):
            ps=np.array([300.,610.,1500.]);lat=np.array([-.7,0.,.7])
            p=np.geomspace(1e-8,.99,n)[:,None]*ps
            u=np.broadcast_to(np.linspace(-250.,300.,n)[:,None],p.shape).copy();v=u*.3
            mask=np.ones_like(p);mask[::2,1]=0.;mask[1::2,1]=.5
            for top,trop in ((False,False),(True,False),(False,True),(True,True)):
                for tau in (0.,-1.,86400.,-4000.e6):
                    expected=[np.zeros((3,1,n),order='F') for _ in range(2)]
                    native.drag(n,3,int(top),int(trop),tau,f(ps[:,None]),layout(p),f(lat[:,None]),
                        layout(u),layout(v),layout(mask),*expected)
                    actual=rayleigh_tendencies(u,v,p,ps,lat,timescale=tau,model_top=top,tropical=trop,mask=mask)
                    errors=[]
                    for value,ref in zip(actual,expected,strict=True):
                        ref=ref[:,0,:].T
                        np.testing.assert_allclose(value,ref,rtol=3e-13,atol=1e-15)
                        errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                    cases.append(dict(layers=n,model_top=top,tropical=trop,timescale=tau,max_errors_m_s2=errors))
    repo=Path(__file__).parents[1]
    paths=['scripts/validate_rayleigh_drag.py','package/src/framework/physics/rayleigh_drag.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
        limits='Rayleigh damping and its top/tropical switches; gravity-wave schemes are separate processes')


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
