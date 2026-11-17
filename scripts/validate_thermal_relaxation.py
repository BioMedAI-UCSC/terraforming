"""Compare thermal relaxation against unchanged NASA and FMS routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.thermal_relaxation import fixed_tendency,height_tendency,analytic_tendency
from native_reference import compile_routines,extract_routines

PHYSICS_BLOB='ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc'
AXIS_BLOB='e9e612036fe858da56ac56868b976fc4b16be1f0'


def compile_reference(source,fms,destination,compiler):
    names=['NEAREST_INDEX_','INTERP_1D_LINEAR_','INTERP_1D_CUBIC_SPLINE_','FIND_INDEX_','INTERP_1D_3D_']
    routines,_=extract_routines(fms/'axis_utils/include/axis_utils2.inc',AXIS_BLOB,names)
    header='''#define FMS_AU_KIND_ 8
#define NEAREST_INDEX_ nearest_index
#define INTERP_1D_LINEAR_ interp_1d_linear
#define INTERP_1D_CUBIC_SPLINE_ interp_1d_cubic_spline
#define FIND_INDEX_ find_index
#define INTERP_1D_3D_ interp_1d
module axis_utils2_mod
implicit none
integer,parameter :: FATAL=1
contains
subroutine mpp_error(level,message)
integer :: level
character(len=*) :: message
print *,message
stop 1
end subroutine
'''+ '\n'.join(routines)+'''
end module
module reference_mod
use iso_c_binding
implicit none
real :: tka,tks
real :: t_zero=315.,t_strat=200.,delh=60.,delv=10.,eps=0.,sigma_b=.7
real,parameter :: KAPPA=189./735.
real,allocatable :: teq(:,:,:)
logical :: mcpu0
contains
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
'''
    adapter='''
subroutine thermal(n,ncol,mode,tau,surface_tau,lat,ps,p,t,z,zinput,target,mask,tdt,teqout) bind(C)
integer(c_int),value :: n,ncol,mode
real(c_double),value :: tau,surface_tau
real(c_double),intent(in) :: lat(ncol,1),ps(ncol,1),p(ncol,1,n),t(ncol,1,n)
real(c_double),intent(in) :: z(ncol,1,n),zinput(ncol,1,n),target(ncol,1,n),mask(ncol,1,n)
real(c_double),intent(out) :: tdt(ncol,1,n),teqout(ncol,1,n)
real :: seconds
seconds=tau
if(seconds<0.) seconds=-86400.*seconds
tka=0.
if(seconds>0.) tka=1./seconds
seconds=surface_tau
if(seconds<0.) seconds=-86400.*seconds
tks=0.
if(seconds>0.) tks=1./seconds
select case(mode)
case(0)
call newtonian_damping(ncol,1,n,lat,ps,p,t,target,tdt,mask)
teqout=target
case(1)
call newtonian_damping_z(ncol,1,n,lat,z,t,zinput,target,teqout,tdt,mask)
case(2)
allocate(teq(ncol,1,n))
call newtonian_damping_anal(lat,ps,p,t,tdt,1000.d0,mask)
teqout=teq
deallocate(teq)
end select
end subroutine
end module
'''
    library,provenance=compile_routines(source/'atmos_param_mars/mars_physics.F90',PHYSICS_BLOB,
        ['newtonian_damping','newtonian_damping_z','newtonian_damping_anal'],header,adapter,destination,compiler)
    provenance['fms_axis_git_blob']=AXIS_BLOB
    provenance['fms_routine_sha256']={name:hashlib.sha256(body.encode()).hexdigest()
        for name,body in zip(names,routines,strict=True)}
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.thermal.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*2+[array]*10
    native.thermal.restype=None
    return native,provenance


def validate(source,fms,compiler):
    jax.config.update('jax_enable_x64',True)
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    layout=lambda x:f(np.asarray(x).T[:,None,:])
    cases=[]
    with tempfile.TemporaryDirectory(prefix='thermal-relaxation-native-') as temporary:
        native,provenance=compile_reference(source,fms,Path(temporary),compiler)
        for levels in (3,40,56):
            lat=np.array([-np.pi/2,-.5,0.,.7,np.pi/2]);ps=np.array([500.,610.,705.,1000.,1500.])
            p=np.geomspace(.001,1.1,levels)[:,None]*ps
            zinput=np.linspace(100000.,0.,levels)[:,None]*np.ones((1,5))
            z=np.linspace(99999.,1.,levels)[:,None]*np.ones((1,5))
            target=170.+40.*np.cos(zinput/100000.*np.pi)+np.arange(5)[None,:]
            t=180.+60.*np.cos(z/50000.)+np.arange(5)[None,:]
            mask=np.ones_like(t);mask[::2,1]=0.;mask[1::2,3]=.5
            for mode in (0,1,2):
                for tau,surface_tau in ((0.,0.),(-4.,-1.),(86400.,345600.),(-1e9,-4.)):
                    expected=[np.empty((5,1,levels),order='F') for _ in range(2)]
                    native.thermal(levels,5,mode,tau,surface_tau,f(lat[:,None]),f(ps[:,None]),
                        layout(p),layout(t),layout(z),layout(zinput),layout(target),layout(mask),*expected)
                    if mode==0:
                        actual=(fixed_tendency(t,target,timescale=tau,mask=mask),target)
                    elif mode==1:
                        actual=height_tendency(t,z,zinput,target,timescale=tau,mask=mask)
                    else:
                        actual=analytic_tendency(t,p,ps,lat,1000.,kappa=189./735.,
                            timescale=tau,surface_timescale=surface_tau,mask=mask)
                    errors=[]
                    for value,ref in zip(actual,expected,strict=True):
                        ref=ref[:,0,:].T
                        np.testing.assert_allclose(value,ref,rtol=5e-13,atol=1e-12)
                        errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                    cases.append(dict(levels=levels,mode=mode,timescale=tau,
                        surface_timescale=surface_tau,max_errors=errors))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_thermal_relaxation.py','package/src/framework/physics/thermal_relaxation.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Fixed, height and analytic tendencies, masks and FMS height interpolation; NetCDF readers and production selection are separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--fms-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,args.fms_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
