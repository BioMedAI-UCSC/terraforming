"""Check released fixed-boundary values and the compiled no-surface fallback."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from native_reference import compile_routines

PHYSICS_BLOB='ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc'


def validate(source,compiler):
    path=source/'atmos_param_mars/mars_physics.F90';text=path.read_text()
    start=text.index('    tsurf(:,:)     = 170.0')
    fixed=text[start:text.index('endif',start)]
    header='''#define MARS_GCM
#define MARS_SURFACE
#define RELEASE
module reference_mod
use iso_c_binding
implicit none
type time_type
integer :: seconds=0
end type
contains
'''
    adapter='''
subroutine boundary(n,tsurf,albedo,sfc_emiss) bind(C)
integer(c_int),value :: n
real(c_double),intent(out) :: tsurf(n,1),albedo(n,1),sfc_emiss(n,1)
'''+fixed+'''
end subroutine
subroutine fallback(n,du,dv,dtout,dq) bind(C)
integer(c_int),value :: n
real(c_double),intent(inout) :: du(1,1,n),dv(1,1,n),dtout(1,1,n),dq(1,1,n,2)
real :: ph(1,1,n+1),p(1,1,n),zh(1,1,n+1),z(1,1,n),surface(1,1)
real :: u(1,1,n),v(1,1,n),t(1,1,n),q(1,1,n,2)
type(time_type) :: Time
integer :: k
do k=1,n+1
 ph(:,:,k)=610.d0*real(k-1)/n
enddo
p=.5*(ph(:,:,1:n)+ph(:,:,2:n+1))
zh=0.
z=0.
surface=170.
u=10.
v=5.
t=180.
q=.001
call diffuse(1,1,1,1,n,2,5.d0,Time,surface,surface,ph,p,zh,z,surface, &
 u,v,t,q,u,v,t,q,du,dv,dtout,dq,surface)
end subroutine
end module
'''
    with tempfile.TemporaryDirectory(prefix='surface-controls-native-') as temporary:
        library,provenance=compile_routines(path,PHYSICS_BLOB,['diffuse'],header,adapter,Path(temporary),compiler)
        native=ctypes.CDLL(str(library))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
        native.boundary.argtypes=[ctypes.c_int]+[array]*3;native.boundary.restype=None
        native.fallback.argtypes=[ctypes.c_int]+[array]*4;native.fallback.restype=None
        values=[np.empty((17,1),order='F') for _ in range(3)]
        native.boundary(17,*values)
        for value,expected in zip(values,(170.,.25,1.),strict=True):
            np.testing.assert_array_equal(value,expected)
        cases=[]
        for n in (2,4,40,56):
            for seed in (0.,1.,-.1):
                arrays=[np.full((1,1,n),seed,order='F') for _ in range(3)]
                arrays.append(np.full((1,1,n,2),seed,order='F'))
                before=[value.copy() for value in arrays]
                native.fallback(n,*arrays)
                for actual,expected in zip(arrays,before,strict=True):
                    np.testing.assert_array_equal(actual,expected)
                cases.append(dict(levels=n,seed=seed))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_surface_controls.py','package/src/framework/physics/boundary_layer.py',
        'package/src/framework/physics/column_physics.py','package/src/framework/physics/dinosaur_physics.py']
    provenance['fixed_boundary_block_sha256']=hashlib.sha256(fixed.encode()).hexdigest()
    return dict(status='pass',reference=provenance,fixed_boundary=dict(temperature_k=170.,albedo=.25,emissivity=1.),
        cases=cases,sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Unchanged fixed-boundary assignments and original diffuse compiled with released flags: tendency outputs remain unchanged because fv3_turb is excluded. Inactive PBL exchange fields are undefined in source; the port defines them as zero, tested separately. Surface-dependent water/dust configurations without initialized reservoirs are rejected.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
