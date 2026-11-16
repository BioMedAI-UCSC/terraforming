"""Check the released cold-start defaults and unchanged surface initializer."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from run_gcm3d_ablation import reference_surface_temperature_k

PATCH_BLOB='4116b38b088d4f39a5d8c1da76acc1ffb07847db'


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    source=Path(source)
    surface=(source/'atmos_param_mars/mars_surface.F90').read_bytes()
    patch=(source/'patches/srcGFDL_acs.patch').read_bytes()
    blob=lambda raw:hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    # The surface pin is shared with the complete soil/frost validator.
    from validate_soil_frost import SURFACE_BLOB as pinned_surface
    assert blob(surface)==pinned_surface
    assert blob(patch)==PATCH_BLOB
    assert re.search(r'^\+\s*isotemp\s*=\s*170\.',patch.decode(),re.M)
    assert re.search(r'^\+\s*do_isothermal\s*=\s*\.true\.',patch.decode(),re.M|re.I)
    text=surface.decode();start=text.index('    if (use_equilibrated_ts) then')
    block=text[start:text.index('    sfc_snow(:,:)= 0.0',start)]
    code='''module reference_mod
use iso_c_binding
implicit none
contains
subroutine cold_start(n,latitude,temperature,soil) bind(C)
integer(c_int),value :: n
real(c_double),intent(in) :: latitude(n)
real(c_double),intent(out) :: temperature(n),soil(n,17)
real :: lat(n,1),t_surf(n,1),tsoil(n,1,17)
real,parameter :: pi=3.1415926535897932384626433832795,soil_temp=170.
integer,parameter :: nlayers=16
logical,parameter :: use_equilibrated_ts=.true.
integer :: k
lat(:,1)=latitude
'''+block+'''
temperature=t_surf(:,1)
soil=tsoil(:,1,:)
end subroutine
end module
'''
    lat=np.unique(np.r_[np.linspace(-np.pi/2,np.pi/2,181),-15.*np.pi/180.+np.array([-1e-8,0.,1e-8])])
    with tempfile.TemporaryDirectory(prefix='cold-start-native-') as temporary:
        directory=Path(temporary);(directory/'reference.f90').write_text(code)
        flags=['-O3','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
        subprocess.run([compiler,*flags,'reference.f90','-o','reference.so'],cwd=directory,check=True)
        native=ctypes.CDLL(str(directory/'reference.so'))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
        native.cold_start.argtypes=[ctypes.c_int,array,array,array];native.cold_start.restype=None
        expected=np.empty_like(lat);soil=np.empty((len(lat),17),order='F')
        native.cold_start(len(lat),lat,expected,soil)
    actual=np.asarray(reference_surface_temperature_k(lat))
    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=1e-13)
    np.testing.assert_array_equal(soil,np.broadcast_to(expected[:,None],soil.shape))
    root=Path(__file__).parents[1]
    paths=('scripts/validate_initial_conditions.py','scripts/run_gcm3d_ablation.py',
        'package/src/celestials/planets/mars/maps.py')
    return dict(status='pass',reference=dict(surface_git_blob=pinned_surface,patch_git_blob=PATCH_BLOB,
        block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),compiler_flags=flags,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]),
        initial_air_temperature_k=170.,latitude_cases=len(lat),soil_nodes=17,
        max_surface_error_k=float(np.max(abs(actual-expected))),
        sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths},
        limits='Unchanged surface cold-start block and pinned isothermal default; Dinosaur retains its separately tested hydrostatic pressure normalization and sigma adaptation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{report['latitude_cases']} native cold-start latitude cases passed")
