"""Compare orbital settings and solar geometry with the pinned astronomy patch."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
from src.framework.physics.gcm import orbital_distance,_true_anomaly,_declination
from src.framework.physics.solar_geometry import daylight_geometry,averaged_zenith
from native_reference import compile_routines

PATCH_BLOB='6deea614300cc5340dfd5bd02fc2d5ddc1bdebee'


def blob(raw):
    return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    raw=(source/'patches/srcFMS.patch').read_bytes()
    if blob(raw)!=PATCH_BLOB:
        raise ValueError('astronomy patch differs from pinned v3.2.1')
    added='\n'.join(line[1:] for line in raw.decode().splitlines()
        if line.startswith('+') and not line.startswith('+++')).encode()
    header="""module reference_mod
use iso_c_binding
implicit none
real, parameter :: pi=acos(-1.d0),radian=180.d0/pi
real :: obliquity=25.19d0,eccentricity=.093d0,equinox_fraction=.269461d0
real :: days_in_year=668.d0,zero_date=488.d0
contains
"""
    adapter="""
subroutine orbit(day,out) bind(C)
real(c_double), value :: day
real(c_double), intent(out) :: out(3)
integer :: whole
real :: fraction
whole=floor(day)
fraction=day-whole
call mars_calender(whole,fraction,out(1),out(2),out(3))
end subroutine
subroutine geometry(n,fjd,decl,arg,lon,lat,hang,mean,frac,cosz,daylight) bind(C)
integer(c_int), value :: n
real(c_double), value :: fjd,decl,arg
real(c_double), intent(in) :: lon(n,1),lat(n,1)
real(c_double), intent(out) :: hang(n,1),mean(n,1),frac(n,1),cosz(n,1),daylight(n,1)
call sol88(decl,lat,hang,mean,frac)
call zenith(fjd,decl,0.d0,lon,lat,hang,arg,cosz,daylight)
end subroutine
end module
"""
    orbit_errors=[];geometry_errors=[]
    with tempfile.TemporaryDirectory(prefix='solar-geometry-') as temporary:
        root=Path(temporary)
        path=root/'added.F90';path.write_bytes(added)
        lib,provenance=compile_routines(path,blob(added),['mars_calender','sol88','zenith'],
            header,adapter,root,compiler)
        provenance['source_patch_git_blob']=PATCH_BLOB
        provenance['source_patch_sha256']=hashlib.sha256(raw).hexdigest()
        native=ctypes.CDLL(str(lib))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
        native.orbit.argtypes=[ctypes.c_double,array];native.orbit.restype=None
        native.geometry.argtypes=[ctypes.c_int]+[ctypes.c_double]*3+[array]*7
        native.geometry.restype=None
        forcing=reference_radiative_forcing()
        for day in np.linspace(0.,3.*668.,37):
            expected=np.empty(3)
            native.orbit(day,expected)
            time=day*88704.
            actual=np.array([orbital_distance(time,forcing)/forcing.semi_major_axis_m,
                _declination(time,forcing),_true_anomaly(time,forcing)+forcing.ls_perihelion_rad])
            error=actual-expected
            error[2]=np.arctan2(np.sin(error[2]),np.cos(error[2]))
            np.testing.assert_allclose(error,0.,atol=3e-12,rtol=0.)
            orbit_errors.append(float(np.max(np.abs(error))))
        latitudes=np.concatenate([np.linspace(-1.55,1.55,31),[-np.pi*.5,np.pi*.5]])
        longitude,latitude=np.meshgrid(np.linspace(0.,2.*np.pi,33,endpoint=False),latitudes,indexing='ij')
        longitude=np.asfortranarray(longitude.ravel());latitude=np.asfortranarray(latitude.ravel())
        for decl in (0.,-.44,.44):
            for fjd in (0.,.25,.55):
                for arg in (.01,.3,2.):
                    expected=[np.zeros(longitude.size) for _ in range(5)]
                    native.geometry(longitude.size,fjd,decl,arg,longitude,latitude,*expected)
                    actual=(*daylight_geometry(decl,latitude),*averaged_zenith(fjd,decl,longitude,latitude,arg))
                    errors=[]
                    for value,ref in zip(actual,expected,strict=True):
                        np.testing.assert_allclose(value,ref,atol=2e-12,rtol=2e-12)
                        errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                    geometry_errors.append(dict(declination=decl,day_fraction=fjd,half_angle=arg,max_errors=errors))
    repo=Path(__file__).parents[1]
    paths=['scripts/validate_solar_geometry.py','package/src/framework/physics/solar_geometry.py',
        'package/src/celestials/planets/mars/reference_constants.py','package/src/framework/physics/gcm.py']
    return dict(status='pass',reference=provenance,orbit_cases=len(orbit_errors),
        max_orbit_error=max(orbit_errors),geometry_cases=geometry_errors,
        sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
        limits='periodic orbital phase and sol88/zenith; reference longwave update scheduling and spin-up are separate')


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
    print(json.dumps(dict(status=report['status'],orbit_cases=report['orbit_cases'],geometry_cases=len(report['geometry_cases']))))
