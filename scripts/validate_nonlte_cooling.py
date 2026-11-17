"""Compare non-LTE cooling with unchanged pinned Fortran routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.nonlte_cooling import cooling,reference_atmosphere,interpolate_profile
from src.framework.physics.radiation_grid import radiation_grid
from native_reference import compile_routines

BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer :: l_levels,ntrace=6,MODEL_ATMOS=1
integer, parameter :: nlnlte=70
real*8 :: pres_mgref_day(70),tn_mgref_day(70),no_mgref_day(70),nco2_mgref_day(70)
real*8 :: nco_mgref_day(70),nn2_mgref_day(70),q15_mgref_day(70)
contains
"""
    adapter="""
subroutine native_cooling(n,p,t,q,ref,out) bind(C)
integer(c_int), value :: n
real(c_double), intent(in) :: p(2*n+3),t(2*n+3),q(n,6),ref(70,8)
real(c_double), intent(out) :: out(n)
real :: player(2*n+3),temp(2*n+3),result(2*n+3)
integer k
l_levels=2*n+3
pres_mgref_day=ref(:,2)
tn_mgref_day=ref(:,3)
nco2_mgref_day=ref(:,4)
no_mgref_day=ref(:,5)
nco_mgref_day=ref(:,6)
nn2_mgref_day=ref(:,7)
q15_mgref_day=ref(:,8)
player=p*.01
temp=t
call simple_nltecool(n,player,q,temp,result)
out=result(4:2*n+2:2)
end subroutine
subroutine native_interp(n,x,y,m,q,out) bind(C)
integer(c_int), value :: n,m
real(c_double), intent(in) :: x(n),y(n),q(m)
real(c_double), intent(out) :: out(m)
call interpnlte(n,x,y,m,q,out)
end subroutine
integer function find_field_index(model,name)
integer model
character(*) name
select case(name)
case('vap_mass_mom')
find_field_index=1
case('co2_mmr_gas')
find_field_index=2
case('co_mmr_gas')
find_field_index=3
case('o_mmr_gas')
find_field_index=4
case('o2_mmr_gas')
find_field_index=5
case('n2_mmr_gas')
find_field_index=6
end select
end function
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
end module
"""
    library,provenance=compile_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',
        BLOB,['simple_nltecool','interpnlte'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.native_cooling.argtypes=[ctypes.c_int]+[array]*5
    native.native_cooling.restype=None
    native.native_interp.argtypes=[ctypes.c_int,array,array,ctypes.c_int,array,array]
    native.native_interp.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    ref=reference_atmosphere()
    names=('altitude_m','pressure_mbar','temperature_k','co2_m3','o_m3','co_m3','n2_m3','cooling_k_s')
    atmosphere=np.asfortranarray(np.stack([ref[name] for name in names],axis=-1))
    cases=[]
    with tempfile.TemporaryDirectory(prefix='nonlte-native-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        x=np.array([-8.,-5.,-2.]);y=np.array([1.,10.,2.])
        query=np.array([-10.,-7.,0.,-4.,-2.,4.]);out=np.empty(6)
        native.native_interp(3,x,y,6,query,out)
        np.testing.assert_allclose(interpolate_profile(query,x,y),out,rtol=2e-15,atol=1e-15)
        for n in (4,12,40):
            run=jax.jit(cooling)
            for bounds in ((1.e-8,4.),(.001,610.)):
                p=np.asfortranarray(np.geomspace(*bounds,n))
                for temperature in (110.,180.,350.):
                    t=np.full(n,temperature);q=np.zeros((n,6),order='F')
                    for oxygen in (0.,.01,.3):
                        q[:,1]=.9-oxygen;q[:,2]=.01;q[:,3]=oxygen;q[:,4]=.01;q[:,5]=.08
                        expected=np.empty(n)
                        half=np.concatenate([p[:1]*.1,np.sqrt(p[:-1]*p[1:]),p[-1:]*1.1])
                        grid=radiation_grid(half,p,t,temperature)
                        pressure=np.asfortranarray(grid.pressure);temp=np.asfortranarray(grid.temperature)
                        native.native_cooling(n,pressure,temp,q,atmosphere,expected)
                        actual=run(t,p,q[:,1],q[:,3],radiation_pressure_pa=pressure)
                        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=2e-15)
                        cases.append(dict(layers=n,pressure_bounds_pa=bounds,temperature_k=temperature,
                            oxygen_mass_fraction=oxygen,max_error_k_s=float(np.max(np.abs(actual-expected)))))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_nonlte_cooling.py','scripts/stage_nonlte_reference.py',
             'package/src/framework/physics/nonlte_cooling.py','package/src/framework/physics/nonlte_reference.npz')},
        limits='preserves integer 1/3 exponent and out-of-range interpolation history; longwave correction is excluded by the reference RELEASE flag')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} non-LTE columns passed")


if __name__=='__main__': main()
