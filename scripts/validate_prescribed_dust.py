"""Check prescribed dust profiles against unchanged released aerosol routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.prescribed_dust import layer_opacities,seasonal_fields
from native_reference import compile_routines

BLOB='8737535bd91a1cc3f1ab92366405007dcb186bae'


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer,parameter :: FATAL=1
integer :: dust_cycle_scheme=0,conrath_type=1
logical :: do_inpt_dust_cycle=.false.,add_storm_opacity=.false.,do_lat_vary_dust=.false.,mcpu0
real :: conrath=.003,optical_depth_pulse_var=0.,pulse_center=30.,pulse_width=1.
real :: areo_cycle(5)=[0.,90.,180.,270.,360.]
real :: zmax_cycle(1,1,5),taufill_cycle(1,1,5)
contains
"""
    adapter="""
subroutine dust(n,ph,p,q,lat,opacity,kind,profile,pulse,season,ls,out) bind(C)
integer(c_int),value :: n,kind,profile,season
real(c_double),value :: lat,opacity,pulse,ls
real(c_double),intent(in) :: ph(n+1),p(n),q(n)
real(c_double),intent(out) :: out(n)
real :: half(1,1,n+1),full(1,1,n),aerosol(1,1,n),dp(1,1,n),tau(1,1,n),lon(1,1),latitude(1,1)
half(1,1,:)=ph
full(1,1,:)=p
aerosol(1,1,:)=q
dp(1,1,:)=ph(2:)-ph(:n)
latitude=lat
lon=0.
conrath_type=kind
dust_cycle_scheme=profile
optical_depth_pulse_var=pulse
do_lat_vary_dust=season==0
do_inpt_dust_cycle=season/=0
taufill_cycle(1,1,:)=[.1,.2,.3,.4,.1]
zmax_cycle(1,1,:)=[20.,30.,40.,50.,20.]
if(profile>0) then
 call dust_distrib_fixed2(1,1,ls,lon,latitude,opacity,610.d0,full,half,dp,tau,aerosol)
else
 call dust_distrib_fixed2(1,1,ls,lon,latitude,opacity,610.d0,full,half,dp,tau)
endif
out=tau(1,1,:)
end subroutine
subroutine error_mesg(where,message,level)
character(*) where,message
integer level
stop 1
end subroutine
subroutine add_storm(ls,lon,lat,out)
real :: ls,lon(:,:),lat(:,:),out(:,:)
out=0.
end subroutine
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
end module
"""
    library,provenance=compile_routines(source/'atmos_param_mars/aerosol.F90',BLOB,
        ['dust_distrib_fixed2','dust_distrib_pulse2','season_table'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.dust.argtypes=[ctypes.c_int,array,array,array,ctypes.c_double,ctypes.c_double,
        ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_double,array]
    native.dust.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='dust-native-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (8,12,40):
            ph=np.geomspace(.01,610.,n+1);p=np.sqrt(ph[:-1]*ph[1:])
            q=np.asfortranarray(np.linspace(.1,1.,n));expected=np.empty(n)
            run=jax.jit(layer_opacities,static_argnames=('conrath_type','cycle_scheme','latitude_variation'))
            for kind in (0,1):
                for profile in (0,1,2):
                    for lat in (-1.2,.4):
                        for pulse in (0.,.2):
                            for season in (0,1):
                                ls=355.
                                native.dust(n,ph,p,q,lat,.15,kind,profile,pulse,season,ls,expected)
                                options={}
                                if season:
                                    opacity,height=seasonal_fields(ls,np.array([0.,90.,180.,270.,360.]),
                                        np.array([.1,.2,.3,.4,.1]),np.array([20.,30.,40.,50.,20.]))
                                    options=dict(scenario_opacity=opacity,top_height_km=height)
                                actual=run(ph,p,.15,latitude_rad=lat,conrath_type=kind,
                                    cycle_scheme=profile,aerosol_ratio=q if profile else None,
                                    latitude_variation=not bool(season),pulse_fraction=pulse,**options)
                                np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-14)
                                cases.append(dict(layers=n,conrath_type=kind,cycle_scheme=profile,latitude_rad=lat,
                                    pulse_fraction=pulse,seasonal=bool(season),max_error=float(np.max(np.abs(actual-expected)))))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_prescribed_dust.py','package/src/framework/physics/prescribed_dust.py')},
        limits='profile, fractional pulse and seasonal interpolation kernels; map sampling, optical properties and storm switches have separate checks')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} prescribed dust columns passed")


if __name__=='__main__': main()
