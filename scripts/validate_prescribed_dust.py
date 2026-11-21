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
from src.framework.physics.prescribed_dust import layer_opacities,seasonal_fields,storm_opacity
from native_reference import compile_routines,extract_routines

BLOB='8737535bd91a1cc3f1ab92366405007dcb186bae'


def compile_reference(source,destination,compiler):
    routines,_=extract_routines(source/'atmos_param_mars/aerosol.F90',BLOB,['add_storm'])
    routine=routines[0];start=routine.index('jd= size(lon,2)')
    storm=routine[start:routine.index('\nreturn',start)]
    header="""module reference_mod
use iso_c_binding
implicit none
integer,parameter :: FATAL=1
integer :: dust_cycle_scheme=0,conrath_type=1
integer,parameter :: nchan_sw=1,nchan_lw=5
integer :: dust_indx(1)=[1],inpt_dust_cycle_dust_index=1
logical :: do_inpt_dust_cycle_ktop=.false.
real :: optical_depth_inpt=.15,optical_depth_pulse=0.,scale_inpt_dust_column=1.
real :: qext_cnst=2.5,sscat_cnst=.9,gfac_cnst=.65
real :: qex_lw_dust(6)=1.,sscat_lw_dust(6)=0.,gfac_lw_dust(6)=0.
logical :: do_inpt_dust_cycle=.false.,add_storm_opacity=.false.,do_lat_vary_dust=.false.,mcpu0
real :: conrath=.003,optical_depth_pulse_var=0.,pulse_center=30.,pulse_width=1.
real :: areo_cycle(5)=[0.,90.,180.,270.,360.]
real :: zmax_cycle(1,1,5),taufill_cycle(1,1,5)
real,parameter :: pi=acos(-1.d0)
real :: storm_sigx=60.,storm_sigy=20.,storm_lonc=10.,storm_latc=15.,add_storm_amp=0.,storm_areo=180.
real :: supplied_slope,supplied_c
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
subroutine fixed_driver(n,ph,p,q,lat,opacity,kind,profile,fraction,absolute,scaling,season,ls,out) bind(C)
integer(c_int),value :: n,kind,profile,season
real(c_double),value :: lat,opacity,fraction,absolute,scaling,ls
real(c_double),intent(in) :: ph(n+1),p(n),q(n)
real(c_double),intent(out) :: out(n)
real :: half(1,1,n+1),full(1,1,n),aerosol(1,1,n,1),dp(1,1,n),ref(1,1,n)
real :: tau(1,1,n,7),scat(1,1,n,7),moment(1,1,n,7),lon(1,1),latitude(1,1),dummy(n)
add_storm_opacity=.false.
call dust(n,ph,p,q,lat,opacity,kind,profile,fraction,season,ls,dummy)
optical_depth_inpt=opacity;optical_depth_pulse=absolute;scale_inpt_dust_column=scaling
do_inpt_dust_cycle_ktop=profile>0
half(1,1,:)=ph;full(1,1,:)=p;aerosol(1,1,:,1)=q
dp(1,1,:)=ph(2:)-ph(:n);lon=0.;latitude=lat
call dust_distrib_fix(1,1,ls,lon,latitude,610.d0,full,half,dp,aerosol,tau,scat,moment,ref)
out=ref(1,1,:)
end subroutine
subroutine error_mesg(where,message,level)
character(*) where,message
integer level
stop 1
end subroutine
subroutine configure_storm(enabled,parameters) bind(C)
integer(c_int),value :: enabled
real(c_double),intent(in) :: parameters(8)
add_storm_opacity=enabled/=0
add_storm_amp=parameters(1);storm_sigx=parameters(2);storm_sigy=parameters(3)
storm_lonc=parameters(4);storm_latc=parameters(5);storm_areo=parameters(6)
supplied_slope=parameters(7);supplied_c=parameters(8)
end subroutine
subroutine storm_field(nx,ny,longitude,latitude,ls,out) bind(C)
integer(c_int),value :: nx,ny
real(c_double),value :: ls
real(c_double),intent(in) :: longitude(nx),latitude(ny)
real(c_double),intent(out) :: out(nx,ny)
real :: lon(nx,ny),lat(nx,ny)
integer :: i,j
do j=1,ny
 lon(:,j)=longitude
enddo
do i=1,nx
 lat(i,:)=latitude
enddo
call add_storm(ls,lon,lat,out)
end subroutine
subroutine add_storm(areolat,lon,lat,optical_depth)
real :: areolat,lon(:,:),lat(:,:),optical_depth(:,:)
integer :: id,jd
real :: anom(size(lat,1),size(lat,2)),aa(size(lat,1),size(lat,2)),aa2(size(lat,1),size(lat,2))
real :: slope,tpi,rt2,sigma_x,sigma_y,theta,a,b,c,lonc,lonc2,latc
slope=supplied_slope;c=supplied_c
"""+storm+"""
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
        ['dust_distrib_fixed2','dust_distrib_pulse2','season_table','dust_distrib_fix','dust_distrib_pulse'],header,adapter,destination,compiler)
    provenance['storm_block_sha256']=hashlib.sha256(storm.encode()).hexdigest()
    provenance['storm_adaptation']='Unchanged numerical add_storm block; adapter explicitly supplies source-uninitialized slope and c. Raw released storm output is undefined.'
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.dust.argtypes=[ctypes.c_int,array,array,array,ctypes.c_double,ctypes.c_double,
        ctypes.c_int,ctypes.c_int,ctypes.c_double,ctypes.c_int,ctypes.c_double,array]
    native.dust.restype=None
    native.fixed_driver.argtypes=[ctypes.c_int,array,array,array,ctypes.c_double,ctypes.c_double,
        ctypes.c_int,ctypes.c_int]+[ctypes.c_double]*3+[ctypes.c_int,ctypes.c_double,array]
    native.fixed_driver.restype=None
    native.configure_storm.argtypes=[ctypes.c_int,array];native.configure_storm.restype=None
    native.storm_field.argtypes=[ctypes.c_int]*2+[array]*2+[ctypes.c_double,array]
    native.storm_field.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    cases=[];storm_cases=[];driver_cases=[]
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
        longitude=np.array([-.1,0.,.2,np.pi,2.*np.pi-.1,2.*np.pi,2.*np.pi+.1])
        latitude=np.array([-np.pi/2.,-.4,0.,.4,np.pi/2.])
        for ls in (179.,180.,181.):
            for center in (10.,200.):
                for slope in (0.,12.):
                    for amplitude in (0.,.4):
                        options=dict(slope_deg=slope,latitude_coefficient=5.,amplitude=amplitude,
                            center_longitude_deg=center)
                        native.configure_storm(1,np.array([amplitude,60.,20.,center,15.,180.,slope,5.]))
                        expected=np.empty((len(longitude),len(latitude)),order='F')
                        native.storm_field(len(longitude),len(latitude),longitude,latitude,ls,expected)
                        actual=storm_opacity(ls,longitude[:,None],latitude[None,:],**options)
                        np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-15)
                        storm_cases.append(dict(solar_longitude_deg=ls,**options,
                            max_error=float(np.max(abs(actual-expected)))))
        ph=np.geomspace(.01,610.,9);p=np.sqrt(ph[:-1]*ph[1:]);q=np.ones(8);expected=np.empty(8)
        for seasonal in (False,True):
            for kind in (0,1):
                for pulse in (0.,.2):
                    native.dust(8,ph,p,q,.4,.15,kind,0,pulse,int(seasonal),180.,expected)
                    options={}
                    if seasonal:
                        options=dict(scenario_opacity=.3+storm_opacity(180.,0.,.4,
                            slope_deg=12.,latitude_coefficient=5.,amplitude=.4,center_longitude_deg=200.),top_height_km=40.)
                    actual=layer_opacities(ph,p,.15,latitude_rad=.4,conrath_type=kind,
                        latitude_variation=not seasonal,pulse_fraction=pulse,**options)
                    np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-14)
                    storm_cases.append(dict(case='composed-profile',seasonal=seasonal,conrath_type=kind,pulse_fraction=pulse))
        for n,surface_pressure in ((8,400.),(40,900.)):
            ph=np.geomspace(.01,surface_pressure,n+1);p=np.sqrt(ph[:-1]*ph[1:])
            q=np.asfortranarray(np.linspace(.1,1.,n));expected=np.empty(n)
            for kind in (0,1):
                for profile in (0,1,2):
                    for lat in (-1.2,.4):
                        for season in (0,1):
                            options={}
                            if season:
                                opacity,height=seasonal_fields(355.,np.array([0.,90.,180.,270.,360.]),
                                    np.array([.1,.2,.3,.4,.1]),np.array([20.,30.,40.,50.,20.]))
                                options=dict(scenario_opacity=opacity,top_height_km=height)
                            for fraction,absolute,scaling in ((0.,0.,1.),(.2,0.,1.),(0.,.1,2.),(.2,.1,2.),(0.,.1,0.)):
                                native.fixed_driver(n,ph,p,q,lat,.15,kind,profile,fraction,absolute,scaling,season,355.,expected)
                                actual=run(ph,p,.15,latitude_rad=lat,conrath_type=kind,cycle_scheme=profile,
                                    aerosol_ratio=q if profile else None,latitude_variation=not bool(season),
                                    pulse_fraction=fraction,pulse_opacity=absolute,column_scale=scaling,**options)
                                np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-14)
                                driver_cases.append(dict(layers=n,surface_pressure_pa=surface_pressure,
                                    conrath_type=kind,cycle_scheme=profile,latitude_rad=lat,seasonal=bool(season),
                                    pulse_fraction=fraction,pulse_opacity=absolute,column_scale=scaling,
                                    max_error=float(np.max(abs(np.asarray(actual)-expected)))))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,cases=cases,storm_cases=storm_cases,driver_cases=driver_cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_prescribed_dust.py','package/src/framework/physics/prescribed_dust.py')},
        limits='Profile, fractional pulse and seasonal interpolation kernels plus unchanged prescribed-dust dispatch with additive pulse and column scaling. Storm checks compare unchanged equations with explicit supplied slope and c; the release leaves those locals undefined. Map sampling and optical properties have separate checks.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} profiles, {len(report['storm_cases'])} explicit storms and {len(report['driver_cases'])} composed prescribed-dust cases passed")


if __name__=='__main__': main()
