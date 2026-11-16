"""Compare the JAX soil/frost solve against unchanged Ames progts."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.soil_frost import soil_layer_thickness,surface_step,subsurface_diffusivity,subsurface_properties
from src.framework.physics.water_surface import optical_properties
from native_reference import compile_routines


SURFACE_BLOB="aca56974025103ab408ecf866cc78b25779fc469"


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
type time_type
integer :: unused=0
end type
integer :: nlayers=16,d2is=7,subsfc_ice_case=7
real :: rho_ground=1500.,cp_ground=627.9,rho_co2=910.,cp_co2=1075.
real :: rho_legacy_soil=1481.39,cp_legacy_soil=840.,rho_legacy_soilice=1781.99,cp_legacy_soilice=1404.09
real :: gk1=1000.,gk2=1000.,alpha=.5,cp_air=735.94,co2_lheat=5.9e5,stefan=5.6734e-8,pi=acos(-1.d0)
logical :: use_legacy_soil=.false.,do_subsfc_ice=.false.,do_waterice_albedo=.false.
logical :: do_moment_water=.false.,do_bulk_water=.false.,do_liquidwater_albedo=.false.
logical :: do_co2_condensation=.true.,mcpu0
real, allocatable :: delzg(:)
real :: thermal_inertia(1,1),grs_ice(1,1),sfc_roughness(1,1),sfc_albedo(1,1),sfc_emiss(1,1)
real :: sfc_frost(1,1,1)=0.,sfc_frost_mom(1,1,1)=0.,sfc_frost_blk(1,1,1)=0.
real :: albedo_ice_np=.65,albedo_ice_sp=.43,emiss_ice_np=.81,emiss_ice_sp=.88
real :: frost_threshold=1.e6,albedo_h2o=.4,albedo_h2o_liquid=.07,emiss_h2o=1.
integer, parameter :: id_ts=-1,id_subsfc=-1,id_sflux=-1,id_thin=-1,id_snow=-1,id_alb=-1
interface send_data
module procedure send_2d,send_3d
end interface
contains
"""
    adapter="""
subroutine soil_controls(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,condense,weight,soil,dz,out,ice,phase) bind(C)
integer(c_int),value :: n,condense
real(c_double),value :: dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,weight
real(c_double),intent(in) :: soil(n),dz(n+1)
real(c_double),intent(out) :: out(n),ice(1),phase(1)
do_co2_condensation=condense/=0
alpha=weight
call soil_frost(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,soil,dz,out,ice,phase)
do_co2_condensation=.true.
alpha=.5
end subroutine
subroutine soil_legacy(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,legacy,icecase,soil,dz,out,ice,phase) bind(C)
integer(c_int), value :: n,legacy,icecase
real(c_double), value :: dt,ti,pressure,snow,dn,sh,ds,emiss,latitude
real(c_double), intent(in) :: soil(n),dz(n+1)
real(c_double), intent(out) :: out(n),ice(1),phase(1)
use_legacy_soil=legacy/=0
subsfc_ice_case=icecase
do_subsfc_ice=.true.
call soil_frost(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,soil,dz,out,ice,phase)
use_legacy_soil=.false.
subsfc_ice_case=7
do_subsfc_ice=.false.
end subroutine
subroutine water_optics(t,snow,water,threshold,liquid,a,e) bind(C)
real(c_double), value :: t,snow,water,threshold
integer(c_int), value :: liquid
real(c_double), intent(out) :: a(1),e(1)
real :: ts(1,1),sn(1,1),frost(1,1),alb(1,1),em(1,1),zeros(1,1),ps(1,1),lat(1,1)
sfc_albedo=.25
sfc_emiss=.95
ts=t
sn=snow
frost=water
zeros=0.
ps=610.
lat=1.
frost_threshold=threshold
call albedo_calc(1,1,1,1,zeros,lat,zeros,ts,ps,sn,frost,alb,em,liquid/=0)
a(1)=alb(1,1)
e(1)=em(1,1)
end subroutine
subroutine soil_frost_profile(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,soil,dz,out,ice,phase) bind(C)
integer(c_int), value :: n
real(c_double), value :: dt,ti,pressure,snow,dn,sh,ds,emiss,latitude
real(c_double), intent(in) :: soil(n),dz(n+1)
real(c_double), intent(out) :: out(n),ice(1),phase(1)
do_subsfc_ice=.true.
call soil_frost(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,soil,dz,out,ice,phase)
do_subsfc_ice=.false.
end subroutine
subroutine soil_frost(n,dt,ti,pressure,snow,dn,sh,ds,emiss,latitude,soil,dz,out,ice,phase) bind(C)
integer(c_int), value :: n
real(c_double), value :: dt,ti,pressure,snow,dn,sh,ds,emiss,latitude
real(c_double), intent(in) :: soil(n),dz(n+1)
real(c_double), intent(out) :: out(n),ice(1),phase(1)
real :: ps(1,1),phalf(1,1,2),lon(1,1),lat(1,1),down(1,1),tgrnd(1,1,n),snowin(1,1)
real :: subday(1,1),tg_dt(1,1,n),shflx(1,1),dsens(1,1),zeros(1,1)
type(time_type) :: clock
nlayers=n
if(allocated(delzg)) deallocate(delzg)
allocate(delzg(n+1))
delzg=dz
thermal_inertia=ti
grs_ice=60.
sfc_roughness=.01
sfc_albedo=.25
sfc_emiss=emiss
ps=pressure
phalf(1,1,:)=[.9*pressure,pressure]
lon=0.
lat=latitude
down=dn
tgrnd(1,1,:)=soil
snowin=snow
shflx=sh
dsens=ds
zeros=0.
call progts(1,1,1,1,1,dt,clock,lon,lat,ps,phalf,down,tgrnd,snowin,subday,tg_dt, &
    shflx,zeros,dsens,zeros,zeros,zeros,zeros,zeros)
out=soil+tg_dt(1,1,:)
phase(1)=subday(1,1)
ice(1)=snow+phase(1)
end subroutine
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
logical function send_2d(id,values,clock,is,js)
integer :: id,is,js
real :: values(:,:)
type(time_type) :: clock
send_2d=.false.
end function
logical function send_3d(id,values,clock,is,js)
integer :: id,is,js
real :: values(:,:,:)
type(time_type) :: clock
send_3d=.false.
end function
end module
"""
    library,provenance=compile_routines(source/"atmos_param_mars/mars_surface.F90",
        SURFACE_BLOB,["progts","trdslv","icedepth","albedo_calc"],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags="F_CONTIGUOUS")
    native.soil_frost.argtypes=[ctypes.c_int]+[ctypes.c_double]*9+[array]*5
    native.soil_frost.restype=None
    native.soil_controls.argtypes=[ctypes.c_int]+[ctypes.c_double]*9+[ctypes.c_int,ctypes.c_double]+[array]*5
    native.soil_controls.restype=None
    native.soil_frost_profile.argtypes=native.soil_frost.argtypes
    native.soil_frost_profile.restype=None
    native.soil_legacy.argtypes=[ctypes.c_int]+[ctypes.c_double]*9+[ctypes.c_int]*2+[array]*5
    native.soil_legacy.restype=None
    native.water_optics.argtypes=[ctypes.c_double]*4+[ctypes.c_int,array,array]
    native.water_optics.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update("jax_enable_x64",True)
    cases=[]
    with tempfile.TemporaryDirectory(prefix="ames-surface-") as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for liquid in (False,True):
            for t in (200.,272.999,273.,290.):
                for snow in (-.01,0.,.01,2.,2.001):
                    for water in (1.999,2.,2.001):
                        a=np.empty(1);e=np.empty(1)
                        native.water_optics(t,snow,water,.002,int(liquid),a,e)
                        base_a=.65 if snow>2. else .25
                        base_e=.81 if snow>2. else .95
                        actual=optical_properties(base_a,base_e,t,snow,water,
                            frost_threshold_m=.002,liquid_water=liquid)
                        np.testing.assert_allclose(actual,[a[0],e[0]],rtol=0.,atol=0.)
                        cases.append(dict(case='water-surface-optics',liquid=liquid,
                            temperature_k=t,snow_kg_m2=snow,water_kg_m2=water))
        definitions=[("bare-warming",200.,0.,180.),("bare-cooling",160.,0.,0.),
            ("new-frost",148.,0.,0.),("frost-cooling",148.,100.,0.),
            ("frost-sublimation",148.,100.,100.),("frost-exhausted",148.,.001,100.),
            ("warm-precipitated-frost",180.,10.,100.)]
        for n in (4,16):
            dz=np.asfortranarray(soil_layer_thickness(n))
            for name,initial,snow,down in definitions:
                soil=np.full(n,initial,dtype=np.float64)
                result=np.empty(n);ice=np.empty(1);phase=np.empty(1)
                native.soil_frost(n,300.,250.,610.,snow,down,-2.,.5,.95,1.,soil,dz,result,ice,phase)
                emissivity=.81 if snow>2. else .95
                actual=surface_step(soil,dz,(250./(1500.*627.9))**2,1500.*627.9,
                    610.,snow,down,-2.,.5,emissivity,300.)
                np.testing.assert_allclose(actual[0],result,rtol=2e-12,atol=2e-12)
                np.testing.assert_allclose(actual[1],ice[0],rtol=2e-12,atol=2e-12)
                np.testing.assert_allclose(actual[2]['co2_phase_change_kg_m2'],phase[0],rtol=2e-12,atol=2e-12)
                if name in ('new-frost','frost-cooling'):
                    assert phase[0]>0.,name
                if name=='new-frost':
                    assert bool(actual[2]['frost_retry'])
                if name=='frost-sublimation':
                    assert phase[0]<0.
                if name=='frost-exhausted':
                    assert abs(ice[0])<1e-12
                cases.append(dict(layers=n,case=name,max_soil_error_k=float(np.max(np.abs(np.asarray(actual[0])-result))),
                    ice_error_kg_m2=float(abs(float(actual[1])-ice[0])),
                    final_surface_temperature_k=float(result[0]),co2_phase_change_kg_m2=float(phase[0]),
                    frost_retry=bool(actual[2]['frost_retry'])))
        n=16
        dz=np.asfortranarray(soil_layer_thickness(n))
        for latitude in (-1.3,-.7,0.,.7,1.3):
            soil=np.asfortranarray(np.linspace(170.,180.,n))
            result=np.empty(n);ice=np.empty(1);phase=np.empty(1)
            native.soil_frost_profile(n,300.,250.,610.,0.,100.,-2.,.5,.95,latitude,soil,dz,result,ice,phase)
            diffusivity=subsurface_diffusivity(latitude,250.,60.,n)
            actual=surface_step(soil,dz,diffusivity,1500.*627.9,610.,0.,100.,-2.,.5,.95,300.)
            np.testing.assert_allclose(actual[0],result,rtol=2e-12,atol=2e-12)
            np.testing.assert_allclose(actual[1],ice[0],rtol=2e-12,atol=2e-12)
            cases.append(dict(layers=n,case='case-7-subsurface-ice',latitude_rad=latitude,
                max_soil_error_k=float(np.max(np.abs(np.asarray(actual[0])-result))),
                ice_error_kg_m2=float(abs(float(actual[1])-ice[0]))))
        for legacy in (False,True):
            for icecase in (0,7):
                for latitude in (-1.3,-.7,0.,.7,1.3):
                    soil=np.asfortranarray(np.linspace(170.,180.,n))
                    result=np.empty(n);ice=np.empty(1);phase=np.empty(1)
                    native.soil_legacy(n,300.,250.,610.,0.,100.,-2.,.5,.95,latitude,
                        int(legacy),icecase,soil,dz,result,ice,phase)
                    diffusion,capacity=subsurface_properties(latitude,250.,60.,n,legacy=legacy,ice_case=icecase)
                    actual=surface_step(soil,dz,diffusion,capacity,610.,0.,100.,-2.,.5,.95,300.)
                    np.testing.assert_allclose(actual[0],result,rtol=2e-12,atol=2e-12)
                    np.testing.assert_allclose(actual[1],ice[0],rtol=2e-12,atol=2e-12)
                    cases.append(dict(case='soil-materials',legacy=legacy,ice_case=icecase,latitude_rad=latitude,
                        max_soil_error_k=float(np.max(np.abs(np.asarray(actual[0])-result)))))
        for n in (4,16):
            dz=np.asfortranarray(soil_layer_thickness(n))
            for condense in (False,True):
                for weight in (0.,.5,1.):
                    for snow,gradient in ((0.,False),(0.,True),(100.,True)):
                        soil=np.asfortranarray(np.linspace(148.,180. if gradient else 148.,n))
                        result=np.empty(n);ice=np.empty(1);phase=np.empty(1)
                        native.soil_controls(n,300.,250.,610.,snow,100.,-2.,.5,.95,1.,int(condense),weight,
                            soil,dz,result,ice,phase)
                        actual=surface_step(soil,dz,(250./(1500.*627.9))**2,1500.*627.9,
                            610.,snow,100.,-2.,.5,.81 if snow>2. else .95,300.,condensation=condense,alpha=weight)
                        valid=condense or snow==0.
                        assert bool(actual[2]['valid_condensation_state'])==valid
                        case=dict(case='soil-controls',layers=n,condensation=condense,alpha=weight,
                            snow_kg_m2=snow,gradient=gradient,native_finite=bool(np.isfinite(result).all()))
                        if valid:
                            np.testing.assert_allclose(actual[0],result,rtol=2e-12,atol=2e-12)
                            np.testing.assert_allclose(actual[1],ice[0],rtol=2e-12,atol=2e-12)
                            np.testing.assert_allclose(actual[2]['co2_phase_change_kg_m2'],phase[0],rtol=2e-12,atol=2e-12)
                            case['max_soil_error_k']=float(np.max(abs(np.asarray(actual[0])-result)))
                        else:
                            assert not np.isfinite(result).all()
                            assert np.isnan(actual[0]).all()
                            case['status']='singular native frost state rejected'
                        cases.append(case)
    return dict(status="pass",reference=provenance,cases=cases,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/soil_frost.py").read_bytes()).hexdigest(),
        water_surface_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/water_surface.py").read_bytes()).hexdigest(),
        vertical_solver_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/vertical_solve.py").read_bytes()).hexdigest(),
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits="uniform and case-7 subsurface-ice soil/frost columns, alpha and independent surface condensation; native disabled condensation with existing frost has a singular zero row and is rejected with nonfinite soil outputs; global climate validation remains pending")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ames-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which("gfortran")
    if compiler is None:
        parser.error("gfortran is required for the independent reference")
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
