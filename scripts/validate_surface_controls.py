"""Check released surface, tracer-freeze, no-forcing and dust-gust controls."""
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
    start=text.index('if (freeze_tracer_fields) then')
    freeze=text[start:text.index('endif',start)+len('endif')]
    start=text.index('uwnd = u + udt * dt')
    gust=text[start:text.index('if (id_stress',start)]
    start=text.index('if (no_forcing) then')
    unforced=text[start:text.index('    return\nendif',start)+len('    return\nendif')]
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
'''
    adapter+='''
subroutine freeze_fields(levels,species,enabled,rdt) bind(C)
integer(c_int),value :: levels,species,enabled
real(c_double),intent(inout) :: rdt(1,1,levels,species)
integer :: n,ntrace
logical :: freeze_tracer_fields
ntrace=species
freeze_tracer_fields=enabled/=0
'''+freeze+'''
end subroutine
subroutine stress_control(dt,factor,drag_value,gust_value,uu,vv,du,dv,out) bind(C)
real(c_double),value :: dt,factor,drag_value,gust_value,uu,vv,du,dv
real(c_double),intent(out) :: out(1,1)
integer,parameter :: kd=1
real :: u(1,1,1),v(1,1,1),udt(1,1,1),vdt(1,1,1),uwnd(1,1,1),vwnd(1,1,1)
real :: wind(1,1),stress(1,1),dragm(1,1),gust(1,1),gust_fact
u=uu;v=vv;udt=du;vdt=dv;dragm=drag_value;gust=gust_value;gust_fact=factor
'''+gust+'''
out=stress
end subroutine
subroutine unforced_control(n,enabled,diffusion,udt,vdt,tdt,rdt,reached) bind(C)
integer(c_int),value :: n,enabled,diffusion
real(c_double),intent(inout) :: udt(1,1,n),vdt(1,1,n),tdt(1,1,n),rdt(1,1,n,2)
integer(c_int),intent(out) :: reached
integer,parameter :: is=1,js=1,id=1,jd=1,ntp=2
integer :: kd,kbot(1,1)
real :: dt,lon(1,1),lat(1,1),dnflux(1,1),tsurf(1,1),p_half(1,1,n+1),p_full(1,1,n)
real :: z_half(1,1,n+1),z_full(1,1,n),u(1,1,n),v(1,1,n),t(1,1,n),r(1,1,n,2),mask(1,1,n)
logical :: no_forcing,do_vert_diff
type(time_type) :: Time
kd=n;dt=5.;lon=0.;lat=0.;kbot=n;mask=1.
p_half=610.;p_full=500.;z_half=0.;z_full=1.;u=1.;v=2.;t=180.;r=.001
reached=0
no_forcing=enabled/=0;do_vert_diff=diffusion/=0
'''+unforced+'''
reached=1
end subroutine
end module
'''
    with tempfile.TemporaryDirectory(prefix='surface-controls-native-') as temporary:
        library,provenance=compile_routines(path,PHYSICS_BLOB,['diffuse'],header,adapter,Path(temporary),compiler)
        native=ctypes.CDLL(str(library))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
        native.boundary.argtypes=[ctypes.c_int]+[array]*3;native.boundary.restype=None
        native.fallback.argtypes=[ctypes.c_int]+[array]*4;native.fallback.restype=None
        native.freeze_fields.argtypes=[ctypes.c_int]*3+[array];native.freeze_fields.restype=None
        native.stress_control.argtypes=[ctypes.c_double]*8+[array];native.stress_control.restype=None
        native.unforced_control.argtypes=[ctypes.c_int]*3+[array]*4+[ctypes.POINTER(ctypes.c_int)]
        native.unforced_control.restype=None
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
        process_cases=[]
        for n in (2,4,40,56):
            for enabled in (False,True):
                tendencies=np.asfortranarray(np.linspace(-1.,2.,n*9).reshape(1,1,n,9))
                before=tendencies.copy()
                native.freeze_fields(n,9,int(enabled),tendencies)
                np.testing.assert_array_equal(tendencies,0. if enabled else before)
                process_cases.append(dict(process='freeze',levels=n,enabled=enabled))
                for diffusion in (False,True):
                    arrays=[np.full((1,1,n),.123,order='F') for _ in range(3)]
                    arrays.append(np.full((1,1,n,2),-.01,order='F'))
                    before=[a.copy() for a in arrays];reached=ctypes.c_int(-1)
                    native.unforced_control(n,int(enabled),int(diffusion),*arrays,ctypes.byref(reached))
                    assert reached.value==int(not enabled)
                    for actual,expected in zip(arrays,before,strict=True): np.testing.assert_array_equal(actual,expected)
                    process_cases.append(dict(process='no_forcing',levels=n,enabled=enabled,diffusion=diffusion))
        from src.framework.gcm._dinosaur import jax,jnp
        from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as body
        from src.framework.physics.column_physics import column_physics
        from src.framework.physics.soil_frost import soil_layer_thickness
        from src.framework.physics.dust_lifting import stress_flux
        jax.config.update('jax_enable_x64',True)
        for n in (4,40,56):
            half=jnp.linspace(0.,610.,n+1);full=.5*(half[:-1]+half[1:])
            kwargs=dict(temperature=jnp.full(n,200.),u=jnp.linspace(30.,5.,n),v=jnp.linspace(-5.,-1.,n),
                gas_tracers=jnp.zeros((n,1)),soil=jnp.full(16,240.),pressure_half=half,pressure_full=full,
                co2_ice_kg_m2=0.,radiative_heating_k_s=jnp.zeros(n),downward_surface_flux_w_m2=0.,dt_seconds=5.,
                body=body,soil_thickness=soil_layer_thickness(16),soil_diffusivity=(250./(1500.*627.9))**2,
                soil_volumetric_heat_capacity=1500.*627.9,emissivity=.81,surface_exchange_start=0,convection_scheme='none',
                moment_particles=jnp.broadcast_to(jnp.array([1e-6,1e8]),(n,2)),moment_water=False,
                surface_dust_kg_m2=30.,dust_lifting_opacity=.2,
                dust_lifting_options=dict(threshold_stress=1e-6,convective_lift=False))
            quiet=column_physics(**kwargs)
            pbl=quiet.diagnostics['pbl']
            drag=610./(body.gas_constant_j_kg_k*200.)*pbl['surface_momentum_coefficient']*pbl['wind_speed_ms']
            for factor in (0.,1.,25.,100.,-1e6):
                stress=np.empty((1,1),order='F')
                native.stress_control(5.,factor,float(drag),1.,float(quiet.u[-1]),float(quiet.v[-1]),0.,0.,stress)
                expected=5.*stress_flux(stress[0,0],1e-6,float(full[-1])/(body.gas_constant_j_kg_k*quiet.temperature[-1]),
                    0.,quiet.co2_ice_kg_m2,0.,.2/full[-1]*610.)
                actual=column_physics(**kwargs,gust_factor=factor)
                np.testing.assert_allclose(actual.diagnostics['dust']['dust_lifting']['stress_lift_kg_m2'],expected,rtol=3e-13)
                np.testing.assert_allclose(actual.temperature,quiet.temperature,rtol=0.,atol=0.)
                process_cases.append(dict(process='gust',levels=n,gust_factor=factor))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_surface_controls.py','package/src/framework/physics/boundary_layer.py',
        'package/src/framework/physics/column_physics.py','package/src/framework/physics/dinosaur_physics.py',
        'package/src/framework/physics/dust_lifting.py','package/src/celestials/planets/mars/reference_constants.py']
    provenance['fixed_boundary_block_sha256']=hashlib.sha256(fixed.encode()).hexdigest()
    provenance['process_block_sha256']={name:hashlib.sha256(value.encode()).hexdigest()
        for name,value in (('freeze',freeze),('gust',gust),('no_forcing',unforced))}
    return dict(status='pass',reference=provenance,fixed_boundary=dict(temperature_k=170.,albedo=.25,emissivity=1.),
        cases=cases,process_cases=process_cases,sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Unchanged fixed-boundary, tracer-freeze, no-forcing and gust blocks; original diffuse compiled with released flags leaves tendency outputs unchanged because fv3_turb is excluded. Gust=1 is the released initialized value; alternate gust drivers are excluded. Freeze routing and the native early return are checked with seeded tendencies; coupled column/global checks cover physical reservoirs, histories, remapping and derivatives separately. Inactive PBL exchange fields are undefined in source; the port defines them as zero. Surface-dependent water/dust configurations without initialized reservoirs are rejected.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),process_cases=len(report['process_cases']))))
