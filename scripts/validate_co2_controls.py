"""Check CO2 switch routing against the unchanged released control block.

Phase-change kernels have separate native validators. This check seeds their
outputs, then executes the original heating, precipitation and mass routing.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp
from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY
from src.framework.physics.column_physics import column_physics
from src.framework.physics.co2_condensation import surface_exchange_start
from src.framework.physics.soil_frost import soil_layer_thickness

PHYSICS_BLOB='ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc'


def compile_reference(source,directory,compiler):
    raw=(source/'atmos_param_mars/mars_physics.F90').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=PHYSICS_BLOB: raise ValueError('reference physics differs from pinned source')
    text=raw.decode();start=text.index('    if (do_co2_condensation) then')
    block=text[start:text.index('!***************** IF DO_SURFACE_MARS = FALSE',start)]
    start=text.rfind('do k = kd,1,-1',0,text.index('klevs_trop = kd-k+1'))
    selection=text[start:text.index('! *********************************************************',start)]
    header='''module reference_mod
use iso_c_binding
implicit none
real,allocatable :: seed_heat(:,:,:),seed_mass(:,:,:),seed_precip(:,:)
contains
subroutine co2_condense(is,ie,js,je,kd,dt,Time,t,tdt,ph,p,precip,dmass,tdt_co2)
integer :: is,ie,js,je,kd,Time
real :: dt,t(:,:,:),tdt(:,:,:),ph(:,:,:),p(:,:,:),precip(:,:),dmass(:,:,:),tdt_co2(:,:,:)
precip=seed_precip
dmass=seed_mass
tdt_co2=seed_heat
end subroutine
subroutine controls(n,flags,start_layer,ps,initial_t,phase_heat,phase_mass,phase_precip,surface_phase,co2,tout,mass,snow) bind(C)
integer(c_int),value :: n,start_layer
integer(c_int),intent(in) :: flags(4)
real(c_double),intent(in) :: ps,initial_t(1,1,n),phase_heat(1,1,n),phase_mass(1,1,n)
real(c_double),intent(in) :: phase_precip,surface_phase,co2(1,1,n)
real(c_double),intent(out) :: tout(1,1,n),mass(1,1,n),snow
integer,parameter :: is=1,ie=1,js=1,je=1,ntrace=2,nco2=1,id_precip=0,Time=0
integer :: kd,klev,klevs_co2,k
real,parameter :: dt=5.
logical :: do_co2_condensation,form_co2_clouds,do_atmos_co2_loss,do_co2_condensation_cycle,do_qmass,checkcons,used
logical,external :: send_data
real :: t(1,1,n),tdt(1,1,n),p_half(1,1,n+1),p_full(1,1,n),delp(1,1,n)
real :: r(1,1,n,2),rdt(1,1,n,2),rdt_co2micro(1,1,n,2),tdt_co2(1,1,n)
real :: dmass(1,1,n),precip(1,1),sfc_snow(1,1),subday(1,1),warray(1,1),mratio(1,1)
real :: stress(1,1),tsurf(1,1),rhouch(1,1),rkh(1,1,n)
kd=n
klevs_co2=n-start_layer-2
do_co2_condensation=flags(1)/=0
do_atmos_co2_loss=flags(2)/=0
do_co2_condensation_cycle=flags(3)/=0
do_qmass=flags(4)/=0
form_co2_clouds=.false.
checkcons=.false.
t=initial_t
tdt=0.
tdt_co2=0.
rdt=0.
r(:,:,:,1)=co2
r(:,:,:,2)=1.-co2
do k=1,n+1
 p_half(:,:,k)=ps*real(k-1)/n
enddo
delp=p_half(:,:,2:n+1)-p_half(:,:,1:n)
p_full=.5*(p_half(:,:,2:n+1)+p_half(:,:,1:n))
subday=surface_phase
sfc_snow=0.
allocate(seed_heat(1,1,n),seed_mass(1,1,n),seed_precip(1,1))
seed_heat=phase_heat
seed_mass=phase_mass
seed_precip=phase_precip
'''
    tail='''
tout=t+dt*tdt
mass=dmass
snow=sfc_snow(1,1)+surface_phase
deallocate(seed_heat,seed_mass,seed_precip)
end subroutine
end module
subroutine co2micro_driver(is,ie,js,je,kd,ntrace,Time,ph,p,t,tdt,r,rdt,stress,tsurf,dt,rhouch,rdt_micro,precip,dmass,rkh,tdt_co2,checkcons)
integer :: is,ie,js,je,kd,ntrace,Time
real :: ph(*),p(*),t(*),tdt(*),r(*),rdt(*),stress(*),tsurf(*),dt,rhouch(*),rdt_micro(*),precip(*),dmass(*),rkh(*),tdt_co2(*)
logical :: checkcons
error stop 'cloud phase kernels use their separate native validator'
end subroutine
logical function send_data(id,value,Time,is,js)
integer :: id,Time,is,js
real :: value(*)
send_data=.false.
end function
'''
    depth='''
subroutine exchange_depth(n,ncol,qmass,polar,tropical,pstd,lat,start_layer) bind(C)
integer(c_int),value :: n,ncol,qmass
real(c_double),value :: polar,tropical
real(c_double),intent(in) :: pstd(n),lat(ncol,1)
integer(c_int),intent(out) :: start_layer
integer :: kd,k,klevs_trop,klevs_pole,klevs_co2
integer,parameter :: is=1,js=1,je=1
integer :: ie
real,parameter :: PI=acos(-1.d0)
real :: klevs_ratio_pole,klevs_ratio_trop
logical :: do_qmass,mcpu0
kd=n;ie=ncol;do_qmass=qmass/=0;mcpu0=.false.
klevs_ratio_pole=polar;klevs_ratio_trop=tropical
klevs_trop=-999;klevs_pole=-999
'''+selection+'''
start_layer=n-klevs_co2-2
end subroutine
'''
    tail=tail.replace('\nend module\n','\n'+depth+'end module\n',1)
    code=header+block+tail
    path=directory/'reference.f90';path.write_text(code)
    flags=['-O3','-fdefault-real-8','-fdefault-double-8','-ffree-line-length-none','-fPIC','-shared']
    subprocess.run([compiler,*flags,str(path),'-o',str(directory/'reference.so')],cwd=directory,check=True)
    native=ctypes.CDLL(str(directory/'reference.so'))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.controls.argtypes=[ctypes.c_int,integers,ctypes.c_int]+[array]*10
    native.controls.restype=None
    native.exchange_depth.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*2+[array]*2+[ctypes.POINTER(ctypes.c_int)]
    native.exchange_depth.restype=None
    provenance=dict(source_git_blob=blob,block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        exchange_depth_block_sha256=hashlib.sha256(selection.encode()).hexdigest(),
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),compiler_flags=flags,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        adaptation='Unchanged control block; phase output seeds and disabled diagnostic adapters. Suspended-cloud phase kernels are validated separately.')
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    f=lambda value:np.asfortranarray(value,dtype=np.float64)
    layout=lambda value:f(np.asarray(value)[None,None,:])
    cases=[];depth_cases=[]
    with tempfile.TemporaryDirectory(prefix='co2-controls-native-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (4,12,40,56):
            for geometry in ('uniform','geometric'):
                pressure=np.linspace(1.,610.,n) if geometry=='uniform' else np.geomspace(.01,610.,n)
                boundary=pressure[-3]/pressure[-1]
                ratios=[(.92,.991),(.7,.99),(1.,1.01),
                    (np.nextafter(boundary,0.),np.nextafter(boundary,np.inf))]
                for latitude in ([0.,.5],[np.pi/4.],[-np.pi/4.],
                    [np.nextafter(np.pi/4.,np.inf)],[-np.nextafter(np.pi/4.,np.inf)],[0.,1.]):
                    for qmass in (False,True):
                        for polar,tropical in ratios:
                            expected=ctypes.c_int(-1)
                            native.exchange_depth(n,len(latitude),int(qmass),polar,tropical,
                                f(pressure),f(np.asarray(latitude)[:,None]),ctypes.byref(expected))
                            actual=surface_exchange_start(pressure,latitude,do_qmass=qmass,
                                polar_ratio=polar,tropical_ratio=tropical)
                            np.testing.assert_equal(actual,max(0,expected.value))
                            depth_cases.append(dict(levels=n,geometry=geometry,latitudes_rad=list(latitude),
                                do_qmass=qmass,polar_ratio=polar,tropical_ratio=tropical,
                                start_layer=actual,coarse_grid_bounded=expected.value<0))
        for levels in (4,40,56):
            half=jnp.linspace(0.,610.,levels+1);full=.5*(half[1:]+half[:-1])
            co2=jnp.linspace(.9,.7,levels)
            kwargs=dict(temperature=jnp.linspace(115.,150.,levels),u=jnp.linspace(30.,5.,levels),
                v=jnp.linspace(-5.,-1.,levels),gas_tracers=jnp.stack([co2,1.-co2],axis=-1),
                soil=jnp.full(16,148.),pressure_half=half,pressure_full=full,co2_ice_kg_m2=100.,
                radiative_heating_k_s=jnp.full(levels,1e-5),downward_surface_flux_w_m2=0.,
                dt_seconds=5.,body=BODY,soil_thickness=soil_layer_thickness(16),
                soil_diffusivity=(250./(1500.*627.9))**2,soil_volumetric_heat_capacity=1500.*627.9,
                emissivity=.81,convection_scheme='none')
            for qmass in (False,True):
                for start in (0,levels-2):
                    options=dict(surface_exchange_start=start,co2_gas_index=0 if qmass else None)
                    complete=column_physics(**kwargs,**options)
                    no_phase=column_physics(**kwargs,**options,do_co2_condensation=False)
                    surface=complete.diagnostics['surface']['co2_phase_change_kg_m2']
                    weight=jnp.where(jnp.arange(levels)>=start,jnp.diff(half)*(co2 if qmass else 1.),0.)
                    mass=complete.gas_mass_loss_kg_m2-surface*weight/weight.sum()
                    heat=(complete.temperature-no_phase.temperature)/5.
                    precip=complete.diagnostics['atmospheric_co2_precipitation_kg_m2']
                    for condensation in (False,True):
                        for loss in (False,True):
                            for feedback in (False,True):
                                flags=np.array([condensation,loss,feedback,qmass],dtype=np.int32)
                                expected_t=np.empty((1,1,levels),order='F');expected_mass=np.empty_like(expected_t,order='F');snow=f(0.)
                                native.controls(levels,flags,start,f(610.),layout(no_phase.temperature),layout(heat),
                                    layout(mass),f(precip),f(surface),layout(co2),expected_t,expected_mass,snow)
                                actual=column_physics(**kwargs,**options,do_co2_condensation=condensation,
                                    do_atmos_co2_loss=loss,do_co2_condensation_cycle=feedback)
                                np.testing.assert_allclose(actual.temperature,expected_t[0,0],rtol=2e-14,atol=1e-13)
                                np.testing.assert_allclose(actual.gas_mass_loss_kg_m2,expected_mass[0,0],rtol=2e-12,atol=1e-13)
                                np.testing.assert_allclose(actual.co2_ice_kg_m2-100.,snow,rtol=2e-10,atol=1e-13)
                                cases.append(dict(levels=levels,start_layer=start,flags=flags.tolist()))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_co2_controls.py','package/src/framework/physics/column_physics.py',
        'package/src/framework/physics/dinosaur_physics.py','package/src/framework/physics/co2_condensation.py']
    return dict(status='pass',reference=provenance,cases=cases,depth_cases=depth_cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Seeded immediate-condensation control block, all three flags and do_qmass surface weights; unchanged initialization depth loops and strict local-domain latitude selection. Dinosaur uses one global domain and bounds the two extra exchange layers at its first layer on coarse grids. Phase kernels, suspended clouds, global projection and derivatives require separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']),depth_cases=len(report['depth_cases']))))
