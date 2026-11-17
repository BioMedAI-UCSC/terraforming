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
from src.framework.physics.hydrostatic_initialization import surface_pressure,air_temperature
from src.framework.physics.vertical_grid import coefficients

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
    controls=re.findall(r'^\+\s*namelist\s*/mars_iso_nml/([^\n]+)',patch.decode(),re.M|re.I)
    assert len(controls)==1 and set(controls[0].replace(' ','').split(','))=={'isotemp','do_isothermal'}
    start=patch.decode().index('+      p0 = 0.8*drym')
    lines=patch.decode()[start:patch.decode().index('+#elif defined(VENUS_GCM)',start)].splitlines()
    assert all(line.startswith('+') for line in lines)
    air_block='\n'.join(line[1:] for line in lines)
    text=surface.decode();start=text.index('    if (use_equilibrated_ts) then')
    block=text[start:text.index('    sfc_snow(:,:)= 0.0',start)]
    edit_start=text.index('if( edit_subsfc_temp  ) then')
    edit=text[edit_start:text.index('\nendif',edit_start)+len('\nendif')]
    assert 'tsoil(:,:,nlayers+1)=  150.0' in edit
    driver=(source/'atmos_param_mars/mars_physics.F90').read_bytes()
    assert blob(driver)=='ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc'
    assert 'dimension(is:ie,js:je,nlayers)      ::  tgrnd' in text
    assert 'dimension(is:ie,js:je,nlayers)     ::  tg_dt' in text
    assert 'real, dimension(size(t,1),size(t,2),size(tsoil,3))  :: tgrnd,' in driver.decode()
    assert 'tgrnd=tgrnd+tg_tnd' in driver.decode()
    code='''module reference_mod
use iso_c_binding
implicit none
contains
subroutine cold_start(n,equilibrated,edit_deep,uniform_temperature,latitude,temperature,soil) bind(C)
integer(c_int),value :: n,equilibrated,edit_deep
real(c_double),value :: uniform_temperature
real(c_double),intent(in) :: latitude(n)
real(c_double),intent(out) :: temperature(n),soil(n,17)
real :: lat(n,1),t_surf(n,1),tsoil(n,1,17)
real,parameter :: pi=3.1415926535897932384626433832795
real :: soil_temp
integer,parameter :: nlayers=16
logical :: use_equilibrated_ts,edit_subsfc_temp
integer :: k
lat(:,1)=latitude
soil_temp=uniform_temperature;use_equilibrated_ts=equilibrated/=0;edit_subsfc_temp=edit_deep/=0
'''+block+edit+'''
temperature=t_surf(:,1)
soil=tsoil(:,1,:)
end subroutine
end module
'''
    code+='''module air_reference_mod
use iso_c_binding
implicit none
real,parameter :: rdgas=189.02,grav=3.72,cp_air=735.94
contains
logical function is_master()
is_master=.false.
end function
real function g_sum(domain,field,is,ie,js,je,ng,area,one,average)
integer,intent(in) :: domain,is,ie,js,je,ng,one
logical,intent(in) :: average
real,intent(in) :: field(is:ie,js:je),area(is:ie,js:je)
g_sum=sum(field*area)/sum(area)
end function
subroutine cold_air(n,km,choice,selected_temperature,drym,phi,weights,ak,bk,out,tout,dpout) bind(C)
integer(c_int),value :: n,km,choice
real(c_double),value :: selected_temperature,drym
real(c_double),intent(in) :: phi(n),weights(n),ak(km+1),bk(km+1)
real(c_double),intent(out) :: out(n),tout(n,km),dpout(n,km)
integer :: is,ie,js,je,ng,domain,i,j,k
logical :: do_isothermal
real :: hs(n,1),area(n,1),ps(n,1),pt(n,1,km),delp(n,1,km)
real :: isotemp,p0,t0,gztop,psm,dps,localp,tsat
is=1;ie=n;js=1;je=1;ng=0;domain=0
hs(:,1)=phi;area(:,1)=weights;isotemp=selected_temperature;do_isothermal=choice/=0
'''+air_block+'''
out=ps(:,1);tout=pt(:,1,:);dpout=delp(:,1,:)
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
        native.cold_start.argtypes=[ctypes.c_int]*3+[ctypes.c_double,array,array,array];native.cold_start.restype=None
        native.cold_air.argtypes=[ctypes.c_int]*3+[ctypes.c_double]*2+[array]*7;native.cold_air.restype=None
        expected=np.empty_like(lat);soil=np.empty((len(lat),17),order='F');cases=[]
        for equilibrated in (False,True):
            for temperature in (150.,170.,230.):
                for edit_deep in (False,True):
                    native.cold_start(len(lat),int(equilibrated),int(edit_deep),temperature,lat,expected,soil)
                    actual=np.asarray(reference_surface_temperature_k(lat,equilibrated=equilibrated,
                        soil_temperature_k=temperature))
                    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=1e-13)
                    np.testing.assert_array_equal(soil[:,:16],np.broadcast_to(expected[:,None],(len(lat),16)))
                    deep=np.where((np.abs(lat)>70.*np.pi/180.)&edit_deep,150.,expected)
                    np.testing.assert_array_equal(soil[:,16],deep)
                    cases.append(dict(equilibrated=equilibrated,soil_temperature_k=temperature,
                        edit_subsfc_temp=edit_deep,max_surface_error_k=float(np.max(abs(actual-expected)))))
        air_cases=[];n=23;weights=np.asfortranarray(np.linspace(.1,2.,n)**2)
        for layers in (24,56):
            for dry in (610.,705.,50000.):
                ak,bk=(np.asfortranarray(x) for x in coefficients(layers,dry))
                for selected in (130.,170.,250.):
                    for isothermal in (False,True):
                        for terrain in (False,True):
                            phi=np.asfortranarray(3.72*(np.linspace(-8000.,18000.,n) if terrain else np.zeros(n)))
                            pressure=np.empty(n);temperature=np.empty((n,layers),order='F');dp=np.empty_like(temperature)
                            native.cold_air(n,layers,int(isothermal),selected,dry,phi,weights,ak,bk,pressure,temperature,dp)
                            actual=surface_pressure(phi,weights,dry_mass_pa=dry,temperature_k=selected,top_pressure_pa=ak[0])
                            np.testing.assert_allclose(actual,pressure,rtol=2e-12,atol=2e-12)
                            half=ak[:,None]+bk[:,None]*actual
                            actual_temperature=air_temperature(half,actual,temperature_k=selected,isothermal=isothermal)
                            np.testing.assert_allclose(actual_temperature,temperature.T,rtol=2e-12,atol=2e-12)
                            np.testing.assert_allclose(np.diff(half,axis=0),dp.T,rtol=2e-12,atol=2e-12)
                            air_cases.append(dict(layers=layers,dry_mass_parameter_pa=dry,temperature_k=selected,
                                isothermal=isothermal,terrain=terrain,max_pressure_error_pa=float(np.max(abs(actual-pressure))),
                                max_temperature_error_k=float(np.max(abs(actual_temperature-temperature.T)))))
    root=Path(__file__).parents[1]
    paths=('scripts/validate_initial_conditions.py','scripts/run_gcm3d_ablation.py',
        'package/src/celestials/planets/mars/maps.py','package/src/framework/physics/hydrostatic_initialization.py')
    return dict(status='pass',reference=dict(surface_git_blob=pinned_surface,patch_git_blob=PATCH_BLOB,
        block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),compiler_flags=flags,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]),
        initial_air_temperature_k=170.,latitude_cases=len(lat),initialization_cases=cases,
        hydrostatic_cases=air_cases,hydrostatic_block_sha256=hashlib.sha256(air_block.encode()).hexdigest(),
        atmosphere_controls=[dict(source_control='isotemp',implementation='--initial-air-temperature'),
            dict(source_control='do_isothermal',implementation='--isothermal-atmosphere/--no-isothermal-atmosphere')],
        global_reduction_adapter='single-process weighted sum replaces FMS g_sum; numerical cold-start block unchanged',
        initialized_soil_nodes=17,solver_soil_nodes=16,
        extra_node_audit=dict(driver_git_blob=blob(driver),edit_block_sha256=hashlib.sha256(edit.encode()).hexdigest(),
            status="edit_subsfc_temp changes only entry 17; progts input/output declarations cover 16 entries; caller extra tendency is unwritten"),
        max_surface_error_k=max(case['max_surface_error_k'] for case in cases),
        sha256={path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths},
        limits='Unchanged surface and atmospheric cold-start blocks, including log-weighted pressure correction and optional frost-limited temperature profile. The surface extra-node defect is audited, not copied into active physics. Dinosaur projects log pressure to its spectral space and evaluates the non-isothermal profile on sigma interfaces.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['initialization_cases'])} surface initializations across {report['latitude_cases']} latitudes and {len(report['hydrostatic_cases'])} atmospheric configurations passed")
