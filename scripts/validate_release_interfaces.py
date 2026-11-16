"""Audit real and unavailable physics interfaces in the pinned public release."""
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
from native_reference import extract_routines

BLOBS={
    'atmos_param_mars/mars_physics.F90':'ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc',
    'atmos_param_mars/null_physics_mod.F90':'4ca7a58d46e31dd8fb281469bbbcd8d50071bba2',
    'build_run/compile.bash':'3a43a2fdb4e26dc0980e3e5bfaa5efbe0861a597',
    'build_run/fms_mars_clouds_v3.2':'83b611e6527cf26bef3ec0064c05aebe0bb0d080',
    'atmos_param_mars/sedim_mod.F90':'df1424b6258a68d7dfb6c7b5ebfb7180224070ad',
    'atmos_param_mars/BLcode/pblmod_mgcm.F90':'38b3f41e4d07f9a2e3c9d7a689e50577cac7858c',
    'atmos_param_mars/micromod_mgcm.F90':'4897f72860b59d7effeb6650dbf3d4484b3aca71',
    'atmos_param_mars/aerosol_util.F90':'353441748fa247b8d938e4993f06d313268580f7',
    'atmos_param_mars/dust_update.F90':'1e470cc1efb44c6627cbe3168c69c2d1f7640570',
}

# Each released driver control needs an explicit implementation or exclusion.
DRIVER_CONTROLS=[
    ('no_forcing freeze_tracer_fields gust_fact','implemented',
     'dinosaur_physics.build_step and column_physics; surface-controls report'),
    ('t_zero t_strat delh delv eps sigma_b ka ks do_teq_z do_mars_radiation do_simple_radiation','implemented',
     'thermal_relaxation, radiative_transfer and simple_radiation; analytic kernel is not called by native dispatch'),
    ('kf do_conserve_energy rayleighModelTop_flag rayleighModelTop_pres_cutoff rayleighModelTop_tau_days rayleighModelTop_pres_inflex rayleighTropical_flag rayleighTropical_pres_cutoff rayleighTropical_tau_days rayleighTropical_pres_inflex rayleighTropical_lat_width','implemented',
     'rayleigh_drag.rayleigh_tendencies and column_physics dissipative heating'),
    ('do_mars_surface do_vert_diff do_ames_pbl','implemented',
     'column_physics surface/PBL controls; release diffuse fallback checked natively'),
    ('do_convective_adjust do_fv3_convect do_moist_convective_adjust','implemented',
     'column_physics convection_scheme and moist_convection_options'),
    ('do_co2_condensation_cycle do_co2_condensation form_co2_clouds do_atmos_co2_loss do_qmass klevs_ratio_pole klevs_ratio_trop','implemented',
     'column_physics, co2_microphysics and co2_condensation.surface_exchange_start; co2-controls report'),
    ('do_coagul_dst inj_vap_pbl facsubl liquid_vap_pres blk_latent_heat form_bulk_clouds','implemented',
     'dust_coagulation, water_surface options and column_physics form_bulk_clouds'),
    ('GW_drag_TOG topo_drag_fac','implemented',
     'terrain_drag_options and nonorographic_drag_options; terrain factor scales final tendencies'),
    ('do_dust_source_sink do_bin_water_cycle do_pchem','unavailable',
     'released fatal placeholder interfaces checked below'),
    ('trflux trsink','inactive',
     'tracer_source_sink has no caller in released mars_physics'),
    ('tau_diffusion diffusion_smooth','excluded',
     'only used in fv3_turb branches excluded by release flags'),
    ('checkcons','diagnostics adaptation',
     'kernel/global material ledgers and validation checks; native FMS debug abort behavior is not copied'),
]


def namelist_controls(text,name):
    match=re.search(r'namelist\s*/'+re.escape(name)+r'/(.*?)(?=\n\n)',text,re.I|re.S)
    assert match is not None,name
    return re.findall(r'[A-Za-z][A-Za-z0-9_]*',match[1].replace('&',''))


def validate(source,compiler):
    for name,expected in BLOBS.items():
        raw=(source/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected,name
    build=(source/'build_run/compile.bash').read_text()
    definitions=re.search(r'^cppDefs="([^"]+)"',build,re.M)
    assert definitions is not None
    definitions=definitions[1].split()
    assert definitions==['-DMARS_GCM','-DMARS_SURFACE','-DRELEASE']
    profile=(source/'build_run/fms_mars_clouds_v3.2').read_text()
    unsupported=re.search(r'^\s*!\s*=\s*2\s+not supported\s*$',profile,re.M|re.I)
    assert unsupported is not None
    path=source/'atmos_param_mars/mars_physics.F90'
    text=path.read_text()
    controls=namelist_controls(text,'mars_physics_nml')
    inventory=[dict(source_control=name,status=status,implementation=implementation)
        for names,status,implementation in DRIVER_CONTROLS for name in names.split()]
    covered=[item['source_control'] for item in inventory]
    assert len(covered)==len(set(covered))
    assert set(controls)==set(covered),(set(controls)-set(covered),set(covered)-set(controls))
    module_controls=[]
    for filename,namelist,mapping in (
        ('atmos_param_mars/sedim_mod.F90','sedim_nml',{'scaveng':'particle_scavenging'}),
        ('atmos_param_mars/BLcode/pblmod_mgcm.F90','ames_pblmod_nml',{'htflux_recalc':'recalculate_heat_flux'}),
        ('atmos_param_mars/micromod_mgcm.F90','microphys_nml',{
            'microtimestep':'moment_cloud_options.microtimestep','latent_heat':'moment_cloud_options.latent_heat',
            'makeclouds':'form_moment_clouds','subeffmode':'moment_cloud_options.effective_input_mode',
            'facsubl':'inactive module-local variable','mteta_case':'moment_cloud_options.contact_case',
            'mteta':'moment_cloud_options.contact'})):
        actual=namelist_controls((source/filename).read_text(),namelist)
        assert len(actual)==len(set(actual)) and set(actual)==set(mapping),(namelist,actual)
        module_controls.extend(dict(namelist=namelist,source_control=name,implementation=mapping[name]) for name in actual)
    micro,_=extract_routines(source/'atmos_param_mars/micromod_mgcm.F90',
        BLOBS['atmos_param_mars/micromod_mgcm.F90'],['micro_driver'])
    micro=micro[0];start=micro.index('if (makeclouds) then');end=micro.index('endif  !! makeclouds',start)
    guard=micro[start:end]
    assert re.search(r'call\s+nucleacond\b',guard,re.I)
    for match in re.finditer(r'^\s*(tdt_micro|rdt_micro)\([^\n]*?\)\s*=([^\n]*)',micro,re.M):
        if not start<=match.start()<end: assert match[2].strip()=='0.0',match[0]
    for name in ('tlprev','qpiprev','firstcall_micro'):
        for match in re.finditer(r'^\s*'+name+r'(?:\([^\n]*?\))?\s*=',micro,re.M):
            assert start<=match.start()<end,(name,match[0])
    raw_micro=(source/'atmos_param_mars/micromod_mgcm.F90').read_text()
    without_comments='\n'.join(line.split('!',1)[0] for line in raw_micro.splitlines())
    assert len(re.findall(r'\bfacsubl\b',without_comments,re.I))==2
    aerosol=(source/'atmos_param_mars/aerosol_util.F90').read_text()
    dust=(source/'atmos_param_mars/dust_update.F90').read_text()
    start=dust.index('    if(lifting_nosed) then')
    lifting=dust[start:dust.index('    end if',start)+len('    end if')]
    guards='\n'.join(line for line in aerosol.splitlines() if line.startswith('if (do_moment_'))
    assert len(guards.splitlines())==3
    adapter='''module validity_mod
use iso_c_binding
implicit none
integer,parameter :: FATAL=1
logical :: do_moment_dust,do_moment_water,do_bulk_water,invalid
contains
subroutine error_mesg(name,message,level)
character(*),intent(in) :: name,message
integer,intent(in) :: level
invalid=.true.
end subroutine
subroutine check_modes(dust,water,bulk,allowed) bind(C)
integer(c_int),value :: dust,water,bulk
integer(c_int),intent(out) :: allowed
do_moment_dust=dust/=0;do_moment_water=water/=0;do_bulk_water=bulk/=0;invalid=.false.
'''+guards+'''
allowed=merge(0,1,invalid)
end subroutine
subroutine check_lifting(n,enabled,tendency,mask) bind(C)
integer(c_int),value :: n,enabled
real(c_double),intent(in) :: tendency(n,1,1,2)
integer(c_int),intent(out) :: mask(n,2)
real(c_double) :: rdt_dst(n,1,1,2)
logical :: lifting_nosed,lifting_dust(n,1,2)
integer,parameter :: kd=1,ndx=1
rdt_dst=tendency;lifting_nosed=enabled/=0;lifting_dust=.false.
'''+lifting+'''
mask=merge(1,0,lifting_dust(:,1,:))
end subroutine
end module
'''
    mode_cases=[]
    with tempfile.TemporaryDirectory(prefix='aerosol-validity-native-') as temporary:
        code=Path(temporary)/'modes.f90';code.write_text(adapter)
        library=Path(temporary)/'modes.so'
        subprocess.run([compiler,'-O3','-ffree-line-length-none','-fPIC','-shared',str(code),'-o',str(library)],
            cwd=temporary,check=True)
        native=ctypes.CDLL(str(library))
        native.check_modes.argtypes=[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_int)]
        native.check_modes.restype=None
        native.check_lifting.argtypes=[ctypes.c_int]*2+[
            np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')]
        native.check_lifting.restype=None
        tendency=np.zeros((5,1,1,2),order='F')
        tendency[:,0,0,0]=[-1.,0.,1e-30,1.,-1e-30]
        tendency[:,0,0,1]=[1.,1.,-1.,0.,1.]
        lifting_cases=[]
        for enabled in (False,True):
            mask=np.empty((5,2),dtype=np.int32,order='F')
            native.check_lifting(5,int(enabled),tendency,mask)
            expected=np.broadcast_to((enabled&(tendency[:,0,0,0]>0.))[:,None],mask.shape)
            np.testing.assert_array_equal(mask,expected)
            lifting_cases.append(dict(enabled=enabled,mass_tendency=tendency[:,0,0,0].tolist(),mask=mask.tolist()))
        for dust in (False,True):
            for water in (False,True):
                for bulk in (False,True):
                    allowed=ctypes.c_int(-1)
                    native.check_modes(int(dust),int(water),int(bulk),ctypes.byref(allowed))
                    assert allowed.value==int((not water or dust) and not (bulk and (dust or water)))
                    mode_cases.append(dict(moment_dust=dust,moment_water=water,bulk_water=bulk,allowed=bool(allowed.value)))
    flags=['-cpp','-E','-P',*definitions]
    release=subprocess.check_output([compiler,*flags,str(path)],text=True)
    assert 'use null_physics_mod, only: cloud_physics' in release
    assert 'photochem_driver' in release
    assert not re.search(r'^use\s+(cloud_physics_mod|photochem_mod|dust_source_mod|tagging_method_mod)',release,re.M)
    alternate=subprocess.check_output([compiler,*flags,'-Dfv3_turb',str(path)],text=True)
    calls=lambda text:set(re.findall(r'^\s*call\s+(\w+)',text,re.I|re.M))
    excluded=sorted(calls(alternate)-calls(release))
    assert {'surface_flux_2d','vert_turb_driver'}<=set(excluded)
    assert not re.search(r'^use\s+vert_(turb|diff)_driver_mod',release,re.M)
    assert not re.search(r'^\s*call\s+tracer_source_sink\b',release,re.I|re.M)
    assert not re.search(r'^\s*(?:if\s*\(.*?\)\s*)?alpha\s*=\s*dt\s*/\s*tau_diffusion',release,re.I|re.M)
    names=['cloud_physics','cloud_physics_init','photochem_driver','dust_source_sink','dust_source_init']
    routines,_=extract_routines(source/'atmos_param_mars/null_physics_mod.F90',BLOBS['atmos_param_mars/null_physics_mod.F90'],names)
    cases=[]
    for name,body in zip(names,routines,strict=True):
        assert re.search(r'call\s+error_mesg\([^\n]*FATAL\)',body,re.I),name
        calls=re.findall(r'^\s*call\s+(\w+)',body,re.I|re.M)
        assert calls==['error_mesg'],(name,calls)
        cases.append(dict(interface=name,released_implementation='fatal placeholder',
            routine_sha256=hashlib.sha256(body.encode()).hexdigest()))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=dict(commit='ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26',
        source_git_blobs=BLOBS,preprocessor_flags=flags,
        preprocessed_sha256=hashlib.sha256(release.encode()).hexdigest(),
        moment_cloud_guard_sha256=hashlib.sha256(guard.encode()).hexdigest(),
        aerosol_mode_guard_sha256=hashlib.sha256(guards.encode()).hexdigest(),
        dust_lifting_guard_sha256=hashlib.sha256(lifting.encode()).hexdigest(),
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]),
        cases=cases,driver_controls=inventory,module_controls=module_controls,aerosol_mode_cases=mode_cases,
        unsupported_dust_radiation=dict(mode=2,source='build_run/fms_mars_clouds_v3.2',
            evidence=unsupported[0].strip(),status='not supported by the released configuration'),
        dust_lifting_guard_cases=lifting_cases,
        excluded_build_interfaces=dict(required_definition='fv3_turb',calls=excluded,
            alternate_preprocessed_sha256=hashlib.sha256(alternate.encode()).hexdigest()),
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_release_interfaces.py','scripts/native_reference.py')},
        limits='The inventory covers the main driver plus sedimentation, PBL and moment-cloud controls. The moment-cloud guard leaves tendency outputs zero and cloud history inactive when cloud formation is disabled. Numerical behavior uses the separate native reports and integration checks. '
            'Bin-cloud microphysics, photochemistry and legacy binned dust-source equations are not in '
            'these released implementations. Tagging imports are excluded by RELEASE. '
            'Moment water, bulk water and moment dust have separate real modules and validations. '
            'Alternate FV3 turbulence calls require fv3_turb, which the pinned build does not define; '
            'this exclusion does not implement or validate those alternate schemes.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} unavailable release interfaces verified")
