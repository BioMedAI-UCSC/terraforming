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
    'atmos_param_mars/mars_surface.F90':'aca56974025103ab408ecf866cc78b25779fc469',
    'atmos_param_mars/mars_physics.F90':'ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc',
    'atmos_param_mars/null_physics_mod.F90':'4ca7a58d46e31dd8fb281469bbbcd8d50071bba2',
    'build_run/compile.bash':'3a43a2fdb4e26dc0980e3e5bfaa5efbe0861a597',
    'build_run/fms_mars_clouds_v3.2':'83b611e6527cf26bef3ec0064c05aebe0bb0d080',
    'atmos_param_mars/sedim_mod.F90':'df1424b6258a68d7dfb6c7b5ebfb7180224070ad',
    'atmos_param_mars/BLcode/pblmod_mgcm.F90':'38b3f41e4d07f9a2e3c9d7a689e50577cac7858c',
    'atmos_param_mars/micromod_mgcm.F90':'4897f72860b59d7effeb6650dbf3d4484b3aca71',
    'atmos_param_mars/aerosol.F90':'8737535bd91a1cc3f1ab92366405007dcb186bae',
    'atmos_param_mars/aerosol_util.F90':'353441748fa247b8d938e4993f06d313268580f7',
    'atmos_param_mars/dust_update.F90':'1e470cc1efb44c6627cbe3168c69c2d1f7640570',
    'atmos_param_mars/Rad/radiation_driver.F90':'91059b1b696ef35c05db016f1b890b9c7591f482',
    'atmos_param_mars/Rad/rtmod_mgcm.F90':'40d8879c3735bbec6cabc63817cad5220f281532',
    'atmos_param_mars/coagulation.F90':'e0c1818849461fa252ff1980dd264a93c57d79ab',
    'atmos_param_mars/blkh2omod_mgcm.F90':'92667ec8575160bc8e6a68e88cd0a93397d3d450',
    'atmos_param_mars/co2micromod_mgcm.F90':'aa566e12d6e01afa346162385055f41d8d2bc18f',
    'atmos_param_mars/palmer_topo_drag.F90':'e25163faee55b4c4d31d90f2d09db13148230123',
    'atmos_param_mars/Rad/simple_rad_driver.F90':'3e17830d742bcec90d00165db8c8037876d65f5a',
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


RADIATION_MODULE_CONTROLS=[
    ('tmom do_fv3_dust_opt fv3_gfac fv3_sscat do_dust_ir_scale fv3_ir_scale','inactive',
     'declaration and namelist only'),
    ('radactive_water','implemented','radiative_transfer water_vapor_tracer'),
    ('radactive_cloud radactive_cloud_bulk radactive_co2cloud bulk_ccn bulk_co2_ccn','implemented',
     'particle_optics.make_particle_provider selects explicit species and nuclei counts'),
    ('do_nlte_simple','implemented','radiative_transfer nonlte_tracers and nonlte_cooling'),
    ('rtdata_path','data adaptation','staged pinned table assets and recorded source hashes'),
    ('ames_15band use_extended_cor_ks','implemented','radiation_mode selects 12band, 15band or extended'),
    ('do_cia cia_co2 cia_h2','implemented','radiative_transfer cia_molar_fractions'),
    ('radactive_dust_TOG','implemented supported modes','explicit empty, moment or prescribed particle providers; mode 2 unsupported by release'),
    ('use_boxinterp12','implemented','radiative_transfer linear_k_interpolation'),
    ('radactive_cloud_bin scale_cloud_bin simple_cloud_radius','excluded bin-cloud mode',
     'optional fixed-radius bin-cloud optics set aside; public bin-cloud microphysics is a fatal placeholder'),
    ('do_irflux_scale','implemented','radiative_transfer scale_surface_flux'),
    ('scale_by_scon','implemented','reference_radiative_forcing solar_constant_w_m2'),
    ('user_fixed_dust_opts qxv_read qsv_read gv_read qxi_read qsi_read gi_read','implemented',
     'prescribed_dust fixed_optics and make_fixed_dust_provider optical_properties'),
]

SIMPLE_RADIATION_CONTROLS=[
    ('day_zero tdamp','inactive','declaration and namelist only'),
    ('kwave radscale rad_amp rad_vscale modulate_heating','excluded','numerical references require SPEC_CORE, absent from release flags'),
    ('use_forget_swheat','implemented','simple_radiation forget_shortwave_heating'),
    ('rad_calc_intv do_diurnal_avg_rad fixed_calender_date','implemented',
     'simple_radiation radiation_interval_seconds, daily_mean and fixed_calendar_date'),
    ('use_newton_damping','implemented','simple_radiation newton_damping retains source infrared suppression'),
    ('optical_depth_inpt sscat_cnst gfac_cnst conrath alfir tauir do_lat_vary_dust no_atmos_dn_ir_flx reference_press',
     'implemented','simple_radiation column_fluxes optical_depth, single_scattering_albedo, asymmetry, conrath, pressure_exponent, tau_infrared, latitude_variation, no_downward_infrared and reference_pressure'),
]


SURFACE_CONTROLS=[
    ('drag_cnst','inactive','declaration and namelist only'),
    ('zoland','diagnostic only','stored roughness map and unused local zo; main-driver numerical uses are excluded by release preprocessing; PBL has its own roughness equations'),
    ('soil_ti soil_alb','explicit input adaptation','reference loader requires supplied maps; uniform arrays can be supplied directly to the existing forcing and soil kernels'),
    ('soil_temp use_equilibrated_ts','implemented','runner reference_surface_temperature_k and uniform-soil CLI/restart identity'),
    ('do_co2_condensation alpha','implemented','column and Dinosaur surface_condensation and soil_alpha; singular native frost/off state rejected'),
    ('do_subsfc_ice subsfc_ice_case use_legacy_soil nlayers d2is gk1 gk2','implemented',
     'subsurface_properties subsurface_ice, ice_case, legacy, soil_layers, southern_minimum_layers, north_ice_inertia and south_ice_inertia; explicit properties supplied to Dinosaur'),
    ('albedo_ice_np albedo_ice_sp emiss_ice_np emiss_ice_sp','implemented',
     'RadiativeForcing co2_frost_albedo_north/south and co2_frost_emissivity_north/south'),
    ('np_cap_ti np_cap_lat np_cap_ti_max rescale_sh_ti','implemented',
     'surface_thermal_inertia north_inertia, north_latitude_deg, north_maximum and regional_scale; runner CLI/restart identity'),
    ('frost_threshold do_waterice_albedo albedo_h2o emiss_h2o albedo_h2o_liquid do_liquidwater_albedo','implemented',
     'RadiativeForcing water_frost_threshold_m, water_frost_radiative_properties, water_ice_albedo, water_emissivity, water_liquid_albedo and liquid_water_albedo'),
    ('edit_subsfc_temp','source extra-node defect','initial-conditions report: writes initialized entry 17 outside 16-entry progts; caller extra tendency unwritten'),
    ('nlay_rst','restart adaptation','NPZ persists complete state and requires the same soil shape; native restart layer-count conversion is not copied'),
    ('init_sfc_frost','implemented','boundary_inputs.polar_water_on_grid initial_frost_kg_m2'),
]


AEROSOL_CONTROLS=[
    ('fixed_dust special_fixed_dust_case do_assimilated_dust','inactive','declaration and namelist only'),
    ('optical_depth_inpt conrath conrath_type do_lat_vary_dust dust_cycle_scheme optical_depth_pulse pulse_width pulse_center optical_depth_pulse_var',
     'implemented','prescribed_dust.layer_opacities profile, absolute/fractional pulse and cycle options; composed native dispatch checked'),
    ('do_inpt_dust_cycle','implemented','make_fixed_dust_provider scenario and seasonal_fields'),
    ('inpt_dust_cycle_dust_index do_inpt_dust_cycle_ktop','implemented','explicit aerosol_tracer name and layer_opacities aerosol_ratio'),
    ('do_interactive_dust_rad','excluded bin-dust radiation','only changes dustref_bin; radiation mode 2 unsupported by the released configuration'),
    ('add_storm_opacity storm_sigx storm_sigy storm_lonc storm_latc add_storm_amp storm_areo',
     'implemented with explicit source-undefined inputs','storm_opacity and provider storm_options; source slope and latitude coefficient uninitialized'),
]

AEROSOL_OPTICS_CONTROLS=[
    ('dust_rad_size do_spectrum_ir','inactive','declaration and namelist only'),
    ('do_cloud_rad','inactive exported','public declaration and namelist only; no numerical consumer'),
    ('scale_inpt_dust_column','implemented','prescribed_dust.layer_opacities column_scale before the absolute elevated pulse'),
    ('qext_cnst sscat_cnst gfac_cnst irad_tracer nrad_tracers reduce_low_level_dust',
     'excluded bin-dust radiation','affect dustref_bin or discarded legacy spectral arrays; supported Ames dispatch uses native particle coefficients and dustref_fix'),
    ('sscat_sw_ice gfac_sw_ice scale_rad_ice scale_rad_ice_np scale_rad_ice_sp rad_ice_scheme',
     'excluded bin-cloud optics','cld_distrib_fix outputs used by the set-aside fixed-radius bin-cloud path; moment/bulk clouds use their own tracer optics'),
]

AEROSOL_UTIL_CONTROLS=[
    ('do_moment_dust do_moment_water do_bulk_water','implemented','Dinosaur moment_dust/moment_water/bulk_water and native mode-restriction guards'),
    ('dust_map_scale','implemented','load_scenario_on_grid map_scale and native visible/infrared conversion'),
    ('dust_map_scale_bin','excluded bin-dust radiation','binned mode 2 unsupported by the released configuration'),
    ('Reff_backgd Reff_stress Reff_dd','implemented','background effective_radius_m and interactive stress_radius_m/devil_radius_m'),
    ('do_moment_sedim','implemented','Dinosaur/column moment_sedimentation'),
    ('do_15band','implemented','explicit dust_optical_bands and matching radiation_mode'),
    ('Reff_fixed','implemented','prescribed_dust effective_radius_m'),
]

DUST_UPDATE_CONTROLS=[
    ('interact Background','implemented','dust_lifting_options and dust_background_options; native combined-mode restriction'),
    ('ws_co2_ice_thresh_N ws_co2_ice_thresh_S ws_h2o_ice_thresh optd_thresh stress_scheme alfa',
     'implemented','stress_options north_snow_threshold/south_snow_threshold/frost_threshold/opacity_threshold/scheme/efficiency'),
    ('dd_co2_ice_thresh_N dd_co2_ice_thresh_S dd_h2o_ice_thresh conv_delt_threshold alpha_dth alpha_dda devils_fv3 DDA DTH source_ddfv3 Dlift reff_ddfv3 thres_shflx',
     'implemented','devil_options thresholds, temperature_threshold, dth_efficiency, dda_efficiency, fv3, dda, dth, fv3_source, grain_diameter_m, fv3_radius_m and heat_flux_threshold'),
    ('threshold_stress stress_lift convective_lift inject_pbl_dd inject_pbl_stress dust_surf_ini limited_reservoir limres_xfac limres_dec delta_thres dual_mode dual_scale',
     'implemented','interactive_step threshold_stress, stress_lift, convective_lift, inject_pbl_devils, inject_pbl_stress, initial_surface_dust_kg_m2, limited_reservoir, reservoir_factor, threshold_decrease, devil_threshold_increase, dual_mode and dual_scale'),
    ('assim_t_thresh no_assim_over_caps tauscale inject_pbl_bd injscale sinkscale sink_bd dtaulim ltscale bottom_sink injec_type',
     'implemented','background_step minimum_temperature_k, exclude_caps, opacity_scale, inject_through_pbl, injection_scale, sink_scale, remove_excess, minimum_scenario_change, local_time_scale, bottom_sink and injection_type; documented conservative repairs for undefined sink cases'),
    ('segment','implemented','dust_control.seasonal_sample segment and Dinosaur background scenario selection'),
    ('kfix','implemented','interactive/background fixed_layers bottom offsets'),
    ('dgdm_type','implemented','dust_opacity_channel 1/2 and input scenario conversion'),
    ('lifting_nosed','implemented','lifting_suppresses_settling and native positive bottom-source masks'),
    ('opac_from_aerosol','implemented','dust_opacity_from_aerosol and prescribed reference-opacity cache'),
]


def namelist_controls(text,name):
    match=re.search(r'namelist\s*/'+re.escape(name)+r'/',text,re.I)
    assert match is not None,name
    parts=[]
    for line in text[match.end():].splitlines():
        code=line.split('!',1)[0].strip()
        if not code: continue
        parts.append(code.replace('&',''))
        if not code.endswith('&'): break
    return re.findall(r'[A-Za-z][A-Za-z0-9_]*',' '.join(parts))


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
            'mteta':'moment_cloud_options.contact'}),
        ('atmos_param_mars/coagulation.F90','coagulation_nml',{
            'coal_kg':'coagulation_step.coalescence','kernel_b':'coagulation_step.brownian',
            'kernel_g':'coagulation_step.gravitational','kernel_de':'coagulation_step.diffusion_enhancement',
            'kernel_ti':'inactive released table path; FULLCOAG turbulent inertia uses uninitialized eps',
            'effb':'coagulation_step.efficiency','coal_fac':'coagulation_step.factor'}),
        ('atmos_param_mars/blkh2omod_mgcm.F90','blkh2oclouds_nml',{
            'ccn_blkh2o':'water_cloud_options.nuclei_per_kg','prec_threshold':'water_cloud_options.precipitation_threshold'}),
        ('atmos_param_mars/co2micromod_mgcm.F90','co2microphys_nml',{'ccn_co2':'cloud_nuclei_per_kg'}),
        ('atmos_param_mars/palmer_topo_drag.F90','palmer_drag_nml',{
            'KAP':'terrain_drag_options.coefficient','do_IR_damping':'terrain_drag_options.ir_damping',
            'grid_res':'boundary_inputs.terrain_standard_deviation_on_grid grid_resolution'})):
        actual=namelist_controls((source/filename).read_text(),namelist)
        assert len(actual)==len(set(actual)) and set(actual)==set(mapping),(namelist,actual)
        module_controls.extend(dict(namelist=namelist,source_control=name,implementation=mapping[name]) for name in actual)
    coagulation=subprocess.check_output([compiler,'-E','-cpp','-P',*definitions,
        str(source/'atmos_param_mars/coagulation.F90')],text=True)
    stripped='\n'.join(line.split('!',1)[0] for line in coagulation.splitlines())
    assert len(re.findall(r'\bkernel_ti\b',stripped,re.I))==2
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
    driver=(source/'atmos_param_mars/Rad/radiation_driver.F90').read_text()
    driver_mapping={
        'use_forget_swheat':('inactive','declaration and namelist only'),
        'use_dust_lwheat':('inactive','declaration and namelist only'),
        'use_newton_damping':('inactive','declaration and namelist only'),
        'nchan_swx':('inactive','declaration and namelist only'),
        'rad_calc_intv':('implemented','radiation_update_seconds and radiation_interval_seconds'),
        'do_diurnal_avg_rad':('implemented','forcing.diurnal=False and daylight_geometry'),
        'rampup_rad':('implemented','shortwave_ramp_sols, atmospheric correction and surface shortwave_scale'),
        'fixed_calender_date':('implemented','radiation_clock and spectral fixed_calendar_date; optical-provider radiative clock; shared with simple radiation'),
        'use_ames_sw_rad':('source-undefined when disabled','default true; uninitialized local fluxes when false'),
        'use_ames_lw_rad':('source-undefined when disabled','default true; uninitialized local fluxes when false'),
    }
    assert set(namelist_controls(driver,'radiation_driver_nml'))==set(driver_mapping)
    stripped='\n'.join(line.split('!',1)[0] for line in driver.splitlines())
    for name,(status,_) in driver_mapping.items():
        if status=='inactive': assert len(re.findall(r'\b'+name+r'\b',stripped,re.I))==2,name
    radiation_driver_controls=[dict(source_control=name,status=status,implementation=implementation)
        for name,(status,implementation) in driver_mapping.items()]
    radiation_module_controls=[]
    for filename,namelist,mapping in (
        ('rtmod_mgcm.F90','ames_rtmod_nml',RADIATION_MODULE_CONTROLS),
        ('simple_rad_driver.F90','simple_rad_driver_nml',SIMPLE_RADIATION_CONTROLS)):
        raw=(source/'atmos_param_mars/Rad'/filename).read_text()
        actual=namelist_controls(raw,namelist)
        entries=[dict(namelist=namelist,source_control=name,status=status,implementation=implementation)
            for names,status,implementation in mapping for name in names.split()]
        names=[item['source_control'] for item in entries]
        assert len(names)==len(set(names)) and set(names)==set(actual),(namelist,actual,names)
        stripped='\n'.join(line.split('!',1)[0] for line in raw.splitlines())
        for entry in entries:
            if entry['status']=='inactive':
                assert len(re.findall(r'\b'+entry['source_control']+r'\b',stripped,re.I))==2,entry
        if filename=='simple_rad_driver.F90':
            released=subprocess.check_output([compiler,'-E','-cpp','-P',*definitions,
                str(source/'atmos_param_mars/Rad'/filename)],text=True)
            released='\n'.join(line.split('!',1)[0] for line in released.splitlines())
            for entry in entries:
                if entry['status']=='excluded':
                    assert len(re.findall(r'\b'+entry['source_control']+r'\b',released,re.I))==2,entry
        radiation_module_controls.extend(entries)
    surface=(source/'atmos_param_mars/mars_surface.F90').read_text()
    surface_controls=[dict(source_control=name,status=status,implementation=implementation)
        for names,status,implementation in SURFACE_CONTROLS for name in names.split()]
    names=[item['source_control'] for item in surface_controls]
    assert len(names)==len(set(names)) and set(names)==set(namelist_controls(surface,'surface_data_nml'))
    stripped='\n'.join(line.split('!',1)[0] for line in surface.splitlines())
    assert len(re.findall(r'\bdrag_cnst\b',stripped,re.I))==2
    assert len(re.findall(r'\bzo\b',stripped,re.I))==2
    aerosol_controls=[]
    for filename,namelist,mapping in (
        ('aerosol.F90','aerosol_nml',AEROSOL_CONTROLS),
        ('aerosol.F90','aerosol_optics_nml',AEROSOL_OPTICS_CONTROLS),
        ('aerosol_util.F90','aerosol_util_nml',AEROSOL_UTIL_CONTROLS),
        ('dust_update.F90','dust_update_nml',DUST_UPDATE_CONTROLS)):
        raw=(source/'atmos_param_mars'/filename).read_text()
        entries=[dict(namelist=namelist,source_control=name,status=status,implementation=implementation)
            for names,status,implementation in mapping for name in names.split()]
        names=[item['source_control'] for item in entries]
        actual=namelist_controls(raw,namelist)
        assert len(names)==len(set(names)) and set(names)==set(actual),(namelist,set(actual)-set(names),set(names)-set(actual))
        stripped='\n'.join(line.split('!',1)[0] for line in raw.splitlines())
        for entry in entries:
            if entry['status'].startswith('inactive'):
                count=3 if entry['status']=='inactive exported' else 2
                assert len(re.findall(r'\b'+entry['source_control']+r'\b',stripped,re.I))==count,entry
        aerosol_controls.extend(entries)
    routines,_=extract_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',
        BLOBS['atmos_param_mars/Rad/rtmod_mgcm.F90'],['ames_rt_driver'])
    spectral_driver=routines[0]
    prefix=spectral_driver[:spectral_driver.index('if (dosw) then')]
    suffix=spectral_driver[spectral_driver.index('endif !end lw check'):]
    for name in ('fluxupv','fluxdnv','fmnetv','nfluxtopv','fluxupi','fluxdni','fmneti','nfluxtopi'):
        assert not re.search(r'^\s*'+name+r'(?:\([^\n]*?\))?\s*=',prefix+suffix,re.I|re.M),name
    assert 'irtot0 = gcp*(FMNETI(1) - NFLUXTOPI)' in suffix
    assert 'swtot0 = gcp*(FMNETV(1) - NFLUXTOPV)' in suffix
    ramp_start=driver.index('    if ( rampup_rad > 0.0 ) then')
    ramp_block=driver[ramp_start:driver.index('    endif',ramp_start)+len('    endif')]
    aerosol=(source/'atmos_param_mars/aerosol_util.F90').read_text()
    dust_source=(source/'atmos_param_mars/dust_update.F90').read_text()
    start=dust_source.index('    if(lifting_nosed) then')
    lifting=dust_source[start:dust_source.index('    end if',start)+len('    end if')]
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
    adapter=adapter.replace('end module','''subroutine check_shortwave_ramp(n,sols,rampup_rad,hsw,trans) bind(C)
integer(c_int),value :: n
real(c_double),value :: sols,rampup_rad
real(c_double),intent(inout) :: hsw(n),trans(n)
real(c_double),parameter :: pi=acos(-1.d0)
real(c_double) :: ramp
'''+ramp_block+'''
end subroutine
end module''')
    mode_cases=[]
    with tempfile.TemporaryDirectory(prefix='aerosol-validity-native-') as temporary:
        code=Path(temporary)/'modes.f90';code.write_text(adapter)
        library=Path(temporary)/'modes.so'
        subprocess.run([compiler,'-O3','-ffree-line-length-none','-fPIC','-shared',str(code),'-o',str(library)],
            cwd=temporary,check=True)
        native=ctypes.CDLL(str(library))
        native.check_shortwave_ramp.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double]+[
            np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')]*2
        native.check_shortwave_ramp.restype=None
        from src.framework.gcm._dinosaur import jax
        from src.framework.physics.radiative_transfer import shortwave_ramp
        jax.config.update('jax_enable_x64',True)
        ramp_cases=[]
        for duration in (-1.,0.,1.,5.,10.):
            for sols in (0.,.25,1.,2.5,5.,7.5,10.,15.):
                heating=np.array([1.,-2.,3.]);surface=np.array([100.,50.,0.])
                native.check_shortwave_ramp(3,sols,duration,heating,surface)
                factor=float(shortwave_ramp(sols,duration))
                np.testing.assert_allclose(heating,np.array([1.,-2.,3.])*factor,rtol=2e-15,atol=2e-15)
                np.testing.assert_allclose(surface,np.array([100.,50.,0.])*factor,rtol=2e-15,atol=2e-15)
                ramp_cases.append(dict(duration_sols=duration,sols=sols,factor=factor))
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
    main=re.search(r'\bsubroutine\s+mars_physics\s*\(.*?end\s+subroutine\s+mars_physics\b',release,re.I|re.S)
    assert main is not None
    main='\n'.join(line.split('!',1)[0] for line in main[0].splitlines())
    assert len(re.findall(r'\bzo\b',main,re.I))==2
    dust_without_comments='\n'.join(line.split('!',1)[0] for line in dust_source.splitlines())
    assert len(re.findall(r'\bsfc_roughness\b',dust_without_comments,re.I))==1
    excluded=sorted(calls(alternate)-calls(release))
    assert {'surface_flux_2d','vert_turb_driver'}<=set(excluded)
    assert not re.search(r'^use\s+vert_(turb|diff)_driver_mod',release,re.M)
    assert not re.search(r'^\s*call\s+tracer_source_sink\b',release,re.I|re.M)
    assert not re.search(r'^\s*(?:if\s*\(.*?\)\s*)?alpha\s*=\s*dt\s*/\s*tau_diffusion',release,re.I|re.M)
    released_dust=subprocess.check_output([compiler,*flags,str(source/'atmos_param_mars/dust_update.F90')],text=True)
    released_dust='\n'.join(line.split('!',1)[0] for line in released_dust.splitlines())
    assert not re.search(r'call\s+tagging_main\b',released_dust,re.I)
    assert len(re.findall(r'\bsources_scenario\b',released_dust,re.I))==2
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
        shortwave_ramp_block_sha256=hashlib.sha256(ramp_block.encode()).hexdigest(),
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]),
        cases=cases,driver_controls=inventory,module_controls=module_controls,aerosol_mode_cases=mode_cases,
        radiation_driver_controls=radiation_driver_controls,radiation_module_controls=radiation_module_controls,
        surface_controls=surface_controls,aerosol_controls=aerosol_controls,
        dust_optional_inputs={
            'radius_scenario.nc':'implemented: seasonal nodal injection shares by physical mode; static removal fractions and particle radii unchanged',
            'localtime_frac.nc':'implemented: native-hour field labels, seasonal/nodal fractions and persisted local-time history',
            'sources.nc':'tagging-only numerical consumers excluded by RELEASE; seasonal source variable has declaration and assignment only',
            'inidust.nc':'tagging initial-distribution field; tagging calls excluded by RELEASE'},
        shortwave_ramp_cases=ramp_cases,
        unsupported_dust_radiation=dict(mode=2,source='build_run/fms_mars_clouds_v3.2',
            evidence=unsupported[0].strip(),status='not supported by the released configuration'),
        dust_lifting_guard_cases=lifting_cases,
        excluded_build_interfaces=dict(required_definition='fv3_turb',calls=excluded,
            alternate_preprocessed_sha256=hashlib.sha256(alternate.encode()).hexdigest()),
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_release_interfaces.py','scripts/native_reference.py',
             'package/src/framework/physics/radiative_transfer.py','package/src/framework/physics/dinosaur_physics.py',
             'package/src/framework/physics/gcm.py','package/src/framework/physics/prescribed_dust.py',
             'package/src/framework/physics/simple_radiation.py','package/src/framework/physics/soil_frost.py',
             'package/src/celestials/planets/mars/surface.py','package/src/celestials/planets/mars/dust_scenario.py',
             'package/src/framework/physics/dust_lifting.py','package/src/framework/physics/dust_control.py')},
        limits='The inventory covers the main driver plus sedimentation, PBL, moment/bulk/CO2-cloud, coagulation and terrain-drag controls. The moment-cloud guard leaves tendency outputs zero and cloud history inactive when cloud formation is disabled. Numerical behavior uses the separate native reports and integration checks. '
            'The ten radiation-driver controls and the spectral/simple module controls are inventoried separately. Fixed-date forcing shares the source-checked radiation clock with simple radiation; disabling either native radiation channel leaves local fluxes undefined. Forty unchanged startup-ramp block cases check heating and applied surface flux. '
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
