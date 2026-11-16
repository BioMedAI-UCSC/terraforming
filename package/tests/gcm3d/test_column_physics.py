import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp

from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY
from src.framework.physics.co2_condensation import simple_condensation
from src.framework.physics.column_physics import column_physics,BulkWaterColumnState
from src.framework.physics.soil_frost import soil_layer_thickness

jax.config.update('jax_enable_x64',True)


def test_simple_condensation_latent_heat_matches_mass_away_from_guard():
    half=jnp.array([0.,50.,150.,350.,610.])
    full=(half[1:]+half[:-1])*.5
    temperature=jnp.array([100.,120.,140.,170.])
    t,mass,precip,guard=simple_condensation(temperature,half,full,BODY.cp_j_kg_k,BODY.gravity_m_s2)
    np.testing.assert_allclose(mass*590000.,(t-temperature)*BODY.cp_j_kg_k*
        jnp.diff(half)/BODY.gravity_m_s2,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(jnp.sum(mass),precip,rtol=2e-14)
    assert not np.any(guard)
    assert np.any(np.asarray(mass)>0.)


def test_original_mass_guard_is_exposed_instead_of_hiding_energy_mismatch():
    t,mass,_,guard=simple_condensation(jnp.array([-1000.]),jnp.array([0.,610.]),
        jnp.array([305.]),BODY.cp_j_kg_k,BODY.gravity_m_s2)
    assert bool(guard[0])
    assert float(mass[0])==61.
    assert float(t[0])>0.


def inputs():
    half=jnp.array([0.,50.,150.,350.,610.])
    return dict(temperature=jnp.array([110.,130.,145.,150.]),
        u=jnp.array([30.,20.,10.,5.]),v=jnp.array([-5.,-3.,-2.,-1.]),
        gas_tracers=jnp.arange(8,dtype=jnp.float64).reshape(4,2)*1e-5,
        soil=jnp.full(16,148.),pressure_half=half,pressure_full=(half[1:]+half[:-1])*.5,
        co2_ice_kg_m2=100.,radiative_heating_k_s=jnp.full(4,1e-5),
        downward_surface_flux_w_m2=0.,dt_seconds=300.,body=BODY,
        soil_thickness=soil_layer_thickness(16),soil_diffusivity=(250./(1500.*627.9))**2,
        soil_volumetric_heat_capacity=1500.*627.9,emissivity=.81,surface_exchange_start=0)


def test_coupled_column_reports_complete_mass_source_and_deposition():
    kwargs=inputs()
    result=jax.jit(lambda:column_physics(**kwargs))()
    np.testing.assert_allclose(result.co2_ice_kg_m2-kwargs['co2_ice_kg_m2'],
        jnp.sum(result.gas_mass_loss_kg_m2),rtol=2e-12,atol=2e-12)
    assert result.diagnostics['atmospheric_co2_precipitation_kg_m2']>0.
    assert not np.any(result.diagnostics['co2_mass_guard'])
    assert np.all(np.asarray(result.gas_tracers)>=0.)


@pytest.mark.parametrize('condensation',[False,True])
@pytest.mark.parametrize('atmospheric_loss',[False,True])
@pytest.mark.parametrize('feedback',[False,True])
def test_co2_source_switches_keep_heating_precipitation_and_mass_feedback_separate(condensation,atmospheric_loss,feedback):
    kwargs={**inputs(),'convection_scheme':'none'}
    full=column_physics(**kwargs)
    result=column_physics(**kwargs,do_co2_condensation=condensation,
        do_atmos_co2_loss=atmospheric_loss,do_co2_condensation_cycle=feedback)
    precip=result.diagnostics['atmospheric_co2_precipitation_kg_m2']
    expected_precip=full.diagnostics['atmospheric_co2_precipitation_kg_m2'] if condensation and atmospheric_loss else 0.
    np.testing.assert_allclose(precip,expected_precip,rtol=2e-14)
    surface=full.diagnostics['surface']['co2_phase_change_kg_m2']
    np.testing.assert_allclose(result.co2_ice_kg_m2,kwargs['co2_ice_kg_m2']+surface+expected_precip,rtol=2e-14)
    if feedback:
        np.testing.assert_allclose(jnp.sum(result.gas_mass_loss_kg_m2),surface+expected_precip,rtol=2e-14,atol=1e-14)
    else:
        np.testing.assert_array_equal(result.gas_mass_loss_kg_m2,0.)
    if condensation:
        np.testing.assert_array_equal(result.temperature,full.temperature)
    else:
        assert np.any(np.asarray(result.temperature)<np.asarray(full.temperature))


def test_disabled_co2_cloud_phase_driver_keeps_cloud_tracer_and_water_index():
    kwargs={**inputs(),'convection_scheme':'none'}
    cloud=jnp.linspace(.001,.004,4)
    water=BulkWaterColumnState(1,jnp.full(4,.0001),100.,0.,3)
    result=column_physics(**kwargs,suspended_co2_ice_ratio=cloud,bulk_water=water,
        form_bulk_clouds=False,do_co2_condensation=False)
    np.testing.assert_array_equal(result.suspended_co2_ice_ratio,cloud)
    np.testing.assert_array_equal(result.bulk_water.ice_ratio,water.ice_ratio)
    assert 'ground_momentum_exchange_kg_m_s' in result.diagnostics['co2_cloud']


def test_disabled_mass_feedback_skips_undefined_empty_co2_layer_weights():
    kwargs={**inputs(),'gas_tracers':jnp.zeros((4,2))}
    result=column_physics(**kwargs,co2_gas_index=0,do_co2_condensation_cycle=False)
    assert np.isfinite(result.temperature).all()
    np.testing.assert_array_equal(result.gas_mass_loss_kg_m2,0.)


@pytest.mark.parametrize('control',['do_mars_surface','do_vert_diff','do_ames_pbl'])
def test_inactive_surface_or_pbl_does_not_mix_air_or_apply_surface_drag(control):
    kwargs={**inputs(),'convection_scheme':'none','rayleigh_options':dict(timescale=0.),
        'do_co2_condensation':False}
    result=column_physics(**kwargs,**{control:False})
    np.testing.assert_array_equal(result.temperature,kwargs['temperature']+
        kwargs['radiative_heating_k_s']*kwargs['dt_seconds'])
    np.testing.assert_array_equal(result.u,kwargs['u']);np.testing.assert_array_equal(result.v,kwargs['v'])
    np.testing.assert_array_equal(result.gas_tracers,kwargs['gas_tracers'])
    for key in ('heat_flux_w_m2','stress_x_pa','stress_y_pa','heat_diffusivity_m2_s','momentum_diffusivity_m2_s'):
        np.testing.assert_array_equal(result.diagnostics['pbl'][key],0.)
    if control=='do_mars_surface':
        assert result.soil[0]==170. and result.co2_ice_kg_m2==kwargs['co2_ice_kg_m2']
        np.testing.assert_array_equal(result.soil[1:],kwargs['soil'][1:])
        np.testing.assert_array_equal(result.gas_mass_loss_kg_m2,0.)
    else:
        assert not np.array_equal(result.soil,kwargs['soil'])


def test_no_surface_rejects_reservoirs_missing_from_released_initialization():
    water=BulkWaterColumnState(0,jnp.zeros(4),100.,0.,3.)
    with pytest.raises(ValueError,match='does not initialize'):
        column_physics(**inputs(),bulk_water=water,do_mars_surface=False)


@pytest.mark.parametrize('particles',['none','bulk','moment','dust'])
def test_tracer_freeze_restores_air_species_and_retains_physical_heat_and_surface_changes(particles):
    kwargs={**inputs(),'temperature':jnp.full(4,200.),'soil':jnp.full(16,240.),
        'co2_ice_kg_m2':0.,'dt_seconds':5.,'convection_scheme':'none',
        'suspended_co2_ice_ratio':jnp.full(4,1e-5)}
    if particles in ('bulk','moment'):
        kwargs['bulk_water']=BulkWaterColumnState(1,jnp.linspace(1e-6,4e-6,4),0.,1e-5,3)
    if particles in ('moment','dust'):
        kwargs.update(moment_particles=jnp.broadcast_to(jnp.array(
            [1e-6,1e8,1e6,1e-8] if particles=='moment' else [1e-6,1e8]),(4,4 if particles=='moment' else 2)),
            moment_water=particles=='moment',surface_dust_kg_m2=30.,dust_lifting_opacity=.2,
            dust_lifting_options=dict(threshold_stress=1e-6))
    active=column_physics(**kwargs)
    frozen=column_physics(**kwargs,freeze_tracer_fields=True)
    np.testing.assert_array_equal(frozen.gas_tracers,kwargs['gas_tracers'])
    np.testing.assert_array_equal(frozen.suspended_co2_ice_ratio,kwargs['suspended_co2_ice_ratio'])
    assert not np.array_equal(active.suspended_co2_ice_ratio,frozen.suspended_co2_ice_ratio)
    for name in ('temperature','u','v','soil','co2_ice_kg_m2','gas_mass_loss_kg_m2'):
        np.testing.assert_array_equal(getattr(frozen,name),getattr(active,name))
    assert not np.array_equal(frozen.soil,kwargs['soil'])
    if particles in ('bulk','moment'):
        np.testing.assert_array_equal(frozen.bulk_water.ice_ratio,kwargs['bulk_water'].ice_ratio)
        np.testing.assert_array_equal(frozen.bulk_water.surface_water_kg_m2,active.bulk_water.surface_water_kg_m2)
    if particles in ('moment','dust'):
        np.testing.assert_array_equal(frozen.moment_particles,kwargs['moment_particles'])
        assert not np.array_equal(active.moment_particles,frozen.moment_particles)
        np.testing.assert_array_equal(frozen.diagnostics['dust']['dust_surface_kg_m2'],active.diagnostics['dust']['dust_surface_kg_m2'])


def test_dust_gust_factor_matches_fixed_released_gust_and_has_finite_derivative():
    from src.framework.physics.dust_lifting import stress_flux
    kwargs={**inputs(),'temperature':jnp.full(4,200.),'soil':jnp.full(16,240.),
        'co2_ice_kg_m2':0.,'dt_seconds':5.,'convection_scheme':'none',
        'moment_particles':jnp.broadcast_to(jnp.array([1e-6,1e8]),(4,2)),
        'moment_water':False,'surface_dust_kg_m2':30.,'dust_lifting_opacity':.2,
        'dust_lifting_options':dict(threshold_stress=1e-6,convective_lift=False)}
    quiet=column_physics(**kwargs)
    def flux(factor):
        return column_physics(**kwargs,gust_factor=factor).diagnostics['dust']['dust_lifting']['stress_lift_kg_m2']
    pbl=quiet.diagnostics['pbl']
    drag=610./(BODY.gas_constant_j_kg_k*200.)*pbl['surface_momentum_coefficient']*pbl['wind_speed_ms']
    speed=jnp.sqrt(quiet.u[-1]**2+quiet.v[-1]**2+25.)
    expected=5.*stress_flux(drag*speed,1e-6,480./(BODY.gas_constant_j_kg_k*quiet.temperature[-1]),0.,quiet.co2_ice_kg_m2,0.,.2/480.*610.)
    np.testing.assert_allclose(flux(25.),expected,rtol=3e-14)
    assert flux(25.)>flux(0.)>0.
    assert not np.isfinite(flux(-1e6))
    automatic=jax.grad(flux)(25.);epsilon=.001
    assert np.isfinite(automatic) and automatic>0.
    np.testing.assert_allclose(automatic,(flux(25.+epsilon)-flux(25.-epsilon))/(2.*epsilon),rtol=1e-7)


@pytest.mark.parametrize('modes',[1,2])
@pytest.mark.parametrize('process',['settling','lifting','coagulation','background'])
def test_dust_without_water_keeps_each_material_budget(modes,process):
    kwargs={**inputs(),'temperature':jnp.full(4,200.),'soil':jnp.full(16,200.),
        'co2_ice_kg_m2':0.,'dt_seconds':5.,'convection_scheme':'none'}
    particles=jnp.broadcast_to(jnp.array([1e-6,1e9]*modes),(4,2*modes))
    bank=jnp.full((modes,),30.) if modes>1 else jnp.asarray(30.)
    options=dict(moment_particles=particles,moment_water=False,surface_dust_kg_m2=bank,dust_lifting_opacity=.2)
    if process=='lifting':
        kwargs['soil']=jnp.full(16,240.)
        options['dust_lifting_options']=dict(threshold_stress=1e-6)
    if process=='coagulation': options['coagulation_options']={}
    if process=='background': options['dust_background_options']=dict(scenario_opacity=.4,scenario_change=0.,extinction_efficiency=2.5)
    result=column_physics(**kwargs,**options)
    gas=jnp.diff(kwargs['pressure_half'])/BODY.gravity_m_s2
    before=jnp.sum(gas[:,None]*particles[...,::2],axis=0)+bank
    after=jnp.sum(gas[:,None]*result.moment_particles[...,::2],axis=0)+result.diagnostics['dust']['dust_surface_kg_m2']
    np.testing.assert_allclose(after,before,rtol=2e-12,atol=1e-12)
    assert result.bulk_water is None and result.diagnostics['bulk_water']=={}
    assert result.moment_particles.shape==(4,2*modes)
    if process=='lifting':
        assert np.all(np.asarray(result.diagnostics['dust']['dust_lifting']['stress_lift_kg_m2'])>0.)
    if process=='background':
        assert np.any(np.asarray(result.diagnostics['dust']['dust_surface_kg_m2'])<np.asarray(bank))


def test_surface_mass_exchange_uses_initial_co2_mass_in_each_selected_layer():
    co2=jnp.array([.9,.8,.7,.6])
    kwargs={**inputs(),'temperature':jnp.full(4,210.),'dt_seconds':5.,
        'gas_tracers':jnp.stack([co2,1.-co2],axis=-1),'surface_exchange_start':1}
    result=column_physics(**kwargs,co2_gas_index=0)
    assert float(result.diagnostics['atmospheric_co2_precipitation_kg_m2'])==0.
    phase=result.diagnostics['surface']['co2_phase_change_kg_m2']
    assert abs(float(phase))>1e-8
    weighted=jnp.diff(kwargs['pressure_half'])*co2
    weighted=weighted.at[0].set(0.)
    np.testing.assert_allclose(result.gas_mass_loss_kg_m2,phase*weighted/weighted.sum(),rtol=2e-14)


def test_coupled_column_reverse_derivative_reaches_radiative_heating():
    kwargs=inputs()
    def objective(heating):
        result=column_physics(**{**kwargs,'radiative_heating_k_s':jnp.full(4,heating)})
        return jnp.sum(result.temperature)+jnp.sum(result.soil)+result.co2_ice_kg_m2
    auto=jax.grad(objective)(1e-5)
    assert np.isfinite(auto) and abs(float(auto))>0.
    for epsilon in (1e-7,3e-8,1e-8):
        finite=(objective(1e-5+epsilon)-objective(1e-5-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,finite,rtol=2e-6,atol=1e-5)


def test_coupled_column_rejects_invalid_surface_exchange_region():
    with pytest.raises(ValueError,match='layer index'):
        column_physics(**{**inputs(),'surface_exchange_start':4})


def test_legacy_convection_switch_matches_standalone_fields_and_reports_budgets():
    from src.framework.physics.legacy_convection import adjust
    kwargs=inputs();full=kwargs['pressure_full']
    kwargs['temperature']=jnp.array([260.,280.,320.,350.])*(full/610.)**BODY.kappa
    kwargs['soil']=jnp.full(16,220.)
    plain=column_physics(**kwargs,convection_scheme='none')
    legacy=jax.jit(lambda:column_physics(**kwargs,convection_scheme='legacy'))()
    expected=adjust(plain.temperature,plain.u,plain.v,plain.gas_tracers,
        kwargs['pressure_half'],full,BODY.kappa,gravity=BODY.gravity_m_s2)
    for actual,reference in zip((legacy.temperature,legacy.u,legacy.v,legacy.gas_tracers),expected,strict=True):
        np.testing.assert_allclose(actual,reference,rtol=2e-13,atol=1e-13)
    assert legacy.diagnostics['convection']['scheme_index']==2
    np.testing.assert_allclose(legacy.diagnostics['convection']['enthalpy_change_j_m2'],0.,atol=1e-8)
    with pytest.raises(ValueError,match='none, fv3 or legacy'):
        column_physics(**kwargs,convection_scheme='unknown')


def test_moist_adjustment_follows_dry_convection_and_routes_precipitation_to_water():
    from src.framework.physics.moist_convection import adjust
    half=jnp.linspace(0.,50000.,5);full=.5*(half[1:]+half[:-1])
    kwargs={**inputs(),'temperature':jnp.array([225.3,250.7,275.1,299.9]),
        'pressure_half':half,'pressure_full':full,'gas_tracers':jnp.full((4,1),.011),
        'soil':jnp.full(16,250.),'co2_ice_kg_m2':0.,'radiative_heating_k_s':jnp.zeros(4),'dt_seconds':5.}
    water=BulkWaterColumnState(0,jnp.zeros(4),100.,0.,3)
    plain=column_physics(**kwargs,bulk_water=water,form_bulk_clouds=False)
    expected=adjust(plain.temperature,plain.gas_tracers[...,0],half,full,5.)
    result=jax.jit(lambda:column_physics(**kwargs,bulk_water=water,
        form_bulk_clouds=False,moist_convection_options={}))()
    np.testing.assert_allclose(result.temperature,plain.temperature+5.*expected.temperature_tendency,
        rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(result.gas_tracers[...,0],plain.gas_tracers[...,0]+5.*expected.vapor_tendency,
        rtol=2e-13,atol=2e-15)
    np.testing.assert_allclose(result.bulk_water.surface_water_kg_m2,
        plain.bulk_water.surface_water_kg_m2+5.*(expected.rain_kg_m2_s+expected.snow_kg_m2_s),rtol=2e-14)
    carrier=jnp.diff(half)/BODY.gravity_m_s2
    np.testing.assert_allclose(jnp.sum(result.gas_tracers[...,0]*carrier)+result.bulk_water.surface_water_kg_m2,
        jnp.sum(kwargs['gas_tracers'][...,0]*carrier)+100.,rtol=2e-13)


def test_nonorographic_drag_uses_old_heights_after_terrain_before_convection(monkeypatch):
    import importlib
    module=importlib.import_module('src.framework.physics.column_physics')
    from src.framework.physics.nonorographic_drag import initial_drag_state
    from src.framework.physics.terrain_drag import hydrostatic_heights
    kwargs=inputs();events=[];captured={}
    terrain=module.terrain_tendencies;wave=module.gravity_wave_step;convection=module.pass2
    def terrain_check(*args,**options):
        events.append('terrain')
        result=terrain(*args,**options)
        captured['wind']=args[0]+kwargs['dt_seconds']*result.u_tendency
        return result
    def wave_check(t,p,z,u,*args,**options):
        events.append('wave')
        np.testing.assert_allclose(u,captured['wind'],rtol=1e-14)
        old_z,_=hydrostatic_heights(kwargs['temperature'],kwargs['pressure_half'],kwargs['pressure_full'],
            gas_constant=BODY.gas_constant_j_kg_k,gravity=BODY.gravity_m_s2,top_pressure_pa=.03)
        np.testing.assert_array_equal(z,old_z)
        result,state=wave(t,p,z,u,*args,**options)
        captured['post_wave']=u+kwargs['dt_seconds']*result.u_tendency
        return result,state
    def convection_check(t,u,*args,**options):
        events.append('convection')
        np.testing.assert_array_equal(u,captured['post_wave'])
        return convection(t,u,*args,**options)
    monkeypatch.setattr(module,'terrain_tendencies',terrain_check)
    monkeypatch.setattr(module,'gravity_wave_step',wave_check)
    monkeypatch.setattr(module,'pass2',convection_check)
    result=column_physics(**kwargs,terrain_drag_options=dict(top_pressure_pa=.03,terrain_std_m=100.),
        nonorographic_drag_options=dict(top_pressure_pa=.03,source_index=3,source_amplitude=.08),
        nonorographic_state=initial_drag_state(kwargs['u']))
    assert events==['terrain','wave','convection']
    assert result.nonorographic_state is not None


def test_optional_drag_reheating_returns_removed_kinetic_energy():
    kwargs={**inputs(),'u':jnp.full(4,100.),'v':jnp.zeros(4),
        'rayleigh_options':dict(timescale=900.)}
    cold=column_physics(**kwargs)
    warm=column_physics(**kwargs,conserve_drag_energy=True)
    diagnostic=warm.diagnostics['rayleigh']
    dt=kwargs['dt_seconds']
    unew=kwargs['u']+diagnostic['u_tendency_m_s2']*dt
    removed=.5*(kwargs['u']**2-unew**2)
    np.testing.assert_allclose(diagnostic['dissipated_heat_k']*BODY.cp_j_kg_k,removed,
        rtol=2e-14,atol=1e-12)
    assert not cold.diagnostics['rayleigh']['reheating_enabled']
    assert warm.diagnostics['rayleigh']['reheating_enabled']
    assert float(jnp.sum(warm.temperature))>float(jnp.sum(cold.temperature))


def test_suspended_mode_closes_gas_cloud_frost_mass_without_mixing_ice_as_gas():
    kwargs=inputs();q=jnp.array([.001,.005,.002,.01])
    result=jax.jit(lambda:column_physics(**kwargs,suspended_co2_ice_ratio=q))()
    gas=jnp.diff(kwargs['pressure_half'])/BODY.gravity_m_s2
    np.testing.assert_allclose(jnp.sum(gas*(result.suspended_co2_ice_ratio-q))+
        result.co2_ice_kg_m2-kwargs['co2_ice_kg_m2'],jnp.sum(result.gas_mass_loss_kg_m2),
        rtol=3e-12,atol=3e-12)
    assert result.gas_tracers.shape==kwargs['gas_tracers'].shape
    assert np.min(result.suspended_co2_ice_ratio)>=0.
    assert np.isfinite(result.diagnostics['co2_cloud']['deposition_momentum_kg_m_s']).all()


def test_cloud_momentum_source_is_reported_without_being_given_to_ground():
    kwargs=inputs();q=jnp.array([.01,.005,.002,.01])
    result=column_physics(**kwargs,suspended_co2_ice_ratio=q)
    gas=jnp.diff(kwargs['pressure_half'])/BODY.gravity_m_s2
    old_mass=gas*(1.+q)
    new_mass=gas-result.gas_mass_loss_kg_m2+gas*result.suspended_co2_ice_ratio
    cloud=result.diagnostics['co2_cloud']
    residual=cloud['ground_momentum_exchange_kg_m_s']+jnp.stack([
        jnp.sum(new_mass*new-old_mass*old)
        for old,new in zip((kwargs['u'],kwargs['v']),(result.u,result.v),strict=True)])
    np.testing.assert_allclose(cloud['unresolved_momentum_source_kg_m_s'],residual,atol=2e-12)
    # A source approximation must remain visible, rather than becoming ground torque.
    assert np.max(np.abs(residual))>1e-5


def test_bulk_water_column_keeps_water_and_co2_reservoirs_separate():
    kwargs=inputs()
    water=BulkWaterColumnState(0,jnp.full(4,.001),100.,.001,2.)
    result=jax.jit(lambda:column_physics(**kwargs,bulk_water=water,
        suspended_co2_ice_ratio=jnp.zeros(4)))()
    gas=jnp.diff(kwargs['pressure_half'])/BODY.gravity_m_s2
    before=jnp.sum(gas*(kwargs['gas_tracers'][:,0]+water.ice_ratio))+water.surface_water_kg_m2
    after=jnp.sum(gas*(result.gas_tracers[:,0]+result.bulk_water.ice_ratio))+result.bulk_water.surface_water_kg_m2
    np.testing.assert_allclose(after,before,rtol=3e-12,atol=3e-12)
    np.testing.assert_allclose(result.co2_ice_kg_m2-kwargs['co2_ice_kg_m2']+
        jnp.sum(gas*result.suspended_co2_ice_ratio),jnp.sum(result.gas_mass_loss_kg_m2),
        rtol=3e-12,atol=3e-12)
    assert result.bulk_water.previous_exchange_coefficient>=0.
    assert 0<=result.bulk_water.previous_pbl_top_layer<4


def test_bulk_water_column_reverse_derivative_reaches_surface_exchange_history():
    kwargs={**inputs(),'temperature':jnp.array([150.,180.,210.,240.]),'soil':jnp.full(16,250.)}
    def objective(coefficient):
        water=BulkWaterColumnState(0,jnp.full(4,.001),100.,coefficient,2.)
        result=column_physics(**kwargs,bulk_water=water)
        return jnp.mean(result.temperature)+result.bulk_water.surface_water_kg_m2
    auto=jax.jit(jax.grad(objective))(.001)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for eps in (1e-6,1e-7):
        finite=(objective(.001+eps)-objective(.001-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-6,atol=1e-6)


def test_bulk_surface_cycle_can_run_with_cloud_formation_disabled():
    kwargs=inputs()
    water=BulkWaterColumnState(0,jnp.zeros(4),100.,.001,2.)
    result=column_physics(**kwargs,bulk_water=water,form_bulk_clouds=False)
    np.testing.assert_array_equal(result.bulk_water.ice_ratio,0.)
    assert result.diagnostics['bulk_water']['cloud']=={}
    gas=jnp.diff(kwargs['pressure_half'])/BODY.gravity_m_s2
    before=jnp.sum(gas*kwargs['gas_tracers'][:,0])+100.
    after=jnp.sum(gas*result.gas_tracers[:,0])+result.bulk_water.surface_water_kg_m2
    np.testing.assert_allclose(after,before,rtol=3e-12,atol=3e-12)


def test_soil_water_emissivity_uses_post_exchange_inventory_and_restores_bare_map():
    kwargs={**inputs(),'co2_ice_kg_m2':0.,'soil':jnp.full(16,250.),'emissivity':.73}
    water=BulkWaterColumnState(0,jnp.zeros(4),.0001,.01,3.)
    options=dict(frost_threshold_m=5e-8,water_emissivity=1.)
    result=column_physics(**kwargs,bulk_water=water,form_bulk_clouds=False,
        water_radiative_options=options)
    assert result.bulk_water.surface_water_kg_m2<options['frost_threshold_m']*1000.
    assert result.diagnostics['surface']['emissivity']==.73
    frozen=column_physics(**kwargs,bulk_water=water._replace(previous_exchange_coefficient=0.),
        form_bulk_clouds=False,water_radiative_options=options)
    assert frozen.diagnostics['surface']['emissivity']==1.
    assert not np.array_equal(frozen.soil,result.soil)


def test_terrain_drag_is_applied_before_convection_and_keeps_its_source_residual():
    half=jnp.geomspace(.001,610.,13);full=jnp.sqrt(half[:-1]*half[1:])
    kwargs={**inputs(),'pressure_half':half,'pressure_full':full,
        'temperature':jnp.linspace(145.,215.,12),'u':jnp.linspace(12.,57.,12),
        'v':jnp.linspace(-3.,5.,12),'gas_tracers':jnp.zeros((12,0)),
        'radiative_heating_k_s':jnp.zeros(12),'soil':jnp.full(16,215.),'co2_ice_kg_m2':0.}
    options=dict(top_pressure_pa=.001,terrain_std_m=300.)
    result=jax.jit(lambda:column_physics(**kwargs,terrain_drag_options=options))()
    plain=column_physics(**kwargs)
    assert np.isfinite(result.temperature).all()
    assert np.max(np.abs(result.u-plain.u))>1e-8
    assert result.diagnostics['terrain_drag']['surface_stress_pa']>0.
