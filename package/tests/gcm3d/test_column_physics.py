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
