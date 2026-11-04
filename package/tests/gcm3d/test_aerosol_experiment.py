import dataclasses
import numpy as np
import pytest
pytest.importorskip('dinosaur')
import jax
jax.config.update('jax_enable_x64',True)
from .test_aerosol_state import setup_state
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars import gcm as mars_gcm
from src.framework.physics.gcm import CO2Forcing
from src.terraforming.aerosols import aluminium_rod, Injection
from src.terraforming.aerosols.state import initialize
from src.terraforming.aerosols.experiment import coupled_step, ExperimentOptions, diagnostics


def test_scheduled_coupled_release_and_shutdown_budget(tmp_path):
    coords,specs,climate=setup_state(); particle=aluminium_rod()
    state=initialize(climate,coords,specs,BODY,particle)
    forcing=dataclasses.replace(mars_gcm.radiative_forcing(co2_radiation_enabled=True),emissivity=1.)
    step=coupled_step(coords,specs,BODY,forcing,mars_gcm.co2_forcing(),particle,
        [Injection(3.,8.,10.)],ExperimentOptions(dt_s=5.))
    from src.terraforming.aerosols.checkpoint import save,load
    state=step(state)
    path=tmp_path/'boundary.npz';config={'source_start_s':3.,'source_stop_s':8.}
    save(path,state,config)
    resumed=load(path,state,config)
    for _ in range(2):
        state=step(state);resumed=step(resumed)
    for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(a,b)
    report=diagnostics(state,coords,specs,BODY,particle)
    assert report['elapsed_seconds']==15.
    assert report['injected_mass_kg']==pytest.approx(50.)
    assert abs(report['material_budget_error_kg'])<1e-8
    assert report['status']=='running'
    assert report['production_energy_j'] is None


def test_unsupported_active_aggregation_rejected():
    with pytest.raises(ValueError,match='aggregate optics'):
        ExperimentOptions(mode='active',coagulation=True)
    with pytest.raises(ValueError,match='named source'):
        ExperimentOptions(collection_rate_s=.1)


def test_complete_checkpoint_and_identity_gate(tmp_path):
    from src.terraforming.aerosols.checkpoint import save,load
    coords,specs,climate=setup_state()
    state=initialize(climate,coords,specs,BODY,aluminium_rod(),1e-6)
    config=dict(body='Mars',source_stop_s=8,asset_sha256='verified-test-identity')
    path=tmp_path/'restart.npz'
    save(path,state,config)
    restored=load(path,state,config)
    for a,b in zip(jax.tree_util.tree_leaves(state),jax.tree_util.tree_leaves(restored)):
        np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError,match='configuration'):
        load(path,state,{**config,'source_stop_s':9})


def test_incomplete_warming_is_never_success():
    from src.terraforming.aerosols.reporting import WarmingWindow
    window=WarmingWindow(100.)
    assert not any(window.add(0,100,200,225)['sustained_thresholds_k'].values())
    window.add(100,200,200,225)
    result=window.add(200,300,200,225)
    assert result['complete_three_mars_years']
    assert all(result['sustained_thresholds_k'].values())


def test_zero_particles_preserve_gcm_step():
    from src.framework.physics.gcm import forced_co2_primitive_equations,positivity_preserving_co2_step
    from src.framework.gcm.dynamics import stepper
    coords,specs,climate=setup_state();particle=aluminium_rod()
    forcing=dataclasses.replace(mars_gcm.radiative_forcing(co2_radiation_enabled=True),emissivity=1.)
    co2=mars_gcm.co2_forcing()
    reference=positivity_preserving_co2_step(stepper(forced_co2_primitive_equations(
        coords,BODY,forcing,co2,specs=specs),5.,specs),coords,specs)(climate)
    step=coupled_step(coords,specs,BODY,forcing,co2,particle,[],ExperimentOptions(mode='off',dt_s=5.))
    actual=step(initialize(climate,coords,specs,BODY,particle)).climate
    actual=actual._replace(dynamics=dataclasses.replace(actual.dynamics,tracers={}))
    for a,b in zip(jax.tree_util.tree_leaves(reference),jax.tree_util.tree_leaves(actual)):
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-12)
