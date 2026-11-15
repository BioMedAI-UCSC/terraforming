"""Paired restart and exact annual-boundary accounting independent of climate skill."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
import jax
jax.config.update('jax_enable_x64',True)
from src.interventions.aerosols.paired import initial_warming,accumulate,warming_report,prepare_pair,stream_pair
from src.interventions.aerosols.checkpoint import save,load


def test_nonuniform_linear_signal_integrates_across_years():
    state=initial_warming(7.,0.)
    for stop in (10.,28.,39.,47.):
        state=accumulate(state,stop,2*(stop-7),10.)
    # Difference is 2t over four complete periods; annual means are 10,30,50,70 K.
    report=warming_report(state,10.)
    assert report['completed_mars_years']==4
    assert report['paired_time_mean_warming_k']==pytest.approx(40.)
    np.testing.assert_allclose(report['consecutive_annual_warming_k'],[30.,50.,70.])
    assert report['sustained_thresholds_k']['20']
    assert np.shape(state.last_three_means_k)==(3,)


def test_partial_and_weak_years_never_confirm_threshold():
    state=initial_warming(0.,0.)
    state=accumulate(state,10.,0.,10.)
    state=accumulate(state,20.,80.,10.)
    partial=accumulate(state,29.,80.,10.)
    assert not any(warming_report(partial,10.)['sustained_thresholds_k'].values())
    state=accumulate(partial,30.,80.,10.)
    report=warming_report(state,10.)
    assert report['complete_three_mars_years']
    assert not any(report['sustained_thresholds_k'].values())
    assert report['paired_time_mean_warming_k']>20


def test_paired_full_restart_at_source_boundary(tmp_path):
    initial,steps,context=prepare_pair(dt_s=5.,n_layers=4,sources=[
        dict(start_s=3.,stop_s=8.,rate=10.)])
    configuration=context.pop('configuration')
    year=configuration['cases']['active']['forcing']['orbital_period_s']
    for mode in ('off','passive'):
        np.testing.assert_array_equal(getattr(initial,mode).climate.surface_temperature,
            initial.active.climate.surface_temperature)
    first,report=list(stream_pair(initial,steps,1,**context,orbital_period_s=year))[0]
    save(tmp_path/'pair.npz',first,configuration)
    restored=load(tmp_path/'pair.npz',initial,configuration)
    final,report=list(stream_pair(first,steps,2,**context,orbital_period_s=year))[-1]
    resumed,resumed_report=list(stream_pair(restored,steps,2,**context,orbital_period_s=year))[-1]
    for a,b in zip(jax.tree_util.tree_leaves(final),jax.tree_util.tree_leaves(resumed)):
        np.testing.assert_array_equal(a,b)
    assert resumed_report==report
    assert report['cases']['active']['injected_mass_kg']==50.
    assert report['cases']['passive']['injected_mass_kg']==50.
    assert report['cases']['off']['injected_mass_kg']==0.
    assert not report['warming']['complete_three_mars_years']
    np.testing.assert_allclose(final.passive.climate.surface_temperature,final.off.climate.surface_temperature,atol=1e-12)


def test_failed_pair_preserves_all_last_valid_states():
    import dataclasses
    import jax.numpy as jnp
    from .test_aerosol_state import setup_state
    from src.celestials.planets.mars import MARS_BODY_3D
    from src.interventions.aerosols import aluminium_rod
    from src.interventions.aerosols.state import initialize
    from src.interventions.aerosols.paired import PairedState
    coords,specs,climate=setup_state();particle=aluminium_rod()
    state=initialize(climate,coords,specs,MARS_BODY_3D,particle)
    pair=PairedState(state,state,state,initial_warming(0.))
    def nonfinite(s):
        dynamics=dataclasses.replace(s.climate.dynamics,vorticity=jnp.full_like(s.climate.dynamics.vorticity,jnp.nan))
        return s._replace(climate=s.climate._replace(dynamics=dynamics),elapsed_seconds=s.elapsed_seconds+5.)
    steps=dict(off=nonfinite,passive=lambda s:s,active=lambda s:s)
    retained,report=list(stream_pair(pair,steps,1,coords,specs,MARS_BODY_3D,particle,orbital_period_s=100.))[0]
    assert retained is pair
    assert report['status']=='numerical_failure'
    assert report['failed_cases']==['off']
    assert float(retained.warming.last_s)==0.
    invalid=state._replace(ledger=state.ledger._replace(invalid=jnp.asarray(True)))
    _,report=list(stream_pair(pair,dict(off=lambda s:invalid,passive=lambda s:s,active=lambda s:s),
        1,coords,specs,MARS_BODY_3D,particle,orbital_period_s=100.))[0]
    assert report['status']=='invalid_regime'
    overflow=state._replace(ledger=state.ledger._replace(numerical_dry_energy_correction_j=jnp.asarray(jnp.inf)))
    _,report=list(stream_pair(pair,dict(off=lambda s:overflow,passive=lambda s:s,active=lambda s:s),
        1,coords,specs,MARS_BODY_3D,particle,orbital_period_s=100.))[0]
    assert report['status']=='numerical_failure'


def test_physical_start_preserves_absolute_phase_and_checks_contract(tmp_path):
    import dataclasses
    import jax.numpy as jnp
    from .test_aerosol_state import setup_state
    from src.interventions.aerosols.pilot import prepare
    from src.interventions.aerosols.starting import save_physical_start
    _,_,climate=setup_state()
    climate=climate._replace(surface_temperature=climate.surface_temperature+3.,
        dynamics=dataclasses.replace(climate.dynamics,sim_time=jnp.asarray(123.)))
    _,_,context=prepare(dt_s=5.,n_layers=4)
    contract=context['configuration']['start_contract']
    path=tmp_path/'physical.npz'
    save_physical_start(path,climate,contract)
    pair,_,context=prepare_pair(dt_s=5.,n_layers=4,initial_climate=path)
    for state in (pair.off,pair.passive,pair.active):
        np.testing.assert_array_equal(state.climate.surface_temperature,climate.surface_temperature)
        assert float(state.climate.dynamics.sim_time)==123.
        assert float(state.elapsed_seconds)==0.
    assert context['configuration']['cases']['active']['initial_climate_sha256']
    with pytest.raises(ValueError,match='configuration'):
        prepare(dt_s=5.,n_layers=4,initial_climate=path,co2_lw_scale=.5)
    with pytest.raises(ValueError,match='tracer-free'):
        save_physical_start(path,pair.active.climate,contract)


def test_projection_energy_uses_existing_dry_inventory_definition():
    import dataclasses
    import jax.numpy as jnp
    from .test_aerosol_state import setup_state
    from src.celestials.planets.mars import MARS_BODY_3D as body
    from src.framework.gcm.benchmarks import dry_conserved_arrays,dry_conserved_quantities
    coords,specs,climate=setup_state()
    reference=np.full(coords.vertical.layers,body.reference_temperature_k)
    old=dry_conserved_arrays(climate.dynamics,coords,specs,body,reference)
    host=dry_conserved_quantities(climate.dynamics,coords,specs,body,reference)
    np.testing.assert_allclose(old,[host.mass_kg_m2_sr,host.total_energy_j_m2_sr,
        host.axial_angular_momentum_kg_m_s_sr],rtol=1e-14)
    grid=coords.horizontal
    logp=climate.dynamics.log_surface_pressure+grid.to_modal(jnp.full((1,)+grid.nodal_shape,np.log(1.01)))
    changed=dataclasses.replace(climate.dynamics,log_surface_pressure=logp)
    new=dry_conserved_arrays(changed,coords,specs,body,reference)
    assert float(new[1])==pytest.approx(1.01*float(old[1]),rel=1e-12)
