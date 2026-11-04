"""Paired restart and exact annual-boundary accounting independent of climate skill."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
import jax
jax.config.update('jax_enable_x64',True)
from src.terraforming.aerosols.paired import initial_warming,accumulate,warming_report,prepare_pair,stream_pair
from src.terraforming.aerosols.checkpoint import save,load


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
    from src.terraforming.aerosols import aluminium_rod
    from src.terraforming.aerosols.state import initialize
    from src.terraforming.aerosols.paired import PairedState
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
