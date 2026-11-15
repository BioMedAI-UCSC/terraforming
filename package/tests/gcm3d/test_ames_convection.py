import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax
import jax.numpy as jnp

from src.framework.physics.ames_convection import pass2, adjust_state, build_step

jax.config.update("jax_enable_x64", True)


def column():
    half = np.array([0.,40.,130.,350.,610.])[:,None]
    full = .5*(half[:-1]+half[1:])
    temperature = np.array([100.,120.,160.,240.])[:,None]
    u = np.array([10.,-20.,30.,-40.])[:,None]
    tracers = np.arange(8,dtype=float).reshape(4,1,2)*1e-5
    return temperature,u,-u,tracers,half,full,.25


def test_pass2_conserves_heat_momentum_and_tracers_on_nonuniform_layers():
    values = column()
    adjusted = jax.jit(pass2)(*values)
    weights = np.diff(values[4],axis=0)
    for before, after in zip(values[:4],adjusted,strict=True):
        w = weights[...,None] if before.ndim == 3 else weights
        np.testing.assert_allclose(np.sum(before*w,axis=0),np.sum(after*w,axis=0),rtol=2e-14,atol=2e-14)
    assert np.max(np.abs(adjusted[0]-values[0])) > 1.
    assert np.min(adjusted[3]) >= 0.


def test_stable_columns_and_empty_tracers_are_unchanged():
    t,u,v,r,half,full,kappa = column()
    t = 200.*(full/610.)**kappa
    # A strict stable gradient avoids a roundoff-dependent neutral branch.
    t *= np.linspace(1.2,1.,4)[:,None]
    r = np.empty(t.shape+(0,))
    adjusted = pass2(t,u,v,r,half,full,kappa)
    for before,after in zip((t,u,v,r),adjusted,strict=True):
        np.testing.assert_array_equal(before,after)


def test_reverse_derivative_matches_finite_difference_away_from_switch():
    values = column()
    def objective(scale):
        t,u,v,r = pass2(values[0]*scale,*values[1:])
        return jnp.sum(t*t) + jnp.sum(u*u) + jnp.sum(v*v) + jnp.sum(r*r)
    derivative = jax.grad(objective)(1.)
    for epsilon in (1e-4,1e-5,1e-6):
        finite = (objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(derivative,finite,rtol=2e-8)


def test_pass2_rejects_inconsistent_grid():
    values = list(column())
    values[4] = values[4][:-1]
    with pytest.raises(ValueError,match="pressures"):
        pass2(*values)


def test_dinosaur_adapter_preserves_inventories_and_reverse_derivative():
    from .test_aerosol_state import setup_state
    from src.celestials.planets.mars import MARS_BODY_3D as body
    from src.framework.gcm.dynamics import reference_temperature
    coords,specs,state = setup_state()
    grid = coords.horizontal
    values = column()
    temperature = jnp.broadcast_to(jnp.asarray(values[0])[:,:,None],
        (coords.vertical.layers,)+grid.nodal_shape)
    ref = jnp.asarray(reference_temperature(coords,body))[:,None,None]
    state = state._replace(dynamics=dataclasses.replace(state.dynamics,
        temperature_variation=grid.to_modal(temperature-ref)))
    adjusted = jax.jit(lambda s:adjust_state(s,coords,specs,body))(state)
    np.testing.assert_array_equal(adjusted.dynamics.log_surface_pressure,state.dynamics.log_surface_pressure)
    for name in ("surface_temperature","co2_ice","ground_temperature","surface_momentum"):
        np.testing.assert_array_equal(getattr(adjusted,name),getattr(state,name))
    def objective(scale):
        changed=state._replace(dynamics=dataclasses.replace(state.dynamics,
            temperature_variation=grid.to_modal(temperature*scale-ref)))
        result=adjust_state(changed,coords,specs,body)
        return jnp.sum(grid.to_nodal(result.dynamics.temperature_variation)**2)
    auto=jax.grad(objective)(1.)
    finite=(objective(1.+1e-5)-objective(1.-1e-5))/2e-5
    np.testing.assert_allclose(auto,finite,rtol=2e-8)


def test_complete_step_is_differentiable_and_does_not_double_apply_convection():
    from .test_aerosol_state import setup_state
    from src.celestials.planets.mars import MARS_BODY_3D as body
    from src.celestials.planets.mars import gcm as mars_gcm
    coords,specs,state=setup_state()
    forcing=dataclasses.replace(mars_gcm.radiative_forcing(),convective_adjustment_enabled=True)
    enabled=build_step(coords,specs,body,forcing,mars_gcm.co2_forcing(),5.)
    disabled=build_step(coords,specs,body,dataclasses.replace(forcing,
        convective_adjustment_enabled=False),mars_gcm.co2_forcing(),5.)
    a,b=jax.jit(enabled)(state),jax.jit(disabled)(state)
    for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b),strict=True):
        np.testing.assert_array_equal(x,y)
    def objective(offset):
        result=enabled(state._replace(surface_temperature=state.surface_temperature+offset))
        return jnp.mean(result.surface_temperature)
    auto=jax.grad(objective)(0.)
    finite=(objective(1e-4)-objective(-1e-4))/2e-4
    np.testing.assert_allclose(auto,finite,rtol=2e-8)
