"""P0.7 restart equivalence and averaging-window tests."""

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.framework.gcm.coordinates import coordinate_system  # noqa: E402
from src.framework.gcm.dynamics import integrate, stepper  # noqa: E402
from src.framework.physics.gcm import (  # noqa: E402
    forced_primitive_equations,
    initial_column_state,
)
from src.celestials.planets.mars.gcm import radiative_forcing  # noqa: E402
from src.framework.gcm.restart import (  # noqa: E402
    integrate_with_averaging,
    load_restart,
    save_restart,
)
from src.framework.gcm.specs import physics_specs  # noqa: E402
from src.framework.gcm._dinosaur import jnp, primitive_equations, scales  # noqa: E402

_u = scales.units


def _model():
    coords = coordinate_system("T21", 4)
    specs = physics_specs(MARS_BODY_3D)
    grid = coords.horizontal
    zeros = jnp.zeros((4,) + grid.modal_shape)
    ps = float(specs.nondimensionalize(610.0 * _u.pascal))
    dyn = primitive_equations.State(
        vorticity=zeros,
        divergence=zeros,
        temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(
            jnp.full((1,) + grid.nodal_shape, np.log(ps))
        ),
        tracers={"marker": grid.to_modal(jnp.ones((4,) + grid.nodal_shape))},
        sim_time=0.0,
    )
    state = initial_column_state(dyn, coords, 200.0, specs)
    equation = forced_primitive_equations(
        coords, MARS_BODY_3D, radiative_forcing(diurnal=False), specs=specs
    )
    return state, jax.jit(stepper(equation, 300.0, specs))


def test_restart_matches_continuous_rollout(tmp_path):
    state, advance = _model()
    continuous = integrate(advance, state, 10)
    first = integrate(advance, state, 4)
    restored = load_restart(save_restart(first, tmp_path / "state.npz"))
    restarted = integrate(advance, restored, 6)
    for expected, actual in zip(
        jax.tree_util.tree_leaves(continuous),
        jax.tree_util.tree_leaves(restarted), strict=True,
    ):
        assert np.array_equal(np.asarray(expected), np.asarray(actual))


def test_spinup_and_averaging_window_matches_manual_samples():
    state, advance = _model()
    diagnostic = lambda s: jnp.mean(s.surface_temperature)
    final, mean = integrate_with_averaging(
        advance, state, spinup_steps=2, average_steps=6,
        sample_every=2, diagnostic_fn=diagnostic,
    )
    manual = integrate(advance, state, 2)
    values = []
    for _ in range(3):
        manual = integrate(advance, manual, 2)
        values.append(diagnostic(manual))
    assert np.array_equal(np.asarray(final.surface_temperature),
                          np.asarray(manual.surface_temperature))
    assert float(mean) == pytest.approx(float(jnp.mean(jnp.asarray(values))))
