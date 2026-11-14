"""Exercise the application's actual filter wiring without hiding loss in projection."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mars_calibration import driver
from src.framework.gcm import dynamics
from src.framework.gcm._dinosaur import primitive_equations
from src.framework.physics.gcm import ColumnPhysicsState


@pytest.mark.parametrize("co2_exchange", [False, True])
def test_calibration_projection_uses_the_selected_co2_cycle(monkeypatch, co2_exchange):
    coords = driver.coordinate_system("T21", 4)
    specs = driver.physics_specs(driver.MARS_BODY_3D)
    captured = {}
    monkeypatch.setattr(driver, "forced_co2_primitive_equations", lambda *a, **k: None)
    monkeypatch.setattr("src.framework.physics.gcm.forced_primitive_equations", lambda *a, **k: None)
    monkeypatch.setattr(driver, "stepper", lambda *a, **k: lambda state: state)

    def projection(advance, *args, **kwargs):
        captured.update(kwargs)
        return advance

    monkeypatch.setattr(driver, "positivity_preserving_co2_step", projection)
    driver._build_step(coords, specs, driver.radiative_forcing(), 300.,
                       orography=driver.jnp.zeros(coords.horizontal.modal_shape),
                       co2_exchange=co2_exchange)
    assert (captured["co2_forcing"] is not None) == co2_exchange
    assert captured["dt_seconds"] == 300.


def test_calibration_filter_excludes_pressure_and_tracers_before_projection(monkeypatch):
    coords = driver.coordinate_system("T21", 4)
    specs = driver.physics_specs(driver.MARS_BODY_3D)
    grid = coords.horizontal
    modes = driver.jnp.zeros((4,) + grid.modal_shape).at[:, 0, -1].set(1.)
    # Include pressure perturbations: the historical all-dynamics filter would
    # change the spherical integral of exp(log pressure) even on an identity step.
    pressure = driver.jnp.zeros((1,) + grid.modal_shape).at[:, 0, -1].set(.01)
    state = primitive_equations.State(
        vorticity=modes, divergence=modes, temperature_variation=modes,
        log_surface_pressure=pressure, tracers={"dust": modes}, sim_time=0.,
    )
    surface = driver.jnp.ones((1,) + grid.nodal_shape)
    column = ColumnPhysicsState(state, surface * 200, surface * 3, surface * 190)
    monkeypatch.setattr(dynamics.time_integration, "imex_rk_sil3", lambda *_: lambda s: s)
    monkeypatch.setattr(driver, "forced_co2_primitive_equations", lambda *a, **k: None)
    # Bypass the mass fixer: this test must see the raw filtered pressure.
    monkeypatch.setattr(driver, "positivity_preserving_co2_step", lambda advance, *a, **k: advance)
    advance = driver._build_step(coords, specs, driver.radiative_forcing(), 300.,
                                 orography=driver.jnp.zeros(grid.modal_shape), diffusion_sols=.1)
    result = driver.jax.jit(advance)(column)
    expected_decay = np.exp(-300 / (.1 * driver.MARS_BODY_3D.solar_day_seconds))
    np.testing.assert_allclose(result.dynamics.temperature_variation[:, 0, -1], expected_decay)
    np.testing.assert_array_equal(result.dynamics.log_surface_pressure, pressure)
    np.testing.assert_array_equal(result.dynamics.tracers["dust"], modes)
    before = np.exp(np.asarray(grid.to_nodal(pressure)))
    after = np.exp(np.asarray(grid.to_nodal(result.dynamics.log_surface_pressure)))
    weights = np.asarray(grid.quadrature_weights)
    np.testing.assert_array_equal(np.sum(before * weights), np.sum(after * weights))
    for name in ("surface_temperature", "co2_ice", "ground_temperature"):
        np.testing.assert_array_equal(getattr(result, name), getattr(column, name))
