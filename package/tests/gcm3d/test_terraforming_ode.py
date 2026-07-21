"""Tests for src.gcm3d.terraforming_ode — the "0-D physics on the dinosaur
substrate" prototype (requires the optional 'gcm3d' extra).

This is the go/no-go for making dinosaur the engine backbone: it checks that the
existing torch 0-D coupled ODE, re-expressed as a dinosaur ImplicitExplicitODE,
(1) reproduces the torch tendencies to machine precision, (2) integrates stably
under dinosaur's stepper and tracks a torch RK4 rollout to truncation order,
(3) is differentiable end-to-end, and (4) batches with vmap. Torch is imported
here (the experiment layer) to compare against ``Mars.compute_derivatives`` —
the module under test stays torch-free.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from src.celestials.planets.mars import Mars  # noqa: E402
from src.gcm3d import integrate  # noqa: E402
from src.gcm3d.terraforming_ode import (  # noqa: E402
    T_IDX,
    ZeroDForcing,
    stepper,
    tendency,
    terraforming_ode,
)

jax.config.update("jax_enable_x64", True)


def _forcing_from_mars(mars: Mars, smooth_gates: bool = True) -> ZeroDForcing:
    """Read the cached constants + epoch state a Mars uses in compute_derivatives."""
    s = mars
    return ZeroDForcing(
        albedo=float(s.radiation.albedo),
        solar_flux=float(s.radiation.solar_flux),
        greenhouse_factor=float(s.thermal.greenhouse_factor),
        emissivity=float(s._EMISS),
        stefan_boltzmann=float(s._SB),
        thermal_inertia=float(s._TI),
        radius_m=float(s.intrinsic_params.radius),
        gravity_m_s2=float(s.intrinsic_params.gravity),
        rotation_period_s=float(s.intrinsic_params.rotation_period),
        axial_tilt_rad=float(s.orbital_params.axial_tilt),
        ls_perihelion_rad=float(s._LS_PERI),
        latitude_rad=float(s._init_latitude),
        elapsed_time_s=float(s.elapsed_time),
        orbital_angle_rad=float(s.orbital_angle),
        cap_fraction=float(s._CAP_FRAC),
        q_out_pole=float(s._Q_out_pole),
        latent_heat=float(s._LAT_HEAT),
        ice_north_kg=float(s.water.ice_mass_north),
        ice_south_kg=float(s.water.ice_mass_south),
        ice_ref_kg=float(s._ICE_REF),
        escape_rate_kg_s=float(s._ESCAPE_RATE),
        smooth_gates=smooth_gates,
    )


@pytest.fixture(scope="module")
def smooth_mars():
    return Mars(smooth_gates=True)


# ── Tendency parity: the physics port is exact ────────────────────────────────

class TestTendencyParity:
    """The JAX tendency must equal the torch compute_derivatives kernel."""

    @pytest.mark.parametrize("smooth", [True, False], ids=["smooth-gate", "hard-gate"])
    def test_matches_torch_to_machine_precision(self, smooth):
        mars = Mars(smooth_gates=smooth)
        f = _forcing_from_mars(mars, smooth_gates=smooth)
        rng = np.random.default_rng(0)
        for _ in range(50):
            y = np.array(
                [rng.uniform(150, 300), rng.uniform(200, 1200), rng.uniform(0, 1e16)]
            )
            jax_dy = np.asarray(tendency(jnp.asarray(y), f))
            torch_dy = mars.compute_derivatives(
                torch.tensor(y, dtype=torch.float64)
            ).numpy()
            # float64 machine precision — this is a line-for-line port.
            assert np.allclose(jax_dy, torch_dy, rtol=0, atol=1e-12)


# ── Integration on dinosaur's stepper ─────────────────────────────────────────

class TestRollout:

    def test_rollout_is_stable_and_finite(self, smooth_mars):
        f = _forcing_from_mars(smooth_mars)
        step = stepper(terraforming_ode(f), dt_seconds=100.0)
        final = integrate(step, jnp.array([210.0, 610.0, 1.0e16]), 500)
        assert bool(jnp.all(jnp.isfinite(final)))
        assert 100.0 < float(final[T_IDX]) < 400.0  # temperature stays physical

    def test_tracks_torch_rk4_to_truncation_order(self, smooth_mars):
        # dinosaur imex_rk_sil3 (explicit part) vs a torch RK4, same frozen
        # forcing: different schemes, so they agree to O(dt^p), not exactly.
        f = _forcing_from_mars(smooth_mars)
        y0 = np.array([210.0, 610.0, 1.0e16])
        dt, n = 100.0, 500

        dino = np.asarray(integrate(stepper(terraforming_ode(f), dt), jnp.asarray(y0), n))

        def deriv(y):
            return smooth_mars.compute_derivatives(
                torch.tensor(y, dtype=torch.float64)
            ).numpy()

        y = y0.copy()
        for _ in range(n):
            k1 = deriv(y)
            k2 = deriv(y + 0.5 * dt * k1)
            k3 = deriv(y + 0.5 * dt * k2)
            k4 = deriv(y + dt * k3)
            y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

        assert np.allclose(dino, y, rtol=1e-6)

    def test_integrate_rejects_nonpositive_steps(self, smooth_mars):
        f = _forcing_from_mars(smooth_mars)
        step = stepper(terraforming_ode(f), dt_seconds=100.0)
        with pytest.raises(ValueError):
            integrate(step, jnp.array([210.0, 610.0, 1.0e16]), 0)


# ── The reason for the whole exercise: differentiable + batchable ─────────────

class TestDifferentiableAndBatched:

    def _loss(self, smooth_mars):
        f = _forcing_from_mars(smooth_mars)
        step = stepper(terraforming_ode(f), dt_seconds=100.0)

        def loss(alpha):
            # Well-scaled diagnostic: final surface temperature only (avoids the
            # M_ice ~ 1e16 term swamping a finite-difference check).
            return integrate(step, jnp.array([210.0 * alpha, 610.0, 1.0e16]), 200)[
                T_IDX
            ] ** 2

        return loss

    def test_grad_through_rollout_matches_finite_difference(self, smooth_mars):
        loss = self._loss(smooth_mars)
        g = float(jax.grad(loss)(1.0))
        assert jnp.isfinite(g) and abs(g) > 0.0
        eps = 1e-5
        fd = (loss(1.0 + eps) - loss(1.0 - eps)) / (2 * eps)
        assert g == pytest.approx(float(fd), rel=1e-5)

    def test_vmap_batches_many_trajectories(self, smooth_mars):
        f = _forcing_from_mars(smooth_mars)
        step = stepper(terraforming_ode(f), dt_seconds=100.0)
        batch = jnp.stack(
            [jnp.array([T, 610.0, 1.0e16]) for T in np.linspace(180.0, 260.0, 256)]
        )
        out = jax.jit(jax.vmap(lambda y: integrate(step, y, 200)))(batch)
        assert out.shape == (256, 3)
        assert bool(jnp.all(jnp.isfinite(out)))
