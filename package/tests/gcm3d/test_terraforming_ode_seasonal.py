"""Tests for the seasonal (time-advancing) 0-D terraforming ODE in
``src.gcm3d.terraforming_ode`` (requires the optional 'gcm3d' extra).

Where ``test_terraforming_ode.py`` covers the *frozen-epoch* prototype, this
file covers the integrated version that carries elapsed time ``t`` in the state
and advances the orbit inside the ODE — the piece that makes a rollout produce a
full areocentric-solar-longitude (``Ls``) seasonal cycle:

  (1) Kernel parity: ``seasonal_tendency`` reproduces main's two-cap
      ``Mars.compute_derivatives`` to float64 precision at arbitrary orbit phase.
  (2) Ls outputs: a one-orbit rollout sweeps ``Ls`` across ``[0, 360)`` and the
      solar flux peaks at perihelion / troughs at aphelion.
  (3) The rollout stays finite/physical and is differentiable end-to-end.

Torch is imported here (the experiment layer) to compare against the torch
kernel; the module under test stays torch-free.
"""

from __future__ import annotations


import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import torch  # noqa: E402

from src.celestials.planets.mars import Mars  # noqa: E402
from src.gcm3d.terraforming_ode import (  # noqa: E402
    SeasonalForcing,
    SeasonalTrajectory,
    initial_seasonal_state,
    orbital_angle,
    run_seasonal,
    seasonal_tendency,
    solar_flux,
)

jax.config.update("jax_enable_x64", True)


def _seasonal_forcing_from_mars(mars: Mars, smooth_gates: bool = True) -> SeasonalForcing:
    """Read the cached constants + orbital elements a Mars uses."""
    s = mars
    return SeasonalForcing(
        albedo=float(s.radiation.albedo),
        greenhouse_factor=float(s.thermal.greenhouse_factor),
        emissivity=float(s._EMISS),
        stefan_boltzmann=float(s._SB),
        thermal_inertia=float(s._TI),
        radius_m=float(s.intrinsic_params.radius),
        gravity_m_s2=float(s.intrinsic_params.gravity),
        rotation_period_s=float(s.intrinsic_params.rotation_period),
        latitude_rad=float(s._init_latitude),
        axial_tilt_rad=float(s.orbital_params.axial_tilt),
        ls_perihelion_rad=float(s._LS_PERI),
        orbital_period_s=float(s.orbital_params.orbital_period),
        semi_major_axis_m=float(s.orbital_params.semi_major_axis),
        eccentricity=float(s.orbital_params.eccentricity),
        init_orbital_angle_rad=float(s.orbital_angle),
        cap_fraction=float(s._CAP_FRAC),
        q_out_pole=float(s._Q_out_pole),
        latent_heat=float(s._LAT_HEAT),
        ice_ref_kg=float(s._ICE_REF),
        escape_rate_kg_s=float(s._ESCAPE_RATE),
        smooth_gates=smooth_gates,
    )


@pytest.fixture(scope="module")
def mars() -> Mars:
    return Mars()


# ── (1) Kernel parity vs the torch two-cap kernel ─────────────────────────────

class TestSeasonalTendencyParity:
    """seasonal_tendency[:4] must match Mars.compute_derivatives at any phase."""

    @pytest.mark.parametrize("smooth", [True, False], ids=["smooth-gate", "hard-gate"])
    def test_matches_torch_two_cap_kernel(self, mars, smooth):
        f = _seasonal_forcing_from_mars(mars, smooth_gates=smooth)
        rng = np.random.default_rng(0)
        period = f.orbital_period_s

        # Sweep several orbit phases so h, delta and the solar flux all vary.
        for t in np.linspace(0.0, period, 8, endpoint=False):
            T, P = rng.uniform(150, 300), rng.uniform(200, 1200)
            mN, mS = rng.uniform(0, 1e16), rng.uniform(0, 1e16)

            # JAX seasonal tendency at this elapsed time.
            y = initial_seasonal_state(T, P, mN, mS, t0_s=float(t))
            jax_dy = np.asarray(seasonal_tendency(y, f))

            # Torch reference: pin the orbit state to the same phase and use the
            # identical solar flux the seasonal kernel computes internally, so
            # this isolates the port from how the flux is sourced.
            mars._smooth_gates = smooth
            mars.elapsed_time = torch.tensor(float(t), dtype=torch.float64)
            mars.orbital_angle = torch.tensor(float(orbital_angle(float(t), f)), dtype=torch.float64)
            mars.radiation.solar_flux = torch.tensor(
                float(solar_flux(float(t), f)), dtype=torch.float64
            )
            torch_dy = mars.compute_derivatives(
                torch.tensor([T, P, mN, mS], dtype=torch.float64)
            ).numpy()

            # First four components (T, P, M_N, M_S) are the physics port; the
            # 5th is the trivial dt/dt = 1.
            assert np.allclose(jax_dy[:4], torch_dy, rtol=1e-14, atol=1e-6)
            assert jax_dy[4] == pytest.approx(1.0)


# ── (2) Ls outputs: seasonal sweep + flux extremes ────────────────────────────

class TestLsOutputs:

    def test_ls_sweeps_full_year(self, mars):
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        # One full orbital period at a diurnally-resolved step (~45 steps/sol),
        # sampled down to ~600 points across the year.
        n_steps = 30000
        dt = f.orbital_period_s / n_steps
        traj = run_seasonal(f, y0, dt_seconds=dt, n_steps=n_steps, sample_every=50)

        assert isinstance(traj, SeasonalTrajectory)
        assert traj.ls_deg.min() < 5.0
        assert traj.ls_deg.max() > 355.0
        # Ls covers the whole circle roughly uniformly (no large gap).
        gaps = np.diff(np.sort(traj.ls_deg))
        assert gaps.max() < 5.0

    def test_solar_flux_peaks_at_perihelion(self, mars):
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        n_steps = 30000
        dt = f.orbital_period_s / n_steps
        traj = run_seasonal(f, y0, dt_seconds=dt, n_steps=n_steps, sample_every=50)

        # Analytic perihelion/aphelion fluxes (θ = 0 and π).
        a, e = f.semi_major_axis_m, f.eccentricity
        peri = f.tsi_1au_w_m2 * (f.au_m / (a * (1 - e))) ** 2
        aph = f.tsi_1au_w_m2 * (f.au_m / (a * (1 + e))) ** 2
        assert traj.solar_flux_wm2.max() == pytest.approx(peri, rel=1e-3)
        assert traj.solar_flux_wm2.min() == pytest.approx(aph, rel=1e-3)
        # ~45% seasonal swing in insolation for Mars's eccentricity.
        assert peri / aph == pytest.approx(((1 + e) / (1 - e)) ** 2, rel=1e-6)

    def test_write_csv_has_ls_column(self, mars, tmp_path):
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        traj = run_seasonal(f, y0, dt_seconds=1000.0, n_steps=50, sample_every=5)
        out = traj.write_csv(tmp_path / "mars_seasonal.csv")
        assert out.exists()
        header = out.read_text().splitlines()[0]
        assert "ls_deg" in header and "temperature_k" in header


# ── (3) Stability + differentiability of the rollout ──────────────────────────

class TestSeasonalRollout:

    def test_rollout_stays_finite_and_physical(self, mars):
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        traj = run_seasonal(f, y0, dt_seconds=2000.0, n_steps=500, sample_every=10)
        assert np.all(np.isfinite(traj.temperature_k))
        assert np.all(traj.temperature_k > 100.0) and np.all(traj.temperature_k < 400.0)
        assert np.all(traj.pressure_pa >= 0.0)
        assert np.all(traj.ice_north_kg >= 0.0) and np.all(traj.ice_south_kg >= 0.0)

    def test_run_seasonal_rejects_bad_args(self, mars):
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        with pytest.raises(ValueError):
            run_seasonal(f, y0, dt_seconds=100.0, n_steps=0)
        with pytest.raises(ValueError):
            run_seasonal(f, y0, dt_seconds=100.0, n_steps=10, sample_every=0)

    def test_run_seasonal_keeps_remainder_steps(self, mars):
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        traj = run_seasonal(f, y0, dt_seconds=100.0, n_steps=12, sample_every=5)
        assert len(traj.time_s) == 3
        assert traj.time_s[-1] == pytest.approx(1200.0)

    def test_run_seasonal_rejects_diurnally_coarse_dt(self, mars):
        # Regression: dt ~ 1 sol aliases the diurnal T^4 balance and diverges to
        # non-physical temperatures; guard must refuse it rather than emit
        # silently-wrong seasonal output.
        f = _seasonal_forcing_from_mars(mars)
        y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15)
        coarse = f.rotation_period_s  # ~1 step per rotation
        with pytest.raises(ValueError, match="diurnal"):
            run_seasonal(f, y0, dt_seconds=coarse, n_steps=10)

    def test_grad_flows_through_seasonal_rollout(self, mars):
        f = _seasonal_forcing_from_mars(mars)
        from src.gcm3d.terraforming_ode import seasonal_ode, stepper
        from src.gcm3d import integrate

        step = stepper(seasonal_ode(f), dt_seconds=1000.0)

        def loss(t0_scale):
            y0 = initial_seasonal_state(210.0, 610.0, 5.0e15, 5.0e15, t0_s=0.0)
            # Perturb initial temperature to get a nonzero, well-scaled gradient.
            y0 = y0.at[0].set(210.0 * t0_scale)
            final = integrate(step, y0, 200)
            return final[0]

        g = float(jax.grad(loss)(1.0))
        assert jnp.isfinite(g) and abs(g) > 0.0
