"""Acceptance tests for the ordered P0.6 dry-dycore benchmark suite."""

import pytest

pytest.importorskip("dinosaur")

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.framework.gcm.body import EARTH  # noqa: E402
from src.framework.gcm.benchmarks import (  # noqa: E402
    run_resting_atmosphere,
    run_solid_body_tracer,
    run_balanced_jet,
    run_held_suarez,
    resting_convergence_matrix,
)


def test_resting_atmosphere_drift_thresholds():
    """P0.6a: a flat isothermal rest state remains numerically near rest.

    Thresholds are fixed at roughly twice the measured T21/120 s/100-step drift;
    tightening them is allowed, relaxing them requires a documented convergence
    result and review.
    """
    drift = run_resting_atmosphere(MARS_BODY_3D)
    # Dinosaur evolves log(p_s), so spectral truncation does not conserve the
    # integral of exp(log(p_s)) to roundoff. This pins the observed baseline.
    assert drift.relative_atmospheric_mass < 7.0e-6
    assert drift.max_vorticity < 1.2e-4
    assert drift.max_divergence < 2.2e-4
    assert drift.max_temperature_variation < 4.0e-3
    assert drift.max_log_surface_pressure < 7.0e-5


def test_solid_body_tracer_returns_after_one_revolution():
    """P0.6b: spectral scalar advection returns a Gaussian after 2*pi rotation."""
    error = run_solid_body_tracer(MARS_BODY_3D)
    assert error.relative_mass_error < 2.0e-6
    assert error.relative_l1_error < 2.0e-2
    assert error.relative_l2_error < 1.5e-2


def test_jablonowski_williamson_balanced_jet_stays_balanced():
    """P0.6c: the canonical unperturbed jet remains close to its analytic state."""
    drift = run_balanced_jet(EARTH)
    assert drift.relative_atmospheric_mass < 2.0e-6
    assert drift.max_vorticity < 3.0e-3
    assert drift.max_divergence < 4.0e-3
    assert drift.max_temperature_variation < 2.0e-1
    assert drift.max_log_surface_pressure < 4.0e-4


@pytest.mark.slow
def test_held_suarez_climate_has_canonical_circulation_signature():
    """P0.6d: spun-up dry climate has warm equator and eastward jets."""
    climate = run_held_suarez()
    assert 25.0 < climate.equator_minus_pole_surface_temperature_k < 70.0
    assert 15.0 < climate.peak_eastward_wind_m_s < 60.0
    assert 190.0 < climate.minimum_temperature_k < 260.0
    assert climate.relative_atmospheric_mass_drift < 2.0e-5


@pytest.mark.slow
def test_resting_conservation_converges_across_timestep_and_resolution():
    """P0.6e: dry mass, total energy, and axial momentum remain at roundoff."""
    matrix = resting_convergence_matrix(MARS_BODY_3D)
    for drift in matrix.values():
        assert drift.relative_mass < 1.0e-12
        assert drift.relative_total_energy < 1.0e-12
        assert drift.relative_axial_angular_momentum < 1.0e-12
    # Refining T21's timestep reduces all three drift metrics. At T42 all values
    # are already roundoff-sized, where strict monotonicity is not meaningful.
    coarse = matrix[("T21", 240.0)]
    fine = matrix[("T21", 120.0)]
    assert fine.relative_mass <= coarse.relative_mass
    assert fine.relative_total_energy <= coarse.relative_total_energy
    assert fine.relative_axial_angular_momentum <= coarse.relative_axial_angular_momentum
