"""Tests for src.gcm3d.physics — the per-column radiative energy balance added
as an explicit forcing onto dinosaur's 3-D primitive equations (requires the
optional 'gcm3d' extra).

Covers:
  - mars_radiative_forcing: builds a forcing from the package Mars constants.
  - solar_flux: inverse-square orbit (perihelion brighter than aphelion).
  - cos_zenith_nodal: nodal shape, clipped >= 0, diurnal vs daily-mean forms.
  - radiative_heating_tendency: finite, non-trivial, correct modal shape, and
    per-column parity with the 0-D surface energy balance.
  - forced_primitive_equations: integrates stably, advances sim_time, and grows
    a spatial temperature structure the dry core cannot (the whole point of the
    0-D -> 3-D bridge).
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.gcm3d import physics  # noqa: E402
from src.gcm3d._dinosaur import scales  # noqa: E402
from src.gcm3d.coordinates import coordinate_system  # noqa: E402
from src.gcm3d.dynamics import integrate, reference_temperature, stepper  # noqa: E402
from src.gcm3d.dynamics import primitive_equations as build_dry  # noqa: E402
from src.gcm3d.specs import physics_specs  # noqa: E402

jax.config.update("jax_enable_x64", True)
_u = scales.units


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _coords(n_layers: int = 6):
    return coordinate_system(truncation="T21", n_layers=n_layers)


def _rest_state(coords, specs):
    """A rest, isothermal state (zero variation) with sim_time set."""
    from src.gcm3d._dinosaur import jnp, primitive_equations

    grid = coords.horizontal
    zeros = jnp.zeros((coords.vertical.layers,) + grid.modal_shape)
    return primitive_equations.State(
        vorticity=zeros,
        divergence=zeros,
        temperature_variation=zeros,
        log_surface_pressure=jnp.zeros((1,) + grid.modal_shape),
        sim_time=0.0,
    )


# ── mars_radiative_forcing ────────────────────────────────────────────────────

class TestMarsRadiativeForcing:

    def test_pulls_mars_constants(self):
        """Builder mirrors the package's Mars obliquity/orbit/emissivity."""
        from src.celestials.planets import mars as m

        f = physics.mars_radiative_forcing()
        assert f.axial_tilt_rad == pytest.approx(float(m.MARS_AXIAL_TILT))
        assert f.eccentricity == pytest.approx(float(m.MARS_ECCENTRICITY))
        assert f.emissivity == pytest.approx(float(m.MARS_SURFACE_EMISSIVITY))
        assert f.thermal_inertia > 0 and f.greenhouse_factor >= 1.0

    def test_overrides_apply(self):
        """Albedo/greenhouse overrides pass through."""
        f = physics.mars_radiative_forcing(albedo=0.4, greenhouse_factor=1.5)
        assert f.albedo == pytest.approx(0.4)
        assert f.greenhouse_factor == pytest.approx(1.5)


# ── solar_flux ────────────────────────────────────────────────────────────────

class TestSolarFlux:

    def test_perihelion_brighter_than_aphelion(self):
        """Inverse-square law: flux at perihelion (theta=0) exceeds aphelion."""
        f = physics.mars_radiative_forcing()
        peri = float(physics.solar_flux(0.0, f))
        apo = float(physics.solar_flux(f.orbital_period_s / 2.0, f))
        assert peri > apo > 0.0
        # Mars mean flux is ~590 W/m^2; both should be in a sane band.
        assert 400.0 < apo < 800.0 < peri + 200.0


# ── cos_zenith_nodal ──────────────────────────────────────────────────────────

class TestCosZenith:

    def test_shape_and_nonnegative_diurnal(self):
        coords = _coords()
        f = physics.mars_radiative_forcing(diurnal=True)
        g = coords.horizontal
        cz = np.asarray(
            physics.cos_zenith_nodal(0.0, g.latitudes, g.longitudes, f)
        )
        assert cz.shape == (len(g.longitudes), len(g.latitudes))
        assert cz.min() >= 0.0 and cz.max() <= 1.0 + 1e-9
        # Some part of the globe is sunlit and some is dark at a single instant.
        assert cz.max() > 0.0 and cz.min() == 0.0

    def test_daily_mean_has_no_night_side(self):
        """Daily-mean insolation is nonzero across most latitudes (except polar
        night) and, unlike the diurnal snapshot, is longitude-independent."""
        coords = _coords()
        f = physics.mars_radiative_forcing(diurnal=False)
        g = coords.horizontal
        cz = np.asarray(
            physics.cos_zenith_nodal(0.0, g.latitudes, g.longitudes, f)
        )
        assert cz.shape == (len(g.longitudes), len(g.latitudes))
        # longitude-independent: every row identical
        assert np.allclose(cz, cz[0][None, :])
        # equatorial daily-mean insolation is positive
        eq = len(g.latitudes) // 2
        assert cz[0, eq] > 0.0


# ── radiative_heating_tendency ────────────────────────────────────────────────

class TestHeatingTendency:

    def test_shape_finite_and_nontrivial(self):
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        f = physics.mars_radiative_forcing()
        state = _rest_state(coords, specs)
        h = physics.radiative_heating_tendency(coords=coords, state=state, specs=specs, body=MARS_BODY_3D, f=f)
        h = np.asarray(h)
        assert h.shape == state.temperature_variation.shape
        assert np.isfinite(h).all()
        assert np.abs(h).max() > 0.0

    def test_column_parity_with_zero_d_balance(self):
        """The nodal heating equals the 0-D energy balance dT/dt at each column.

        Rebuild Q_in - eps*sigma*(T/gh)^4 over thermal_inertia by hand in SI,
        convert to nondimensional time, and compare against the tendency the
        module produces mapped back to nodal space (rest state => T = T_ref)."""
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        f = physics.mars_radiative_forcing()
        state = _rest_state(coords, specs)
        g = coords.horizontal

        h_nodal = np.asarray(g.to_nodal(
            physics.radiative_heating_tendency(coords=coords, state=state, specs=specs, body=MARS_BODY_3D, f=f)
        ))

        # Hand-computed reference at t=0, rest state (T = reference profile).
        t_s = 0.0
        cz = np.asarray(physics.cos_zenith_nodal(t_s, g.latitudes, g.longitudes, f))
        q_in = (1.0 - f.albedo) * float(physics.solar_flux(t_s, f)) * cz
        ref_t = np.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(-1, 1, 1)
        temp_k = np.clip(ref_t, 1.0, None)
        q_out = f.emissivity * f.stefan_boltzmann * (temp_k / max(f.greenhouse_factor, 1.0)) ** 4
        dtdt_si = (q_in[None] - q_out) / f.thermal_inertia
        time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
        expect = dtdt_si * time_scale_s

        # Put the hand-computed SI field through the *same* spectral round-trip
        # (to_modal then to_nodal) the module uses, so this asserts physics parity
        # independent of dinosaur's truncation (the sharp day/night terminator
        # rings at T21, which is the transform's business, not the balance's).
        expect_rt = np.asarray(g.to_nodal(g.to_modal(expect)))
        assert np.allclose(h_nodal, expect_rt, rtol=1e-6, atol=1e-9)


# ── forced_primitive_equations ────────────────────────────────────────────────

class TestForcedEquations:

    def test_advances_sim_time_and_stays_finite(self):
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        f = physics.mars_radiative_forcing()
        eq = physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs)
        state = _rest_state(coords, specs)
        final = integrate(stepper(eq, 600.0, specs), state, 40)
        assert float(final.sim_time) > 0.0
        for leaf in jax.tree_util.tree_leaves(final):
            assert np.isfinite(np.asarray(leaf)).all()

    def test_grows_spatial_temperature_structure(self):
        """The forcing must create horizontal temperature contrast the dry core
        (which holds the isothermal reference) cannot — the 0-D -> 3-D payoff."""
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        g = coords.horizontal
        ref_t = np.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(-1, 1, 1)
        state = _rest_state(coords, specs)

        # Dry: temperature variation stays ~0 -> surface field is flat.
        dry = integrate(stepper(build_dry(coords, MARS_BODY_3D, specs=specs), 600.0, specs), state, 40)
        dry_surf = (np.asarray(g.to_nodal(dry.temperature_variation)) + ref_t)[-1]
        assert dry_surf.std() < 1.0

        # Forced: surface field develops a hot dayside / cold nightside contrast.
        f = physics.mars_radiative_forcing()
        forced = integrate(
            stepper(physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs), 600.0, specs),
            state, 40,
        )
        forced_surf = (np.asarray(g.to_nodal(forced.temperature_variation)) + ref_t)[-1]
        forced_k = np.asarray(specs.dimensionalize(forced_surf, _u.kelvin).magnitude)
        assert forced_surf.std() > 5.0
        assert np.isfinite(forced_k).all()
        assert forced_k.min() > 50.0 and forced_k.max() < 400.0

    def test_differentiable_through_rollout(self):
        """Gradients flow from a final-state scalar back to the albedo forcing,
        proving the coupling preserves end-to-end differentiability."""
        coords = _coords(n_layers=4)
        specs = physics_specs(MARS_BODY_3D)
        state = _rest_state(coords, specs)
        base = physics.mars_radiative_forcing()

        def mean_temp(albedo):
            f = dataclasses.replace(base, albedo=albedo)
            eq = physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs)
            final = integrate(stepper(eq, 600.0, specs), state, 8)
            return jax.numpy.mean(final.temperature_variation.real)

        grad = float(jax.grad(mean_temp)(0.25))
        assert math.isfinite(grad)
        # More albedo -> less absorbed sunlight -> cooler: gradient is negative.
        assert grad < 0.0


# ── CO2 condensation cycle ────────────────────────────────────────────────────

def _co2_state():
    """Build coords/specs/forcings and an initial (dyn, ice) tuple at rest."""
    from src.gcm3d._dinosaur import jnp, primitive_equations

    coords = _coords(n_layers=6)
    specs = physics_specs(MARS_BODY_3D)
    grid = coords.horizontal
    zeros = jnp.zeros((coords.vertical.layers,) + grid.modal_shape)
    # Realistic Mars surface pressure (~610 Pa), uniform: the CO2 frost point is
    # pressure-dependent, so the initial state must be a real atmosphere, not the
    # nondimensional p_s=1 (which dimensionalises to ~vacuum).
    ps_nd = float(specs.nondimensionalize(610.0 * _u.pascal))
    log_sp_nodal = jnp.full((1,) + grid.nodal_shape, float(np.log(ps_nd)))
    dyn = primitive_equations.State(
        vorticity=zeros,
        divergence=zeros,
        temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(log_sp_nodal),
        sim_time=0.0,
    )
    f = physics.mars_radiative_forcing()
    cf = physics.mars_co2_forcing(escape_rate_kg_s=0.0)
    state = physics.initial_co2_state(dyn, coords, ice_pa=0.0, specs=specs)
    return coords, specs, grid, f, cf, state


def _column_masses(grid, state):
    """Global atmospheric and frost mass via proper spherical quadrature weights.

    Uses the grid's Gaussian ``quadrature_weights`` (which integrate exactly over
    the sphere, sum = 4*pi) rather than a raw cos(lat) approximation, so the
    conservation residual reflects the physics, not the integration rule.
    """
    dyn, ice = state
    w = np.asarray(grid.quadrature_weights)  # (n_lon, n_lat)
    ps = np.asarray(np.exp(grid.to_nodal(dyn.log_surface_pressure)))[0]
    return float((ps * w).sum()), float((np.asarray(ice)[0] * w).sum())


class TestCO2FrostPoint:

    def test_calibration_points(self):
        # Mars surface (6.1 hPa) ~148 K; CO2 sublimation point at 1 atm ~194 K.
        assert float(physics.co2_frost_point_k(610.0)) == pytest.approx(148, abs=1.5)
        assert float(physics.co2_frost_point_k(101325.0)) == pytest.approx(194, abs=1.5)

    def test_monotonic_increasing_with_pressure(self):
        ps = np.linspace(50.0, 1.0e5, 60)
        t = np.array([float(physics.co2_frost_point_k(p)) for p in ps])
        assert np.all(np.diff(t) > 0)

    def test_finite_in_vacuum(self):
        assert np.isfinite(float(physics.co2_frost_point_k(0.0)))


class TestCO2Cycle:

    def test_conserves_total_mass_with_escape_off(self):
        """With escape=0 the CO2 cycle only *moves* mass atmosphere<->frost, so
        area-weighted (atmosphere + frost) is conserved (to spectral tolerance)."""
        coords, specs, grid, f, cf, state = _co2_state()
        eq = physics.forced_co2_primitive_equations(
            coords, MARS_BODY_3D, f, cf, specs=specs
        )
        step = jax.jit(stepper(eq, 600.0, specs))
        a0, i0 = _column_masses(grid, state)
        for _ in range(200):
            state = step(state)
        a1, i1 = _column_masses(grid, state)
        assert i0 == pytest.approx(0.0)
        assert i1 > 0.0  # frost formed at the winter pole
        assert a1 < a0   # atmosphere lost the condensed mass
        # total conserved to better than 0.1% (residual is spectral truncation)
        assert abs((a1 + i1) - (a0 + i0)) / (a0 + i0) < 1e-3

    def test_frost_forms_at_the_winter_pole(self):
        """CO2 deposits where it is coldest — the winter (dark) pole, at |lat|>60."""
        coords, specs, grid, f, cf, state = _co2_state()
        eq = physics.forced_co2_primitive_equations(
            coords, MARS_BODY_3D, f, cf, specs=specs
        )
        final = integrate(stepper(eq, 600.0, specs), state, 200)
        _, ice = final
        ice = np.asarray(ice)[0]
        j = np.unravel_index(np.argmax(ice), ice.shape)
        assert abs(np.degrees(np.asarray(grid.latitudes)[j[1]])) > 60.0
        assert ice.min() >= -1e-6  # frost stays non-negative

    def test_frost_point_buffers_the_cold_end(self):
        """The latent buffering keeps the coldest surface temperature far warmer
        than the radiation-only run (which cools the winter pole below ~80 K)."""
        coords, specs, grid, f, cf, state = _co2_state()
        eq = physics.forced_co2_primitive_equations(
            coords, MARS_BODY_3D, f, cf, specs=specs
        )
        dyn0, _ = state
        final, _ = integrate(stepper(eq, 600.0, specs), state, 300)
        ref = np.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(-1, 1, 1)
        t_surf = (np.asarray(grid.to_nodal(final.temperature_variation)) + ref)[-1]

        # radiation only, same run
        eqr = physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs)
        finr = integrate(stepper(eqr, 600.0, specs), dyn0, 300)
        t_surf_r = (np.asarray(grid.to_nodal(finr.temperature_variation)) + ref)[-1]

        assert t_surf.min() > t_surf_r.min()  # CO2 raises the cold floor


class TestDiurnalStabilityGuard:

    def test_run_maps_rejects_too_coarse_diurnal_step(self):
        """run_maps must refuse a step too coarse for the moving terminator rather
        than emit NaN maps (regression for the T42/dt=600 diurnal blow-up)."""
        from src.gcm3d import maps, topography as topo

        if not topo._DEFAULT_MOLA.exists():
            pytest.skip("MOLA raster not staged")
        f = physics.mars_radiative_forcing(diurnal=True)
        with pytest.raises(ValueError, match="too coarse"):
            maps.run_maps(
                truncation="T42", n_layers=6, dt_seconds=600.0, n_steps=2, forcing=f
            )

    def test_daily_mean_forcing_has_no_terminator_limit(self):
        """diurnal=False (smooth insolation) is not subject to the guard."""
        from src.gcm3d import maps, topography as topo

        if not topo._DEFAULT_MOLA.exists():
            pytest.skip("MOLA raster not staged")
        f = physics.mars_radiative_forcing(diurnal=False)
        fields = maps.run_maps(
            truncation="T42", n_layers=6, dt_seconds=600.0, n_steps=5, forcing=f
        )
        assert np.isfinite(fields.temperature_k).all()
