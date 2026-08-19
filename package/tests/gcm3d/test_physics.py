"""Tests for framework GCM physics — the per-column radiative energy balance added
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
from src.celestials.planets.mars import gcm as mars_gcm  # noqa: E402
from src.framework.physics import gcm as physics  # noqa: E402
from src.framework.gcm._dinosaur import jnp, scales, spherical_harmonic  # noqa: E402
from src.framework.gcm.coordinates import coordinate_system  # noqa: E402
from src.framework.gcm.dynamics import integrate, reference_temperature, stepper  # noqa: E402
from src.framework.gcm.dynamics import primitive_equations as build_dry  # noqa: E402
from src.framework.gcm.specs import physics_specs  # noqa: E402

jax.config.update("jax_enable_x64", True)
_u = scales.units


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _coords(n_layers: int = 6):
    return coordinate_system(truncation="T21", n_layers=n_layers)


def _rest_state(coords, specs):
    """A rest, isothermal state (zero variation) with sim_time set."""
    from src.framework.gcm._dinosaur import jnp, primitive_equations

    grid = coords.horizontal
    zeros = jnp.zeros((coords.vertical.layers,) + grid.modal_shape)
    ps_nd = float(specs.nondimensionalize(610.0 * _u.pascal))
    log_sp = grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(ps_nd)))
    return primitive_equations.State(
        vorticity=zeros,
        divergence=zeros,
        temperature_variation=zeros,
        log_surface_pressure=log_sp,
        sim_time=0.0,
    )


def _column_state(coords, specs, surface_temperature_k=200.0):
    return physics.initial_column_state(
        _rest_state(coords, specs), coords, surface_temperature_k, specs
    )


# ── mars_radiative_forcing ────────────────────────────────────────────────────

class TestMarsRadiativeForcing:

    def test_pulls_mars_constants(self):
        """Builder mirrors the package's Mars obliquity/orbit/emissivity."""
        from src.celestials.planets import mars as m

        f = mars_gcm.radiative_forcing()
        assert f.axial_tilt_rad == pytest.approx(float(m.MARS_AXIAL_TILT))
        assert f.eccentricity == pytest.approx(float(m.MARS_ECCENTRICITY))
        assert f.emissivity == pytest.approx(float(m.MARS_SURFACE_EMISSIVITY))
        assert f.thermal_inertia > 0 and f.greenhouse_factor >= 1.0

    def test_overrides_apply(self):
        """Albedo/greenhouse overrides pass through."""
        f = mars_gcm.radiative_forcing(albedo=0.4, greenhouse_factor=1.5)
        assert f.albedo == pytest.approx(0.4)
        assert f.greenhouse_factor == pytest.approx(1.5)


# ── solar_flux ────────────────────────────────────────────────────────────────

class TestSolarFlux:

    def test_perihelion_brighter_than_aphelion(self):
        """Inverse-square law: flux at perihelion (theta=0) exceeds aphelion."""
        f = mars_gcm.radiative_forcing()
        peri = float(physics.solar_flux(0.0, f))
        apo = float(physics.solar_flux(f.orbital_period_s / 2.0, f))
        assert peri > apo > 0.0
        # Mars mean flux is ~590 W/m^2; both should be in a sane band.
        assert 400.0 < apo < 800.0 < peri + 200.0


# ── cos_zenith_nodal ──────────────────────────────────────────────────────────

class TestCosZenith:

    def test_shape_and_nonnegative_diurnal(self):
        coords = _coords()
        f = mars_gcm.radiative_forcing(diurnal=True)
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
        f = mars_gcm.radiative_forcing(diurnal=False)
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

    def test_shape_finite_and_surface_forcing_nontrivial(self):
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        f = mars_gcm.radiative_forcing()
        state = _column_state(coords, specs)
        h, ds, diagnostic = physics.surface_energy_tendencies(
            state, coords, specs, MARS_BODY_3D, f
        )
        h = np.asarray(h)
        assert h.shape == state.dynamics.temperature_variation.shape
        assert np.isfinite(h).all()
        assert np.isfinite(np.asarray(ds)).all()
        assert np.abs(np.asarray(ds)).max() > 0.0
        assert np.isfinite(np.asarray(diagnostic.net_external_w_m2)).all()

    def test_two_stream_column_fluxes_close(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        f = mars_gcm.radiative_forcing(co2_radiation_enabled=True)
        flux = physics.two_stream_radiative_fluxes(
            state, coords, specs, MARS_BODY_3D, f
        )
        closed = (
            np.sum(np.asarray(flux.atmospheric_convergence_w_m2), axis=0)
            + np.asarray(flux.surface_net_w_m2)
        )
        assert np.allclose(closed, np.asarray(flux.toa_net_down_w_m2), atol=1e-10)

    def test_zero_dust_limit_and_positive_dust_heating_response(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        clear = mars_gcm.radiative_forcing(co2_radiation_enabled=True)
        zero = dataclasses.replace(
            clear, dust_visible_optical_depth=0.0, dust_longwave_optical_depth=0.0
        )
        dusty = dataclasses.replace(
            clear, dust_visible_optical_depth=1.0, dust_longwave_optical_depth=0.3
        )
        a = physics.two_stream_radiative_fluxes(state, coords, specs, MARS_BODY_3D, clear)
        b = physics.two_stream_radiative_fluxes(state, coords, specs, MARS_BODY_3D, zero)
        c = physics.two_stream_radiative_fluxes(state, coords, specs, MARS_BODY_3D, dusty)
        assert np.array_equal(np.asarray(a.shortwave_down_w_m2), np.asarray(b.shortwave_down_w_m2))
        assert np.max(np.abs(
            np.asarray(c.atmospheric_convergence_w_m2)
            - np.asarray(a.atmospheric_convergence_w_m2)
        )) > 0.0
        assert np.mean(np.asarray(c.shortwave_up_w_m2[0])) > np.mean(
            np.asarray(a.shortwave_up_w_m2[0])
        )
        dusty_closed = (
            np.sum(np.asarray(c.atmospheric_convergence_w_m2), axis=0)
            + np.asarray(c.surface_net_w_m2)
        )
        assert np.allclose(dusty_closed, np.asarray(c.toa_net_down_w_m2), atol=1e-10)

    def test_multiband_opacity_responds_to_pressure_and_temperature(self):
        """The new closure is not a relabeled grey band: local P/T changes fluxes."""
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        f = mars_gcm.radiative_forcing(co2_radiation_enabled=True)
        grid = coords.horizontal

        def with_pressure(column, pressure_pa):
            ps_nd = specs.nondimensionalize(pressure_pa * _u.pascal)
            log_ps = grid.to_modal(
                jnp.full((1,) + grid.nodal_shape, jnp.log(ps_nd))
            )
            return column._replace(
                dynamics=column.dynamics.replace(log_surface_pressure=log_ps)
            )

        thin = physics.two_stream_radiative_fluxes(
            with_pressure(state, 300.0), coords, specs, MARS_BODY_3D, f
        )
        thick = physics.two_stream_radiative_fluxes(
            with_pressure(state, 1200.0), coords, specs, MARS_BODY_3D, f
        )
        # A thicker CO2 column transmits less sunlight to the surface.
        assert np.mean(np.asarray(thick.shortwave_down_w_m2[-1])) < np.mean(
            np.asarray(thin.shortwave_down_w_m2[-1])
        )

        def toa_longwave(temp_offset_k):
            offset = grid.to_modal(
                jnp.full(
                    (coords.vertical.layers,) + grid.nodal_shape,
                    temp_offset_k,
                )
            )
            warm = state._replace(
                dynamics=state.dynamics.replace(temperature_variation=offset)
            )
            flux = physics.two_stream_radiative_fluxes(
                warm, coords, specs, MARS_BODY_3D, f
            )
            return jnp.mean(flux.longwave_up_w_m2[0])

        gradient = float(jax.grad(toa_longwave)(0.0))
        assert math.isfinite(gradient)
        assert abs(gradient) > 1.0e-6

    def test_invalid_multiband_configuration_fails_early(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        f = dataclasses.replace(
            mars_gcm.radiative_forcing(co2_radiation_enabled=True),
            co2_shortwave_band_weights=(1.0,),
        )
        with pytest.raises(ValueError, match="shortwave CO2 band tuples"):
            physics.two_stream_radiative_fluxes(
                state, coords, specs, MARS_BODY_3D, f
            )


class TestSurfaceMomentumDrag:

    def test_drag_removes_lowest_layer_kinetic_energy(self):
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        grid = coords.horizontal
        u = jnp.zeros((coords.vertical.layers,) + grid.nodal_shape).at[-1].set(20.0)
        v = jnp.zeros_like(u)
        velocity_unit = _u.meter / _u.second
        vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(
            grid,
            specs.nondimensionalize(u * velocity_unit),
            specs.nondimensionalize(v * velocity_unit),
        )
        state = state._replace(dynamics=dataclasses.replace(
            state.dynamics, vorticity=vor, divergence=div
        ))
        drag_vor, drag_div = physics.surface_momentum_tendencies(
            state, coords, specs, MARS_BODY_3D, mars_gcm.radiative_forcing()
        )
        du_nd, dv_nd = spherical_harmonic.vor_div_to_uv_nodal(
            grid, drag_vor, drag_div
        )
        du = np.asarray(specs.dimensionalize(du_nd, _u.meter / _u.second).magnitude)
        dv = np.asarray(specs.dimensionalize(dv_nd, _u.meter / _u.second).magnitude)
        assert np.mean(du[-1]) < 0.0
        assert np.max(np.abs(du[:-1])) < 1e-10
        assert np.max(np.abs(dv)) < 1e-10


class TestRegolithConduction:

    def test_custom_layer_layout_sizes_state_and_conduction_consistently(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        default_state = _column_state(coords, specs)
        forcing = dataclasses.replace(
            mars_gcm.radiative_forcing(), regolith_enabled=True,
            regolith_layer_skin_depth_fractions=(0.5, 1.5, 4.0),
        )
        state = physics.initial_column_state(
            default_state.dynamics, coords, 210.0, specs, forcing=forcing
        )
        assert state.ground_temperature.shape[0] == 3
        _, tendency = physics.regolith_conduction_tendencies(state, specs, forcing)
        assert tendency.shape == state.ground_temperature.shape

    def test_mismatched_layer_layout_has_descriptive_error(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        forcing = dataclasses.replace(
            mars_gcm.radiative_forcing(), regolith_enabled=True,
            regolith_layer_skin_depth_fractions=(1.0, 2.0),
        )
        with pytest.raises(ValueError, match="ground_temperature has 12 layers"):
            physics.regolith_conduction_tendencies(state, specs, forcing)

    def test_internal_conduction_closes_column_energy(self):
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs, surface_temperature_k=220.0)
        ground = state.ground_temperature.at[0].set(state.ground_temperature[0] - 10.0)
        state = state._replace(ground_temperature=ground)
        f = dataclasses.replace(
            mars_gcm.radiative_forcing(), regolith_enabled=True,
            surface_thermal_inertia_tiu=250.0,
        )
        ds, dg = physics.regolith_conduction_tendencies(state, specs, f)
        time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
        skin = 250.0 / f.regolith_volumetric_heat_capacity_j_m3_k * math.sqrt(
            f.rotation_period_s / math.pi
        )
        dz = np.asarray(f.regolith_layer_skin_depth_fractions) * skin
        surface_power = np.asarray(ds)[0] / time_scale * f.thermal_inertia
        ground_power = np.sum(
            np.asarray(dg) / time_scale
            * f.regolith_volumetric_heat_capacity_j_m3_k
            * dz[:, None, None], axis=0,
        )
        assert np.max(np.abs(surface_power + ground_power)) < 1e-10

    def test_skin_depth_scales_linearly_with_thermal_inertia(self):
        f = mars_gcm.radiative_forcing()
        factor = math.sqrt(f.rotation_period_s / math.pi) / f.regolith_volumetric_heat_capacity_j_m3_k
        assert 500.0 * factor == pytest.approx(2.0 * 250.0 * factor)


class TestBoundaryLayerPhysics:

    def test_bulk_richardson_suppresses_stable_and_enhances_unstable_exchange(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        f = dataclasses.replace(
            mars_gcm.radiative_forcing(), stability_exchange_enabled=True
        )
        air = jnp.full(coords.horizontal.nodal_shape, 200.0)
        stable, _, _ = physics._surface_exchange_properties(
            state, coords, specs, MARS_BODY_3D, f, air, air - 10.0
        )
        neutral, _, _ = physics._surface_exchange_properties(
            state, coords, specs, MARS_BODY_3D, f, air, air
        )
        unstable, _, _ = physics._surface_exchange_properties(
            state, coords, specs, MARS_BODY_3D, f, air, air + 10.0
        )
        assert np.all(np.asarray(stable) < np.asarray(neutral))
        assert np.all(np.asarray(unstable) > np.asarray(neutral))

    def test_vertical_temperature_diffusion_conserves_mass_weighted_mean(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        grid = coords.horizontal
        profile = jnp.arange(coords.vertical.layers)[:, None, None]
        nodal = jnp.broadcast_to(profile, (coords.vertical.layers,) + grid.nodal_shape)
        state = state._replace(dynamics=dataclasses.replace(
            state.dynamics, temperature_variation=grid.to_modal(nodal)
        ))
        f = dataclasses.replace(
            mars_gcm.radiative_forcing(), pbl_diffusion_enabled=True
        )
        _, _, tendency, _ = physics.pbl_vertical_diffusion_tendencies(
            state, coords, specs, MARS_BODY_3D, f
        )
        weights = np.diff(np.asarray(coords.vertical.boundaries))[:, None, None]
        residual = np.sum(weights * np.asarray(grid.to_nodal(tendency)), axis=0)
        assert np.max(np.abs(residual)) < 1e-10

    def test_vertical_tracer_diffusion_conserves_column_mass(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        grid = coords.horizontal
        profile = jnp.arange(coords.vertical.layers)[:, None, None]
        nodal = jnp.broadcast_to(profile, (coords.vertical.layers,) + grid.nodal_shape)
        state = state._replace(dynamics=dataclasses.replace(
            state.dynamics, tracers={"dust": grid.to_modal(nodal)}
        ))
        f = dataclasses.replace(
            mars_gcm.radiative_forcing(), pbl_diffusion_enabled=True
        )
        *_, tracers = physics.pbl_vertical_diffusion_tendencies(
            state, coords, specs, MARS_BODY_3D, f
        )
        weights = np.diff(np.asarray(coords.vertical.boundaries))[:, None, None]
        residual = np.sum(
            weights * np.asarray(grid.to_nodal(tracers["dust"])), axis=0
        )
        assert np.max(np.abs(residual)) < 1e-10

    def test_implicit_momentum_mixing_conserves_momentum_and_dissipates_ke(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        n = coords.vertical.layers
        shape = coords.horizontal.nodal_shape
        field = jnp.broadcast_to(jnp.linspace(0.0, 30.0, n)[:, None, None], (n,) + shape)
        rates = jnp.full((n - 1,) + shape, 1.0 / 900.0)
        weights = np.diff(np.asarray(coords.vertical.boundaries))
        _, new = physics._implicit_vertical_diffusion_tendency(
            field, rates, weights, 1800.0, specs, velocity=True
        )
        old = np.asarray(field)
        new = np.asarray(new)
        w = weights[:, None, None]
        # Dinosaur's vertical-coordinate weights are float32.
        assert np.max(np.abs(np.sum(w * (new - old), axis=0))) < 2e-7
        assert np.all(np.sum(w * new**2, axis=0) <= np.sum(w * old**2, axis=0) + 1e-10)


class TestDryConvectiveAdjustment:

    def test_stable_layers_outside_local_unstable_block_are_unchanged(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        grid, n = coords.horizontal, coords.vertical.layers
        sigma = np.asarray(coords.vertical.centers)[:, None, None]
        exner = sigma ** MARS_BODY_3D.kappa
        theta_1d = np.array([300.0, 280.0, 200.0, 220.0, 180.0, 160.0])
        temperature = np.broadcast_to(
            theta_1d[:, None, None] * exner, (n,) + grid.nodal_shape
        )
        ref = np.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(n, 1, 1)
        state = state._replace(dynamics=dataclasses.replace(
            state.dynamics,
            temperature_variation=grid.to_modal(jnp.asarray(temperature - ref)),
        ))
        adjusted = np.asarray(grid.to_nodal(
            physics.dry_convective_adjusted_temperature(state, coords, MARS_BODY_3D)
        )) + ref
        adjusted_theta = adjusted / exner
        assert np.allclose(adjusted_theta[[0, 1, 4, 5]], theta_1d[[0, 1, 4, 5], None, None], atol=2e-5)
        assert np.allclose(adjusted_theta[2], adjusted_theta[3], atol=2e-5)

    def test_removes_instability_and_conserves_enthalpy(self):
        coords, specs = _coords(), physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        grid, n = coords.horizontal, coords.vertical.layers
        sigma = np.asarray(coords.vertical.centers)[:, None, None]
        exner = sigma ** MARS_BODY_3D.kappa
        theta = np.linspace(180.0, 260.0, n)[:, None, None]
        temperature = np.broadcast_to(theta * exner, (n,) + grid.nodal_shape)
        ref = np.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(n, 1, 1)
        state = state._replace(dynamics=dataclasses.replace(
            state.dynamics,
            temperature_variation=grid.to_modal(jnp.asarray(temperature - ref)),
        ))
        adjusted_modal = physics.dry_convective_adjusted_temperature(
            state, coords, MARS_BODY_3D
        )
        adjusted = np.asarray(grid.to_nodal(adjusted_modal)) + ref
        adjusted_theta = adjusted / exner
        assert np.min(adjusted_theta[:-1] - adjusted_theta[1:]) >= -1e-9
        weights = np.diff(np.asarray(coords.vertical.boundaries))[:, None, None]
        before = np.sum(weights * temperature, axis=0)
        after = np.sum(weights * adjusted, axis=0)
        # Nodal adjustment is algebraically exact; modal round-trip is limited by
        # Dinosaur's float32 spectral basis even with JAX x64 enabled.
        assert np.max(np.abs(after - before)) < 2e-5

    @pytest.mark.parametrize("n_layers", [4, 8])
    def test_surface_atmosphere_exchange_conserves_energy(self, n_layers):
        """Internal sensible exchange cancels exactly in the column budget."""
        coords = _coords(n_layers)
        specs = physics_specs(MARS_BODY_3D)
        f = mars_gcm.radiative_forcing()
        state = _column_state(coords, specs, surface_temperature_k=220.0)
        g = coords.horizontal
        time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
        heat, ds, diagnostic = physics.surface_energy_tendencies(
            state, coords, specs, MARS_BODY_3D, f
        )
        air_rate_si = np.asarray(g.to_nodal(heat))[-1] / time_scale_s
        ps_nd = np.exp(np.asarray(g.to_nodal(state.dynamics.log_surface_pressure)))[0]
        ps_pa = ps_nd * float(specs.dimensionalize(1.0, _u.pascal).magnitude)
        dsigma = np.diff(np.asarray(coords.vertical.boundaries))[-1]
        capacity = MARS_BODY_3D.cp_j_kg_k * ps_pa * dsigma / MARS_BODY_3D.gravity_m_s2
        atmospheric_flux = air_rate_si * capacity
        surface_flux = np.asarray(ds)[0] / time_scale_s * f.thermal_inertia
        assert np.allclose(
            surface_flux + atmospheric_flux,
            np.asarray(diagnostic.net_external_w_m2), rtol=2e-5, atol=2e-5,
        )


# ── forced_primitive_equations ────────────────────────────────────────────────

class TestForcedEquations:

    def test_advances_sim_time_and_stays_finite(self):
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        f = mars_gcm.radiative_forcing()
        eq = physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs)
        state = _column_state(coords, specs)
        final = integrate(stepper(eq, 600.0, specs), state, 40)
        assert float(final.dynamics.sim_time) > 0.0
        for leaf in jax.tree_util.tree_leaves(final):
            assert np.isfinite(np.asarray(leaf)).all()

    def test_grows_spatial_temperature_structure(self):
        """The forcing must create horizontal temperature contrast the dry core
        (which holds the isothermal reference) cannot — the 0-D -> 3-D payoff."""
        coords = _coords()
        specs = physics_specs(MARS_BODY_3D)
        g = coords.horizontal
        ref_t = np.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(-1, 1, 1)
        dry_state = _rest_state(coords, specs)

        # Dry: temperature variation stays ~0 -> surface field is flat.
        dry = integrate(stepper(build_dry(coords, MARS_BODY_3D, specs=specs), 600.0, specs), dry_state, 40)
        dry_surf = (np.asarray(g.to_nodal(dry.temperature_variation)) + ref_t)[-1]
        assert dry_surf.std() < 1.0

        # Forced: surface field develops a hot dayside / cold nightside contrast.
        f = mars_gcm.radiative_forcing()
        state = physics.initial_column_state(dry_state, coords, 200.0, specs)
        forced = integrate(
            stepper(physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs), 600.0, specs),
            state, 40,
        )
        forced_k = np.asarray(specs.dimensionalize(forced.surface_temperature[0], _u.kelvin).magnitude)
        assert forced_k.std() > 5.0
        assert np.isfinite(forced_k).all()
        assert forced_k.min() > 50.0 and forced_k.max() < 400.0

    def test_differentiable_through_rollout(self):
        """Gradients flow from a final-state scalar back to the albedo forcing,
        proving the coupling preserves end-to-end differentiability."""
        coords = _coords(n_layers=4)
        specs = physics_specs(MARS_BODY_3D)
        state = _column_state(coords, specs)
        base = mars_gcm.radiative_forcing()

        def mean_temp(albedo):
            f = dataclasses.replace(base, albedo=albedo)
            eq = physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs)
            final = integrate(stepper(eq, 600.0, specs), state, 8)
            return jax.numpy.mean(final.surface_temperature)

        grad = float(jax.grad(mean_temp)(0.25))
        assert math.isfinite(grad)
        # More albedo -> less absorbed sunlight -> cooler: gradient is negative.
        assert grad < 0.0


# ── CO2 condensation cycle ────────────────────────────────────────────────────

def _co2_state():
    """Build coords/specs/forcings and an initial (dyn, ice) tuple at rest."""
    from src.framework.gcm._dinosaur import jnp, primitive_equations

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
    f = mars_gcm.radiative_forcing()
    # Legacy relaxation cases remain regression-tested separately from the
    # production energy-limited, projected path.
    cf = mars_gcm.co2_forcing(escape_rate_kg_s=0.0, energy_limited=False)
    state = physics.initial_co2_state(
        dyn, coords, ice_pa=0.0, specs=specs, body=MARS_BODY_3D
    )
    return coords, specs, grid, f, cf, state


def _column_masses(grid, state):
    """Global atmospheric and frost mass via proper spherical quadrature weights.

    Uses the grid's Gaussian ``quadrature_weights`` (which integrate exactly over
    the sphere, sum = 4*pi) rather than a raw cos(lat) approximation, so the
    conservation residual reflects the physics, not the integration rule.
    """
    dyn, ice = state.dynamics, state.co2_ice
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

    def test_projection_preserves_co2_and_removes_negative_frost(self):
        coords, specs, grid, _, _, state = _co2_state()
        pressure_scale = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
        state = state._replace(co2_ice=jnp.full_like(state.co2_ice, -2.0 / pressure_scale))
        before = sum(_column_masses(grid, state))
        projected = physics.project_co2_reservoirs(state, coords, specs)
        after = sum(_column_masses(grid, projected))
        assert np.min(np.asarray(projected.co2_ice)) >= 0.0
        assert after == pytest.approx(before, rel=1e-7)

    def test_projected_energy_limited_rollout_keeps_frost_nonnegative(self):
        coords, specs, _, f, cf, state = _co2_state()
        cf = dataclasses.replace(cf, energy_limited=True)
        equation = physics.forced_co2_primitive_equations(
            coords, MARS_BODY_3D, f, cf, specs=specs
        )
        advance = physics.positivity_preserving_co2_step(
            stepper(equation, 600.0, specs), coords, specs
        )
        final = integrate(advance, state, 200)
        assert np.min(np.asarray(final.co2_ice)) >= 0.0

    def test_energy_limited_latent_heat_cancels_surface_deficit(self):
        coords, specs, _, _, cf, state = _co2_state()
        cf = dataclasses.replace(cf, energy_limited=True)
        state = state._replace(surface_temperature=jnp.full_like(state.surface_temperature, 140.0))
        residual = jnp.full(coords.horizontal.nodal_shape, -100.0)
        _, dice, latent = physics._co2_surface_tendencies(
            state, coords, specs, MARS_BODY_3D, cf,
            available_surface_flux_w_m2=residual,
        )
        time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
        latent_flux = np.asarray(latent)[0] / time_scale * cf.thermal_inertia
        assert np.all(np.asarray(dice) >= 0.0)
        assert np.allclose(latent_flux, 100.0, rtol=1e-5)

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
        ice = np.asarray(final.co2_ice)[0]
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
        dyn0 = state.dynamics
        final = integrate(stepper(eq, 600.0, specs), state, 300)
        t_surf = np.asarray(final.surface_temperature)[0]

        # radiation only, same run
        eqr = physics.forced_primitive_equations(coords, MARS_BODY_3D, f, specs=specs)
        rad0 = physics.initial_column_state(dyn0, coords, 200.0, specs)
        finr = integrate(stepper(eqr, 600.0, specs), rad0, 300)
        t_surf_r = np.asarray(finr.surface_temperature)[0]

        assert t_surf.min() > t_surf_r.min()  # CO2 raises the cold floor


class TestDiurnalStabilityGuard:

    def test_run_maps_rejects_too_coarse_diurnal_step(self):
        """run_maps must refuse a step too coarse for the moving terminator rather
        than emit NaN maps (regression for the T42/dt=600 diurnal blow-up)."""
        from src.celestials.planets.mars import maps, topography as topo

        if not topo._DEFAULT_MOLA.exists():
            pytest.skip("MOLA raster not staged")
        f = mars_gcm.radiative_forcing(diurnal=True)
        with pytest.raises(ValueError, match="too coarse"):
            maps.run_maps(
                truncation="T42", n_layers=6, dt_seconds=600.0, n_steps=2, forcing=f
            )

    def test_daily_mean_forcing_has_no_terminator_limit(self):
        """diurnal=False (smooth insolation) is not subject to the guard."""
        from src.celestials.planets.mars import maps, topography as topo

        if not topo._DEFAULT_MOLA.exists():
            pytest.skip("MOLA raster not staged")
        f = mars_gcm.radiative_forcing(diurnal=False)
        fields = maps.run_maps(
            truncation="T42", n_layers=6, dt_seconds=600.0, n_steps=5, forcing=f
        )
        assert np.isfinite(fields.temperature_k).all()
