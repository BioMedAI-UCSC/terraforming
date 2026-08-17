"""Canonical, planet-aware dry-dycore benchmarks and conservation diagnostics.

These helpers keep benchmark construction separate from Mars parameterizations.
They use Dinosaur's published primitive-equation initial states and the Gaussian
quadrature weights of the spectral grid, so acceptance tests measure the dycore
rather than plotting or interpolation artifacts.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.gcm3d._dinosaur import (
    jax, jnp, primitive_equations as dinosaur_pe, primitive_equations_states,
    scales, time_integration,
)
from dinosaur import xarray_utils
from dinosaur import held_suarez, spherical_harmonic
from dinosaur.primitive_equations import div_sec_lat
from src.gcm3d.dynamics import integrate, primitive_equations, stepper
from src.gcm3d.specs import physics_specs

_u = scales.units


@dataclasses.dataclass(frozen=True)
class DryDriftDiagnostics:
    """Dimensionless drift metrics between two primitive-equation states."""

    relative_atmospheric_mass: float
    max_vorticity: float
    max_divergence: float
    max_temperature_variation: float
    max_log_surface_pressure: float


@dataclasses.dataclass(frozen=True)
class TracerAdvectionDiagnostics:
    relative_mass_error: float
    relative_l1_error: float
    relative_l2_error: float


@dataclasses.dataclass(frozen=True)
class HeldSuarezDiagnostics:
    equator_minus_pole_surface_temperature_k: float
    peak_eastward_wind_m_s: float
    minimum_temperature_k: float
    relative_atmospheric_mass_drift: float


@dataclasses.dataclass(frozen=True)
class ConservedQuantities:
    """Global dry-atmosphere invariants in SI up to a common area factor."""

    mass_kg_m2_sr: float
    total_energy_j_m2_sr: float
    axial_angular_momentum_kg_m_s_sr: float


@dataclasses.dataclass(frozen=True)
class ConservationDrift:
    relative_mass: float
    relative_total_energy: float
    relative_axial_angular_momentum: float


def dry_conserved_quantities(state, coords, specs, body, reference_temperature_k):
    """Pressure-mass-weighted dry mass, total energy, and axial angular momentum.

    Hydrostatic geopotential is reconstructed from
    ``dPhi = -R*T*d(ln p)`` between sigma-level centers. Total specific energy is
    ``cv*T + Phi + |u|²/2``; axial angular momentum is
    ``a*cos(phi)*(u + Omega*a*cos(phi))``. The omitted common ``a²`` area factor
    cancels from all relative-drift diagnostics.
    """
    grid = coords.horizontal
    aux = dinosaur_pe.compute_diagnostic_state_sigma(state, coords)
    temp = (jnp.asarray(reference_temperature_k)[:, None, None]
            + aux.temperature_variation)
    cos_lat = jnp.asarray(grid.cos_lat)
    u_cos, v_cos = aux.cos_lat_u
    safe_cos = jnp.clip(cos_lat, 1e-8, None)
    u_nd, v_nd = u_cos / safe_cos, v_cos / safe_cos
    velocity_scale = float(
        specs.dimensionalize(1.0, _u.meter / _u.second).magnitude
    )
    u, v = u_nd * velocity_scale, v_nd * velocity_scale

    sigma = jnp.asarray(coords.vertical.centers)
    phi = jnp.zeros_like(temp)
    phi = phi.at[-1].set(
        body.gas_constant_j_kg_k * temp[-1] * jnp.log(1.0 / sigma[-1])
    )
    for k in range(coords.vertical.layers - 2, -1, -1):
        increment = (
            body.gas_constant_j_kg_k * 0.5 * (temp[k] + temp[k + 1])
            * jnp.log(sigma[k + 1] / sigma[k])
        )
        phi = phi.at[k].set(phi[k + 1] + increment)

    ps_nd = jnp.exp(grid.to_nodal(state.log_surface_pressure))[0]
    pressure_scale = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    ps = ps_nd * pressure_scale
    dsigma = jnp.asarray(np.diff(np.asarray(coords.vertical.boundaries)))
    layer_mass = ps[None, :, :] * dsigma[:, None, None] / body.gravity_m_s2
    weights = jnp.asarray(grid.quadrature_weights)[None, :, :]
    cv = body.cp_j_kg_k - body.gas_constant_j_kg_k
    specific_energy = cv * temp + phi + 0.5 * (u**2 + v**2)
    axial = body.radius_m * cos_lat * (
        u + body.angular_velocity_s * body.radius_m * cos_lat
    )
    return ConservedQuantities(
        mass_kg_m2_sr=float(jnp.sum(layer_mass * weights)),
        total_energy_j_m2_sr=float(jnp.sum(layer_mass * specific_energy * weights)),
        axial_angular_momentum_kg_m_s_sr=float(jnp.sum(layer_mass * axial * weights)),
    )


def conservation_drift(initial, final, coords, specs, body, reference_temperature_k):
    q0 = dry_conserved_quantities(initial, coords, specs, body, reference_temperature_k)
    q1 = dry_conserved_quantities(final, coords, specs, body, reference_temperature_k)

    def relative(a, b):
        return abs(b - a) / abs(a)

    return ConservationDrift(
        relative(q0.mass_kg_m2_sr, q1.mass_kg_m2_sr),
        relative(q0.total_energy_j_m2_sr, q1.total_energy_j_m2_sr),
        relative(q0.axial_angular_momentum_kg_m_s_sr,
                 q1.axial_angular_momentum_kg_m_s_sr),
    )


def _weighted_surface_pressure(state, coords):
    pressure = jnp.exp(coords.horizontal.to_nodal(state.log_surface_pressure))[0]
    return jnp.sum(pressure * jnp.asarray(coords.horizontal.quadrature_weights))


def dry_drift_diagnostics(initial, final, coords) -> DryDriftDiagnostics:
    """Compute reproducible mass and max-state drift diagnostics."""
    m0 = _weighted_surface_pressure(initial, coords)
    m1 = _weighted_surface_pressure(final, coords)

    def max_delta(name):
        return float(jnp.max(jnp.abs(getattr(final, name) - getattr(initial, name))))

    return DryDriftDiagnostics(
        relative_atmospheric_mass=float(jnp.abs(m1 - m0) / jnp.abs(m0)),
        max_vorticity=max_delta("vorticity"),
        max_divergence=max_delta("divergence"),
        max_temperature_variation=max_delta("temperature_variation"),
        max_log_surface_pressure=max_delta("log_surface_pressure"),
    )


def run_resting_atmosphere(
    body,
    truncation: str = "T21",
    n_layers: int = 12,
    dt_seconds: float = 120.0,
    n_steps: int = 100,
):
    """Run Dinosaur's isothermal, flat, resting-atmosphere benchmark."""
    from src.gcm3d.coordinates import coordinate_system

    coords = coordinate_system(truncation, n_layers)
    specs = physics_specs(body)
    initial_fn, _ = primitive_equations_states.isothermal_rest_atmosphere(
        coords,
        specs,
        tref=body.reference_temperature_k * _u.kelvin,
        p0=body.reference_surface_pressure_pa * _u.pascal,
    )
    initial = initial_fn(jax.random.key(0))
    equation = primitive_equations(coords, body, specs=specs)
    final = jax.jit(lambda state: integrate(
        stepper(equation, dt_seconds, specs), state, n_steps
    ))(initial)
    return dry_drift_diagnostics(initial, final, coords)


def run_solid_body_tracer(
    body,
    truncation: str = "T21",
    speed_m_s: float = 200.0,
    dt_seconds: float = 300.0,
) -> TracerAdvectionDiagnostics:
    """Advect a Gaussian once around the equator in prescribed solid-body flow.

    This isolates the same spectral horizontal scalar-advection operator used by
    the primitive equations. For angular speed ``omega=U/a``, the nondivergent
    zonal wind is ``u=U cos(latitude)`` and the analytic return time is ``2*pi/omega``.
    """
    from src.gcm3d.coordinates import coordinate_system

    coords = coordinate_system(truncation, n_layers=1)
    specs = physics_specs(body)
    grid = coords.horizontal
    tracer0 = primitive_equations_states.gaussian_scalar(
        coords, specs, lon_location=0.0, lat_location=0.0,
        perturbation_radius=0.2, amplitude=1.0,
    )
    _, sin_lat = grid.nodal_mesh
    speed_nd = specs.nondimensionalize(speed_m_s * _u.meter / _u.second)
    # div_sec_lat consumes cos(latitude) times the physical velocity. For solid
    # angular rotation u_physical=U*cos(latitude), hence M=U*cos²(latitude).
    cos2_lat = jnp.clip(1.0 - sin_lat**2, 0.0, None)
    u = speed_nd * cos2_lat[None, :, :]
    v = jnp.zeros_like(u)

    def explicit(q_modal):
        q = grid.to_nodal(q_modal)
        return -div_sec_lat(u * q, v * q, grid)

    ode = time_integration.ImplicitExplicitODE.from_functions(
        explicit,
        lambda q: jnp.zeros_like(q),
        lambda q, step_size: q,
    )
    period_s = 2.0 * np.pi * body.radius_m / speed_m_s
    n_steps = max(1, round(period_s / dt_seconds))
    actual_dt = period_s / n_steps
    final = integrate(stepper(ode, actual_dt, specs), tracer0, n_steps)
    q0 = grid.to_nodal(tracer0)
    q1 = grid.to_nodal(final)
    weights = jnp.asarray(grid.quadrature_weights)[None, :, :]
    mass0 = jnp.sum(q0 * weights)
    mass1 = jnp.sum(q1 * weights)
    delta = q1 - q0
    l1 = jnp.sum(jnp.abs(delta) * weights) / jnp.sum(jnp.abs(q0) * weights)
    l2 = jnp.sqrt(jnp.sum(delta**2 * weights) / jnp.sum(q0**2 * weights))
    return TracerAdvectionDiagnostics(
        relative_mass_error=float(jnp.abs(mass1 - mass0) / jnp.abs(mass0)),
        relative_l1_error=float(l1),
        relative_l2_error=float(l2),
    )


def run_balanced_jet(
    body,
    truncation: str = "T21",
    n_layers: int = 12,
    dt_seconds: float = 600.0,
    duration_seconds: float = 86_400.0,
) -> DryDriftDiagnostics:
    """Run the unperturbed Jablonowski–Williamson balanced zonal jet.

    The exponential spectral filter is part of Dinosaur's canonical integration
    configuration for this benchmark. The default physical parameters of the
    published test are Earth-specific, so callers should normally pass ``EARTH``.
    """
    from src.gcm3d.coordinates import coordinate_system

    coords = coordinate_system(truncation, n_layers)
    specs = physics_specs(body)
    initial_fn, auxiliary = primitive_equations_states.steady_state_jw(coords, specs)
    initial = initial_fn()
    orography = dinosaur_pe.truncated_modal_orography(
        auxiliary[xarray_utils.OROGRAPHY], coords
    )
    equation = dinosaur_pe.PrimitiveEquationsSigma(
        auxiliary[xarray_utils.REF_TEMP_KEY], orography, coords, specs
    )
    dt = specs.nondimensionalize(dt_seconds * _u.second)
    advance = time_integration.imex_rk_sil3(equation, dt)
    advance = time_integration.step_with_filters(
        advance, [time_integration.exponential_step_filter(coords.horizontal, dt)]
    )
    n_steps = round(duration_seconds / dt_seconds)
    final = jax.jit(lambda state: integrate(advance, state, n_steps))(initial)
    return dry_drift_diagnostics(initial, final, coords)


def run_held_suarez(
    truncation: str = "T21",
    n_layers: int = 8,
    dt_seconds: float = 1200.0,
    spinup_days: int = 60,
    average_days: int = 20,
) -> HeldSuarezDiagnostics:
    """Run a compact Held–Suarez climate and average daily diagnostics.

    This CI configuration checks the circulation's canonical qualitative regime;
    publication comparisons should extend the integration to the 1000+ day protocol.
    """
    from src.gcm3d.body import EARTH
    from src.gcm3d.coordinates import coordinate_system

    coords = coordinate_system(truncation, n_layers)
    specs = physics_specs(EARTH)
    initial_fn, auxiliary = primitive_equations_states.isothermal_rest_atmosphere(
        coords, specs, tref=288.0 * _u.kelvin, p0=1.0e5 * _u.pascal,
        p1=100.0 * _u.pascal,
    )
    initial = initial_fn(jax.random.key(0))
    reference = auxiliary[xarray_utils.REF_TEMP_KEY]
    base = dinosaur_pe.PrimitiveEquationsSigma(
        reference, jnp.zeros(coords.horizontal.modal_shape), coords, specs
    )
    forcing = held_suarez.HeldSuarezForcingSigma(coords, specs, reference)
    equation = time_integration.compose_equations([base, forcing])
    dt = specs.nondimensionalize(dt_seconds * _u.second)
    advance = time_integration.imex_rk_sil3(equation, dt)
    advance = time_integration.step_with_filters(
        advance, [time_integration.exponential_step_filter(coords.horizontal, dt)]
    )
    steps_per_day = round(86_400.0 / dt_seconds)

    @jax.jit
    def one_day(state):
        return integrate(advance, state, steps_per_day)

    state = initial
    temperature_samples = []
    wind_samples = []
    for day in range(spinup_days + average_days):
        state = one_day(state)
        if day >= spinup_days:
            temperature_samples.append(
                coords.horizontal.to_nodal(state.temperature_variation)
                + np.asarray(reference)[:, None, None]
            )
            u, _ = spherical_harmonic.vor_div_to_uv_nodal(
                coords.horizontal, state.vorticity, state.divergence
            )
            wind_samples.append(
                specs.dimensionalize(u, _u.meter / _u.second).magnitude
            )
    temperature = np.mean(np.asarray(temperature_samples), axis=(0, 2))
    wind = np.mean(np.asarray(wind_samples), axis=(0, 2))
    lat = np.degrees(np.asarray(coords.horizontal.latitudes))
    equator = np.abs(lat) < 10.0
    poles = np.abs(lat) > 60.0
    mass0 = _weighted_surface_pressure(initial, coords)
    mass1 = _weighted_surface_pressure(state, coords)
    return HeldSuarezDiagnostics(
        equator_minus_pole_surface_temperature_k=float(
            temperature[-1, equator].mean() - temperature[-1, poles].mean()
        ),
        peak_eastward_wind_m_s=float(np.max(wind)),
        minimum_temperature_k=float(np.min(temperature)),
        relative_atmospheric_mass_drift=float(jnp.abs(mass1 - mass0) / mass0),
    )


def resting_convergence_matrix(
    body,
    truncations=("T21", "T42"),
    timesteps_s=(240.0, 120.0),
    n_layers: int = 6,
    duration_seconds: float = 12_000.0,
):
    """Return conservation drift for a fixed-duration resolution/timestep matrix."""
    from src.gcm3d.coordinates import coordinate_system

    results = {}
    for truncation in truncations:
        coords = coordinate_system(truncation, n_layers)
        specs = physics_specs(body)
        initial_fn, _ = primitive_equations_states.isothermal_rest_atmosphere(
            coords, specs,
            tref=body.reference_temperature_k * _u.kelvin,
            p0=body.reference_surface_pressure_pa * _u.pascal,
        )
        initial = initial_fn(jax.random.key(0))
        equation = primitive_equations(coords, body, specs=specs)
        reference = np.full(n_layers, body.reference_temperature_k)
        for dt_seconds in timesteps_s:
            n_steps = round(duration_seconds / dt_seconds)
            final = jax.jit(lambda state: integrate(
                stepper(equation, dt_seconds, specs), state, n_steps
            ))(initial)
            results[(truncation, dt_seconds)] = conservation_drift(
                initial, final, coords, specs, body, reference
            )
    return results
