"""PROTOTYPE — the 0-D terraforming ODE expressed on the dinosaur substrate.

This module is a **proof of concept** for the "dinosaur as the engine backbone"
direction: it takes the existing 0-D coupled ODE physics
(:meth:`src.celestials.planets.mars.Mars.compute_derivatives`, a torch RK4 kernel
integrated by ``engine.TimeController``) and re-expresses it as a
``dinosaur.time_integration.ImplicitExplicitODE`` — the *same* ODE abstraction the
3-D primitive equations use — so it is integrated by dinosaur's *same* stepper
(``imex_rk_sil3``) and the same scan helper
(:func:`src.framework.gcm.dynamics.integrate`).

Why this proves the direction:
  - **Substrate reuse.** Our terraforming physics becomes a first-class dinosaur
    ODE. The 0-D system is non-stiff, so it maps to an ImEx ODE with *zero*
    implicit terms — dinosaur's IMEX-RK-SIL3 then degenerates to its explicit
    Runge-Kutta tableau, a valid explicit integrator. Nothing about the physics
    is special-cased; it rides the same machinery as the dycore.
  - **Differentiable + batchable.** Because the whole rollout is JAX, gradients
    flow end-to-end (``jax.grad``) and it batches with ``jax.vmap`` — the two
    properties the ICLR world-model / inverse-design work needs at 10^5–10^6
    trajectories, which is JAX's strength.
  - **Exact parity.** :func:`tendency` is a line-for-line port of
    ``compute_derivatives``; the tests assert it matches the torch kernel to
    float64 machine precision on random states, and that a dinosaur-stepped
    rollout tracks the torch RK4 rollout to the integration schemes' truncation
    order.

What this is **not** (scope of the prototype, deliberately minimal):
  - The orbital forcing (solar flux, orbital angle, elapsed time) and the two
    polar-cap reservoirs are held **frozen** at an epoch and passed in as
    :class:`ZeroDForcing`. The full engine port would instead carry ``sim_time``
    in the state (dinosaur states already do) and advance the orbit inside
    ``explicit_terms`` — see ``docs/ideas/dinosaur-mars-workplan.md``.
  - It does not touch ``engine/`` or replace the torch solver. It is a branch to
    inform the go/no-go on committing the engine to the JAX substrate.

Torch-free by construction: the parity comparison against the torch model lives
in the tests (the experiment layer), never here — preserving the torch/JAX
isolation. Requires the optional ``gcm3d`` extra.
"""

from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path

import numpy as np

from src.framework.gcm._dinosaur import jax, jnp, time_integration

# State layout for the 0-D coupled system, matching the torch model's ``y``.
T_IDX, P_IDX, MICE_IDX = 0, 1, 2

# Astronomical constants shared with the torch orbit model (device-agnostic
# Python floats): total solar irradiance at 1 AU and the astronomical unit.
TSI_1AU_W_M2 = 1361.0
AU_M = 1.49597870700e11
_TWO_PI = 2.0 * math.pi


@dataclasses.dataclass(frozen=True)
class ZeroDForcing:
    """Frozen constants + epoch forcing for the 0-D terraforming ODE.

    Every field is a plain float (or bool), so this dataclass is pure Python and
    carries no torch/JAX objects — the engine (torch side) can build it from a
    ``Mars`` instance without importing JAX. Values mirror the cached ``self._*``
    constants and current planet/orbit state read by ``compute_derivatives``.
    """

    # Radiative / thermal
    albedo: float
    solar_flux: float
    greenhouse_factor: float
    emissivity: float
    stefan_boltzmann: float
    thermal_inertia: float
    # Geometry / orbit (frozen at an epoch)
    radius_m: float
    gravity_m_s2: float
    rotation_period_s: float
    axial_tilt_rad: float
    ls_perihelion_rad: float
    latitude_rad: float
    elapsed_time_s: float
    orbital_angle_rad: float
    # Polar CO2 caps
    cap_fraction: float
    q_out_pole: float
    latent_heat: float
    ice_north_kg: float
    ice_south_kg: float
    ice_ref_kg: float
    # Non-thermal escape
    escape_rate_kg_s: float
    # Sublimation-gate mode (smooth = differentiable through cap exhaustion)
    smooth_gates: bool = True


def _gate_sublimation(dM, ice, ice_ref: float, smooth: bool):
    """JAX port of ``Mars._gate_sublimation`` (smooth tanh or hard snap)."""
    if smooth:
        return jnp.where(dM < 0.0, dM * jnp.tanh(ice / ice_ref), dM)
    return jnp.where((ice <= 0.0) & (dM < 0.0), jnp.zeros_like(dM), dM)


def tendency(y, f: ZeroDForcing):
    """dy/dt for ``y = [T, P, M_ice]`` — a JAX port of ``compute_derivatives``.

    Exact transcription of the torch kernel with the orbital forcing and polar
    reservoirs frozen in ``f``. Returns a length-3 array ``[dT, dP, dM_ice]``.
    """
    T = jnp.clip(y[T_IDX], 1.0, None)
    # P (y[P_IDX]) and M_ice (y[MICE_IDX]) are clamped in the kernel but do not
    # enter the tendencies; kept implicit for exact parity.

    # --- dT/dt: diurnal energy balance ---
    omega = 2.0 * math.pi / f.rotation_period_s
    h = omega * f.elapsed_time_s - math.pi
    Ls = f.orbital_angle_rad + f.ls_perihelion_rad
    delta = jnp.arcsin(jnp.sin(f.axial_tilt_rad) * jnp.sin(Ls))
    cos_zenith = jnp.clip(
        jnp.sin(f.latitude_rad) * jnp.sin(delta)
        + jnp.cos(f.latitude_rad) * jnp.cos(delta) * jnp.cos(h),
        0.0,
        None,
    )
    Q_in = (1.0 - f.albedo) * f.solar_flux * cos_zenith
    T_eff = T / max(f.greenhouse_factor, 1.0)
    Q_out = f.emissivity * f.stefan_boltzmann * T_eff**4
    dT_dt = (Q_in - Q_out) / f.thermal_inertia

    # --- dM_ice/dt: polar CO2 sublimation / condensation ---
    A_cap = f.cap_fraction * 4.0 * math.pi * f.radius_m**2
    cz_N = jnp.clip(jnp.sin(delta), 0.0, None)
    cz_S = jnp.clip(-jnp.sin(delta), 0.0, None)
    Q_in_N = (1.0 - f.albedo) * f.solar_flux * cz_N
    Q_in_S = (1.0 - f.albedo) * f.solar_flux * cz_S
    net_sub_N = (Q_in_N - f.q_out_pole) * A_cap / f.latent_heat
    net_sub_S = (Q_in_S - f.q_out_pole) * A_cap / f.latent_heat
    dMice_N = _gate_sublimation(-net_sub_N, f.ice_north_kg, f.ice_ref_kg, f.smooth_gates)
    dMice_S = _gate_sublimation(-net_sub_S, f.ice_south_kg, f.ice_ref_kg, f.smooth_gates)
    dMice_dt = dMice_N + dMice_S

    # --- dP/dt: non-thermal escape + cap mass exchange (mass budget) ---
    A_planet = 4.0 * math.pi * f.radius_m**2
    dP_dt = (
        -f.escape_rate_kg_s * f.gravity_m_s2 / A_planet
        + (-dMice_dt * f.gravity_m_s2 / A_planet)
    )

    return jnp.stack([dT_dt, dP_dt, dMice_dt])


def terraforming_ode(f: ZeroDForcing) -> "time_integration.ImplicitExplicitODE":
    """Wrap the 0-D terraforming physics as a dinosaur ``ImplicitExplicitODE``.

    The system is non-stiff, so the implicit side is empty: ``implicit_terms`` is
    zero and ``implicit_inverse`` is the identity. Dinosaur's IMEX stepper then
    reduces to its explicit Runge-Kutta tableau — the physics is stepped purely
    explicitly, exactly as the torch RK4 kernel is.
    """

    def explicit_terms(y):
        return tendency(y, f)

    def implicit_terms(y):
        return jax.tree_util.tree_map(jnp.zeros_like, y)

    def implicit_inverse(y, step_size):  # (1 - dt * 0)^-1 = identity
        return y

    return time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms, implicit_terms, implicit_inverse
    )


def stepper(ode: "time_integration.ImplicitExplicitODE", dt_seconds: float):
    """A single-step function for the 0-D ODE.

    Unlike :func:`src.framework.gcm.dynamics.stepper` (3-D), the 0-D system is dimensional, so
    ``dt_seconds`` is passed to dinosaur's stepper directly — no
    nondimensionalisation.
    """
    return time_integration.imex_rk_sil3(ode, dt_seconds)


# ==============================================================================
# Seasonal 0-D ODE — the orbit advances *inside* the ODE, so a rollout produces
# a full areocentric-solar-longitude (Ls) seasonal cycle.
#
# This drops the ``ZeroDForcing`` "frozen at an epoch" assumption (the one item
# the prototype deferred): elapsed time ``t`` is carried in the state, and the
# orbital angle, solar longitude ``Ls``, insolation and solar flux are derived
# from ``t`` every step — exactly as ``BatchedController.advance_orbit`` +
# ``compute_derivatives`` do on the torch side. The state is the two-cap layout
# main uses (``[T, P, M_north, M_south]``) plus ``t``, so each polar cap gates
# its own reservoir and the pair exchanges in anti-phase across the seasons.
#
# Deliberately *not* included (out of scope, per the workplan): the FAST
# relaxation overlay (thermal tide, diurnal swing) — those are diagnostics on
# top of ``compute_fast_physics``, not mass/energy tendencies. And there is no
# horizontal grid, so there is no orography: MOLA topography is irrelevant to a
# global-mean 0-D column. It only matters to the 3-D dycore's surface-pressure
# field (which still uses ``flat_orography``).
# ==============================================================================

# Seasonal state layout: [T, P, M_north, M_south, t].
ST_T, ST_P, ST_MN, ST_MS, ST_TIME = 0, 1, 2, 3, 4


@dataclasses.dataclass(frozen=True)
class SeasonalForcing:
    """Constants + orbital elements for the time-advancing 0-D terraforming ODE.

    Unlike :class:`ZeroDForcing`, the solar flux, orbital angle and polar caps
    are **not** frozen: the flux and ``Ls`` are recomputed from the state's
    elapsed time ``t`` each step, and the caps are integrated state variables.
    Every field is a plain float/bool, so this stays pure Python — the torch
    engine can build it from a ``Mars`` without importing JAX.
    """

    # Radiative / thermal
    albedo: float
    greenhouse_factor: float
    emissivity: float
    stefan_boltzmann: float
    thermal_inertia: float
    # Geometry / rotation
    radius_m: float
    gravity_m_s2: float
    rotation_period_s: float
    latitude_rad: float
    axial_tilt_rad: float
    ls_perihelion_rad: float
    # Orbit (Keplerian ellipse; mean anomaly advances uniformly and Kepler's
    # equation determines eccentric/true anomaly)
    orbital_period_s: float
    semi_major_axis_m: float
    eccentricity: float
    init_orbital_angle_rad: float
    # Polar CO2 caps
    cap_fraction: float
    q_out_pole: float
    latent_heat: float
    ice_ref_kg: float
    # Non-thermal escape
    escape_rate_kg_s: float
    # Astronomical constants (overridable for other bodies)
    tsi_1au_w_m2: float = TSI_1AU_W_M2
    au_m: float = AU_M
    # Sublimation-gate mode (smooth = differentiable through cap exhaustion)
    smooth_gates: bool = True


def mean_anomaly(t, f: SeasonalForcing):
    """Mean anomaly, the orbital angle that advances uniformly in time."""
    return f.init_orbital_angle_rad + _TWO_PI * t / f.orbital_period_s


def eccentric_anomaly(t, f: SeasonalForcing):
    """Solve Kepler's equation ``M = E - e sin(E)`` by fixed Newton iteration."""
    M = mean_anomaly(t, f)
    E = M
    for _ in range(6):
        E = E - (E - f.eccentricity * jnp.sin(E) - M) / (
            1.0 - f.eccentricity * jnp.cos(E)
        )
    return E


def orbital_angle(t, f: SeasonalForcing):
    """True anomaly at elapsed time ``t`` from a Keplerian orbit."""
    E = eccentric_anomaly(t, f)
    e = f.eccentricity
    return 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0),
    )


def solar_longitude(t, f: SeasonalForcing):
    """Areocentric solar longitude ``Ls`` (radians) at elapsed time ``t``."""
    return orbital_angle(t, f) + f.ls_perihelion_rad


def solar_flux(t, f: SeasonalForcing):
    """Inverse-square solar flux (W m^-2) at elapsed time ``t``.

    Kepler distance ``r = a(1-e cos E)`` then ``S = S_1AU·(AU/r)²``.
    """
    E = eccentric_anomaly(t, f)
    distance = f.semi_major_axis_m * (1.0 - f.eccentricity * jnp.cos(E))
    return f.tsi_1au_w_m2 * (f.au_m / distance) ** 2


def seasonal_tendency(y, f: SeasonalForcing):
    """dy/dt for ``y = [T, P, M_north, M_south, t]`` — the two-cap kernel.

    A line-for-line JAX port of ``Mars.compute_derivatives`` (main's two-cap
    kernel), with the orbital forcing rebuilt from ``t`` instead of frozen. The
    5th component's tendency is 1 (``dt/dt = 1``), so the orbit sweeps a full
    year over one orbital period.
    """
    T = jnp.clip(y[ST_T], 1.0, None)
    ice_N = y[ST_MN]
    ice_S = y[ST_MS]
    t = y[ST_TIME]

    # --- orbital forcing derived from elapsed time ---
    omega = _TWO_PI / f.rotation_period_s
    h = omega * t - math.pi
    Ls = solar_longitude(t, f)
    delta = jnp.arcsin(jnp.sin(f.axial_tilt_rad) * jnp.sin(Ls))
    S = solar_flux(t, f)

    # --- dT/dt: diurnal energy balance ---
    cos_zenith = jnp.clip(
        jnp.sin(f.latitude_rad) * jnp.sin(delta)
        + jnp.cos(f.latitude_rad) * jnp.cos(delta) * jnp.cos(h),
        0.0,
        None,
    )
    Q_in = (1.0 - f.albedo) * S * cos_zenith
    T_eff = T / max(f.greenhouse_factor, 1.0)
    Q_out = f.emissivity * f.stefan_boltzmann * T_eff**4
    dT_dt = (Q_in - Q_out) / f.thermal_inertia

    # --- dM_ice/dt: per-cap polar CO2 sublimation / condensation ---
    A_cap = f.cap_fraction * 4.0 * math.pi * f.radius_m**2
    cz_N = jnp.clip(jnp.sin(delta), 0.0, None)
    cz_S = jnp.clip(-jnp.sin(delta), 0.0, None)
    Q_in_N = (1.0 - f.albedo) * S * cz_N
    Q_in_S = (1.0 - f.albedo) * S * cz_S
    net_sub_N = (Q_in_N - f.q_out_pole) * A_cap / f.latent_heat
    net_sub_S = (Q_in_S - f.q_out_pole) * A_cap / f.latent_heat
    dMice_N = _gate_sublimation(-net_sub_N, ice_N, f.ice_ref_kg, f.smooth_gates)
    dMice_S = _gate_sublimation(-net_sub_S, ice_S, f.ice_ref_kg, f.smooth_gates)

    # --- dP/dt: non-thermal escape + cap mass exchange (mass budget) ---
    A_planet = 4.0 * math.pi * f.radius_m**2
    dP_dt = (
        -f.escape_rate_kg_s * f.gravity_m_s2 / A_planet
        + (-(dMice_N + dMice_S) * f.gravity_m_s2 / A_planet)
    )

    return jnp.stack([dT_dt, dP_dt, dMice_N, dMice_S, jnp.ones_like(dT_dt)])


def seasonal_ode(f: SeasonalForcing) -> "time_integration.ImplicitExplicitODE":
    """Wrap :func:`seasonal_tendency` as a dinosaur ``ImplicitExplicitODE``.

    Non-stiff, so the implicit side is empty (identity inverse) and dinosaur's
    IMEX-RK-SIL3 reduces to its explicit tableau — same as :func:`terraforming_ode`.
    """

    def explicit_terms(y):
        return seasonal_tendency(y, f)

    def implicit_terms(y):
        return jax.tree_util.tree_map(jnp.zeros_like, y)

    def implicit_inverse(y, step_size):
        return y

    return time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms, implicit_terms, implicit_inverse
    )


def initial_seasonal_state(
    temperature_k: float,
    pressure_pa: float,
    ice_north_kg: float,
    ice_south_kg: float,
    t0_s: float = 0.0,
):
    """Build a seasonal state array ``[T, P, M_north, M_south, t]``."""
    return jnp.asarray(
        [temperature_k, pressure_pa, ice_north_kg, ice_south_kg, t0_s],
        dtype=jnp.float64,
    )


@dataclasses.dataclass(frozen=True)
class SeasonalTrajectory:
    """Sampled seasonal rollout with the Ls-indexed diagnostics.

    All fields are NumPy arrays of equal length (one entry per sample). ``ls_deg``
    is the areocentric solar longitude in ``[0, 360)`` — the season coordinate
    every Mars seasonal plot is drawn against.
    """

    time_s: np.ndarray
    sol: np.ndarray
    ls_deg: np.ndarray
    temperature_k: np.ndarray
    pressure_pa: np.ndarray
    ice_north_kg: np.ndarray
    ice_south_kg: np.ndarray
    ice_total_kg: np.ndarray
    solar_flux_wm2: np.ndarray

    def write_csv(self, path) -> Path:
        """Write the trajectory as a CSV (Ls-indexed), returning the path.

        Columns mirror the existing Mars seasonal exports so the same plotting
        code (Ls on the x-axis) works unchanged.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cols = [
            "sol",
            "ls_deg",
            "temperature_k",
            "pressure_pa",
            "ice_mass_kg",
            "ice_north_kg",
            "ice_south_kg",
            "solar_flux_wm2",
        ]
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(cols)
            for i in range(len(self.time_s)):
                w.writerow(
                    [
                        f"{self.sol[i]:.6f}",
                        f"{self.ls_deg[i]:.4f}",
                        f"{self.temperature_k[i]:.6f}",
                        f"{self.pressure_pa[i]:.6f}",
                        f"{self.ice_total_kg[i]:.6e}",
                        f"{self.ice_north_kg[i]:.6e}",
                        f"{self.ice_south_kg[i]:.6e}",
                        f"{self.solar_flux_wm2[i]:.6f}",
                    ]
                )
        return path


def run_seasonal(
    f: SeasonalForcing,
    y0,
    dt_seconds: float,
    n_steps: int,
    sample_every: int = 1,
) -> SeasonalTrajectory:
    """Integrate the seasonal ODE and return an Ls-indexed :class:`SeasonalTrajectory`.

    Steps ``n_steps`` of dinosaur's stepper (via :func:`jax.lax.scan`), recording
    the state every ``sample_every`` steps, then derives ``Ls``, ``sol`` and the
    solar flux from each sampled elapsed time. The rollout itself is a single
    jitted scan, so it stays differentiable/batchable; only the returned arrays
    are pulled to host as NumPy for output.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if sample_every < 1:
        raise ValueError(f"sample_every must be >= 1, got {sample_every}")
    # Stability: the explicit step must resolve the diurnal energy balance
    # (hour angle h = 2π t / rotation_period). Near ~1 step per rotation the
    # T^4 radiative relaxation aliases and the integrator diverges to
    # non-physical temperatures, so refuse a step that coarse rather than emit
    # silently-wrong seasonal output. ~8 steps/rotation is the stability floor;
    # ~40+ (dt <= rotation_period/40) is recommended for converged accuracy.
    max_stable_dt = f.rotation_period_s / 8.0
    if dt_seconds > max_stable_dt:
        raise ValueError(
            f"dt_seconds={dt_seconds:g} is too coarse to resolve the diurnal "
            f"cycle (rotation_period={f.rotation_period_s:g} s); the explicit "
            f"integrator would diverge. Use dt_seconds <= {max_stable_dt:g} "
            f"(recommend <= {f.rotation_period_s / 40.0:g})."
        )

    step = stepper(seasonal_ode(f), dt_seconds)
    n_samples = n_steps // sample_every
    remainder = n_steps % sample_every

    def outer(carry, _):
        def inner(yy, _):
            return step(yy), None

        y, _ = jax.lax.scan(inner, carry, None, length=sample_every)
        return y, y

    carry, samples = jax.lax.scan(outer, y0, None, length=n_samples)
    if remainder:
        def remainder_step(yy, _):
            return step(yy), None

        carry, _ = jax.lax.scan(remainder_step, carry, None, length=remainder)
        samples = jnp.concatenate([samples, carry[None, :]], axis=0)
    samples = np.asarray(samples)  # [n_samples, 5]

    t = samples[:, ST_TIME]
    mean = f.init_orbital_angle_rad + _TWO_PI * t / f.orbital_period_s
    eccentric = mean.copy()
    for _ in range(6):
        eccentric -= (eccentric - f.eccentricity * np.sin(eccentric) - mean) / (
            1.0 - f.eccentricity * np.cos(eccentric)
        )
    theta = 2.0 * np.arctan2(
        np.sqrt(1.0 + f.eccentricity) * np.sin(eccentric / 2.0),
        np.sqrt(1.0 - f.eccentricity) * np.cos(eccentric / 2.0),
    )
    ls_deg = np.degrees(theta + f.ls_perihelion_rad) % 360.0
    distance = f.semi_major_axis_m * (1.0 - f.eccentricity * np.cos(eccentric))
    flux = f.tsi_1au_w_m2 * (f.au_m / distance) ** 2
    ice_n = samples[:, ST_MN]
    ice_s = samples[:, ST_MS]

    return SeasonalTrajectory(
        time_s=t,
        sol=t / f.rotation_period_s,
        ls_deg=ls_deg,
        temperature_k=samples[:, ST_T],
        pressure_pa=samples[:, ST_P],
        ice_north_kg=ice_n,
        ice_south_kg=ice_s,
        ice_total_kg=ice_n + ice_s,
        solar_flux_wm2=flux,
    )
