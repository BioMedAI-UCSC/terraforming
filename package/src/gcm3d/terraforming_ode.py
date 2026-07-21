"""PROTOTYPE — the 0-D terraforming ODE expressed on the dinosaur substrate.

This module is a **proof of concept** for the "dinosaur as the engine backbone"
direction: it takes the existing 0-D coupled ODE physics
(:meth:`src.celestials.planets.mars.Mars.compute_derivatives`, a torch RK4 kernel
integrated by ``engine.TimeController``) and re-expresses it as a
``dinosaur.time_integration.ImplicitExplicitODE`` — the *same* ODE abstraction the
3-D primitive equations use — so it is integrated by dinosaur's *same* stepper
(``imex_rk_sil3``) and the *same* scan helper (:func:`src.gcm3d.integrate`).

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

import dataclasses
import math

from src.gcm3d._dinosaur import jax, jnp, time_integration

# State layout for the 0-D coupled system, matching the torch model's ``y``.
T_IDX, P_IDX, MICE_IDX = 0, 1, 2


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

    Unlike :func:`src.gcm3d.stepper` (3-D), the 0-D system is dimensional, so
    ``dt_seconds`` is passed to dinosaur's stepper directly — no
    nondimensionalisation.
    """
    return time_integration.imex_rk_sil3(ode, dt_seconds)
