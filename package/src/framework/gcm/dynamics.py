"""Dry primitive-equations dynamics on the dinosaur sigma-coordinate core.

Planet-agnostic: every builder takes a
:class:`~src.framework.gcm.body.BodyConstants`.
Assembles a runnable, differentiable dry dynamical core for any body::

    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm import physics_specs, primitive_equations, stepper, integrate
    coords = coordinate_system("T21", n_layers=12)
    specs  = physics_specs(body)
    eq     = primitive_equations(coords, body)
    step   = stepper(eq, dt_seconds=120.0, specs=specs)
    final  = integrate(step, initial_state, n_steps=300)

The equations are a ``dinosaur.primitive_equations.PrimitiveEquationsSigma`` (an
``ImplicitExplicitODE``); body-specific physics tendencies are added to
``explicit_terms`` in later phases. Reference temperature and orography are
construction-time config (static): the reference profile is the semi-implicit
linearisation anchor, and orography is the lower boundary (flat by default).
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.framework.gcm._dinosaur import jax
from src.framework.gcm._dinosaur import primitive_equations as _pe
from src.framework.gcm._dinosaur import scales, time_integration
from src.framework.gcm.body import BodyConstants
from src.framework.gcm.specs import physics_specs

_u = scales.units


def reference_temperature(coords, body: BodyConstants):
    """Per-layer reference temperature (nondimensional), the linearisation anchor.

    A constant body-representative profile (``body.reference_temperature_k``);
    dinosaur takes a fast path for constant profiles. Returned as a static numpy
    array of shape ``[n_layers]`` — model config, not a differentiated input.
    """
    specs = physics_specs(body)
    profile = np.full(coords.vertical.layers, body.reference_temperature_k)
    return specs.nondimensionalize(profile * _u.kelvin)


def flat_orography(coords):
    """Zero (flat) modal orography — the default lower boundary.

    Real topography, truncated to modal coefficients and passed to
    :func:`primitive_equations`, is supported via the same argument.
    """
    return jax.numpy.zeros(coords.horizontal.modal_shape)


def primitive_equations(coords, body: BodyConstants, specs=None, orography=None):
    """Build the dry ``PrimitiveEquationsSigma`` for ``body`` on ``coords``.

    Parameters
    ----------
    coords : dinosaur CoordinateSystem
        From :func:`src.framework.gcm.coordinates.coordinate_system`.
    body : BodyConstants
        Planet/moon constants.
    specs : units.SimUnits, optional
        Physics-specs; built from ``body`` if omitted.
    orography : Array, optional
        Modal orography (lower boundary). Flat if omitted.
    """
    if specs is None:
        specs = physics_specs(body)
    ref_temp = reference_temperature(coords, body)
    oro = flat_orography(coords) if orography is None else orography
    return _pe.PrimitiveEquationsSigma(ref_temp, oro, coords, specs)


def horizontal_hyperdiffusion_filter(coords, dt_seconds: float, specs,
                                    tau_seconds: float, order: int = 4):
    """Post-step damping of modal winds and temperature, leaving inventories alone.

    The shortest mode decays by ``exp(-dt_seconds / tau_seconds)`` per step.
    ``order`` is the power of the Laplacian: 4 means eighth-order diffusion.
    Pressure, tracers, time, and nodal surface reservoirs are not filtered:
    filtering log pressure or a mixing ratio does not conserve material mass.
    Tracers require their own conservative, positivity-preserving transport.
    """
    for name, value in (("dt_seconds", dt_seconds), ("tau_seconds", tau_seconds)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and > 0, got {value}")
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("hyperdiffusion order must be a positive integer")
    dt = float(specs.nondimensionalize(dt_seconds * _u.second))
    tau = float(specs.nondimensionalize(tau_seconds * _u.second))
    filter_dynamics = time_integration.horizontal_diffusion_step_filter(
        coords.horizontal, dt, tau, order=order
    )

    def apply(previous, next_state):
        column = hasattr(next_state, "dynamics")
        dynamics = next_state.dynamics if column else next_state
        previous_dynamics = previous.dynamics if column else previous
        filtered = filter_dynamics(previous_dynamics, dynamics)
        dynamics = dataclasses.replace(
            dynamics,
            vorticity=filtered.vorticity,
            divergence=filtered.divergence,
            temperature_variation=filtered.temperature_variation,
        )
        return next_state._replace(dynamics=dynamics) if column else dynamics

    return apply


def stepper(equation, dt_seconds: float, specs, *, coords=None,
            hyperdiffusion_tau_seconds: float | None = None):
    """Semi-implicit (IMEX-RK-SIL3) time-step function for ``equation``.

    ``dt_seconds`` is physical; it is nondimensionalised with ``specs`` before
    being handed to dinosaur's stepper. Pass ``coords`` and a positive
    ``hyperdiffusion_tau_seconds`` to damp unresolved-scale wind/temperature
    noise after each step. Low-level dynamics benchmarks retain the explicit
    opt-in; the user-facing Mars map runner enables damping by default.
    """
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be finite and > 0, got {dt_seconds}")
    if hyperdiffusion_tau_seconds is not None and coords is None:
        raise ValueError("coords is required when hyperdiffusion is enabled")
    dt = specs.nondimensionalize(dt_seconds * _u.second)
    step = time_integration.imex_rk_sil3(equation, dt)
    if hyperdiffusion_tau_seconds is not None:
        diffusion = horizontal_hyperdiffusion_filter(
            coords, dt_seconds, specs, hyperdiffusion_tau_seconds
        )
        step = time_integration.step_with_filters(step, [diffusion])
    return step


def integrate(step_fn, state, n_steps: int):
    """Advance ``state`` by ``n_steps`` of ``step_fn`` via ``jax.lax.scan``.

    Differentiable end-to-end: gradients flow from a diagnostic of the returned
    state back to the initial ``state`` (verified in the tests).
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")

    def body(carry, _):
        return step_fn(carry), None

    final, _ = jax.lax.scan(body, state, None, length=n_steps)
    return final
