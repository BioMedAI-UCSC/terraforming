"""Dry primitive-equations dynamics on the dinosaur sigma-coordinate core.

Planet-agnostic: every builder takes a :class:`~src.gcm3d.body.BodyConstants`.
Assembles a runnable, differentiable dry dynamical core for any body::

    from src.gcm3d import coordinate_system, physics_specs, primitive_equations, stepper, integrate
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

import numpy as np

from src.gcm3d._dinosaur import jax
from src.gcm3d._dinosaur import primitive_equations as _pe
from src.gcm3d._dinosaur import scales, time_integration
from src.gcm3d.body import BodyConstants
from src.gcm3d.specs import physics_specs

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
        From :func:`src.gcm3d.coordinate_system`.
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


def stepper(equation, dt_seconds: float, specs):
    """Semi-implicit (IMEX-RK-SIL3) time-step function for ``equation``.

    ``dt_seconds`` is physical; it is nondimensionalised with ``specs`` before
    being handed to dinosaur's stepper.
    """
    dt = specs.nondimensionalize(dt_seconds * _u.second)
    return time_integration.imex_rk_sil3(equation, dt)


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
