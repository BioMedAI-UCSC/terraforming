"""Physics-specs for the dinosaur primitive equations, from a ``BodyConstants``.

This is the seam by which planetary constants reach the governing equations —
planet-agnostic. Crucially, it does **not** require patching dinosaur's
Earth-hardcoded ``scales.py``: dinosaur's equations consume a
``units.SimUnitsProtocol`` (``physics_specs``) carrying radius, angular velocity,
gravity, gas constant and kappa, built via ``SimUnits.from_si``. We pass the
values from any :class:`~src.gcm3d.body.BodyConstants`.

Nondimensionalisation convention (matches dinosaur/NeuralGCM): length is scaled
by the body radius and time by ``1/(2Ω)``, so any body has ``radius = 1`` and
``angular_velocity = 0.5`` in nondimensional units — consistent with the spectral
``Grid`` whose radius defaults to 1. Using Earth's default scale for a smaller
body makes its radius ≠ 1 and the equations reject the mismatch.
"""

from __future__ import annotations

from src.gcm3d._dinosaur import scales, units
from src.gcm3d.body import BodyConstants

_u = scales.units


def nondimensionalization_scale(body: BodyConstants) -> "scales.Scale":
    """Body-anchored nondimensionalisation scale (length=radius, time=1/2Ω)."""
    return scales.Scale(
        body.radius_m * _u.meter,
        (1.0 / (2.0 * body.angular_velocity_s)) * _u.second,
        1.0 * _u.kilogram,
        1.0 * _u.kelvin,
    )


def physics_specs(body: BodyConstants) -> "units.SimUnits":
    """Build the ``physics_specs`` (``units.SimUnits``) for ``body``.

    Threads the body's radius, rotation, gravity, gas constant and kappa into a
    dinosaur ``SimUnits`` using the body-anchored scale, so ``radius`` and
    ``angular_velocity`` nondimensionalise to the grid's convention.
    """
    return units.SimUnits.from_si(
        radius_si=body.radius_m * _u.meter,
        angular_velocity_si=body.angular_velocity_s / _u.second,
        gravity_acceleration_si=body.gravity_m_s2 * _u.meter / _u.second**2,
        ideal_gas_constant_si=body.gas_constant_j_kg_k
        * _u.joule
        / (_u.kilogram * _u.kelvin),
        kappa_si=body.kappa * _u.dimensionless,
        scale=nondimensionalization_scale(body),
    )
