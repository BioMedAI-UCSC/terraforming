"""Placeholder Mars physics for the JCM backend.

This module gives the JCM model a simple, data-free physics.  It gets the
``MarsGCM`` pipeline to step end-to-end before the real Mars terms exist
(CO₂ cycle, dust, radiation).  Replace it in a later step.

Design: JCM's ``HeldSuarez`` term is generic.  It applies Newtonian relaxation
of the temperature toward an analytic equilibrium, plus Rayleigh friction near
the surface.  Only its numbers are Earth-specific.  So a Mars placeholder is the
same term with Mars numbers.  We subclass ``HeldSuarez`` and change the
defaults.  We do not copy the analytic code.

The values here are rough placeholders.  They give a stable, dry Mars-like
circulation.  They are **not** a validated Mars climatology.

Reference: Held & Suarez (1994), the standard dynamical-core benchmark.
"""

from __future__ import annotations

from typing import Any

from dinosaur.scales import units

from jcm.physics.composable_physics import ComposablePhysics
from jcm.physics.held_suarez.held_suarez_physics import HeldSuarez

# The parameters are dinosaur unit Quantities (a pint generic).  We annotate
# them as Any: pint's Quantity is a bound-TypeVar generic and is not a valid
# bare annotation form for the type checker.

# Mars placeholder parameters (rough — see module docstring).
# Temperatures: Mars is cold; strong equator-to-pole contrast.
_MARS_MIN_T: Any = 140 * units.degK   # polar-winter floor, near CO₂ frost
_MARS_MAX_T: Any = 250 * units.degK   # sub-solar equatorial peak
_MARS_DTY: Any = 90 * units.degK      # equator-to-pole contrast
_MARS_DTHZ: Any = 10 * units.degK     # vertical-stability term
# Relaxation rates: Mars air is thin, so it relaxes fast (a few sols).
_MARS_KA: Any = 1 / (5 * units.day)     # free atmosphere (slow)
_MARS_KS: Any = 1 / (0.5 * units.day)   # near surface (fast)
_MARS_KF: Any = 1 / (1 * units.day)     # Rayleigh friction


class MarsHeldSuarez(HeldSuarez):
    """Held-Suarez forcing with Mars default parameters.

    This is a placeholder.  It reuses the JCM Held-Suarez analytic form and only
    changes the default numbers to Mars values.  The parent class reads the live
    constants singleton at construction, so it uses the Mars constants that
    ``_apply_mars_constants`` sets first.
    """

    name = "mars_held_suarez"
    category = "mars_held_suarez"

    def __init__(
        self,
        sigma_b: float = 0.7,
        kf: Any = _MARS_KF,
        ka: Any = _MARS_KA,
        ks: Any = _MARS_KS,
        minT: Any = _MARS_MIN_T,
        maxT: Any = _MARS_MAX_T,
        dTy: Any = _MARS_DTY,
        dThz: Any = _MARS_DTHZ,
    ) -> None:
        """Set the Mars defaults and hand them to the parent term."""
        super().__init__(
            sigma_b=sigma_b,
            kf=kf,
            ka=ka,
            ks=ks,
            minT=minT,
            maxT=maxT,
            dTy=dTy,
            dThz=dThz,
        )


def mars_physics(**kwargs) -> ComposablePhysics:
    """Return a ComposablePhysics with the single Mars placeholder term.

    Pass keyword arguments to override the Mars defaults.  They go to
    ``MarsHeldSuarez.__init__``.

    Call ``_apply_mars_constants()`` before this function.  The term reads the
    constants singleton when you build it.
    """
    return ComposablePhysics(
        terms=[MarsHeldSuarez(**kwargs)],
        checkpoint_terms=False,
        vectorize_columns=False,
    )
