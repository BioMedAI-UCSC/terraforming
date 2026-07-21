"""Spectral coordinate systems for the 3-D GCM core (planet-agnostic).

Thin wrappers around ``dinosaur``'s spherical-harmonic grid and terrain-following
sigma vertical coordinate. These are the same for any planet — the discretisation
does not depend on which body is being simulated — so this module takes no
``BodyConstants``; body constants enter through the equations/physics-specs
(see ``gcm3d.specs`` / ``gcm3d.dynamics``).
"""

from __future__ import annotations

from src.gcm3d._dinosaur import (
    coordinate_systems,
    sigma_coordinates,
    spherical_harmonic,
)

# Named triangular truncations. T21 is the smoke/dev resolution; T42/T85 are for
# production runs.
_GRIDS = {
    "T21": spherical_harmonic.Grid.T21,
    "T42": spherical_harmonic.Grid.T42,
    "T85": spherical_harmonic.Grid.T85,
}


def grid(truncation: str = "T42") -> spherical_harmonic.Grid:
    """Return the horizontal spectral ``Grid`` for a named triangular truncation.

    Parameters
    ----------
    truncation : {"T21", "T42", "T85"}
        Triangular spectral truncation. T21 for smoke tests, T42 default, T85
        for higher resolution.
    """
    try:
        return _GRIDS[truncation]()
    except KeyError:
        raise ValueError(
            f"unknown truncation {truncation!r}; expected one of {sorted(_GRIDS)}"
        ) from None


def coordinate_system(
    truncation: str = "T42",
    n_layers: int = 25,
) -> coordinate_systems.CoordinateSystem:
    """Build a ``CoordinateSystem`` = spectral grid × sigma levels.

    Parameters
    ----------
    truncation : {"T21", "T42", "T85"}
        Horizontal spectral truncation (see :func:`grid`).
    n_layers : int
        Number of equidistant terrain-following sigma levels (p/p_s). Sigma
        coordinates rescale with surface pressure automatically — essential for
        bodies with large seasonal surface-pressure swings (e.g. Mars's CO2
        cycle moves ~25 % of the atmosphere).
    """
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    return coordinate_systems.CoordinateSystem(
        grid(truncation), sigma_coordinates.SigmaCoordinates.equidistant(n_layers)
    )
