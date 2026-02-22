"""Celestials package – single public API surface.

All public symbols are exported from this top-level package.
Subfolders (``framework/``, ``planets/``) are plain directories containing
``.py`` files, **not** Python sub-packages (no ``__init__.py``).

Usage::

    from src.celestials import Planet, PlanetaryState, OrbitalParameters
    from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

_PKG_DIR = os.path.dirname(__file__)


def _load(dotted_name: str, rel_path: str) -> types.ModuleType:
    """Load a ``.py`` file and register it under *dotted_name* in ``sys.modules``.

    This lets files inside ``src/`` keep their existing cross-file imports
    (e.g. ``from src.celestials.framework.planet import Planet``) without
    requiring ``__init__.py`` in every intermediate directory.
    """
    abs_path = os.path.abspath(os.path.join(_PKG_DIR, *rel_path.split("/")))
    spec = importlib.util.spec_from_file_location(dotted_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# Also register the intermediate namespace so that
# ``from src.celestials.framework.planet import …`` resolves the dotted path.
for _ns in ("src.framework", "src.planets"):
    if _ns not in sys.modules:
        sys.modules[_ns] = types.ModuleType(_ns)

# ── Load sub-modules ─────────────────────────────────────────────────
_planet_mod = _load("src.framework.planet", "../framework/planet.py")
_mars_mod   = _load("src.planets.mars",     "planets/mars.py")

# ── Public re-exports ────────────────────────────────────────────────
# Framework
Planet            = _planet_mod.Planet
PlanetaryState    = _planet_mod.PlanetaryState
OrbitalParameters = _planet_mod.OrbitalParameters

# Mars
Mars                     = _mars_mod.Mars
MARS_MASS                = _mars_mod.MARS_MASS
MARS_RADIUS              = _mars_mod.MARS_RADIUS
MARS_GRAVITY             = _mars_mod.MARS_GRAVITY
MARS_ROTATION_PERIOD     = _mars_mod.MARS_ROTATION_PERIOD
MARS_SEMI_MAJOR_AXIS     = _mars_mod.MARS_SEMI_MAJOR_AXIS
MARS_ECCENTRICITY        = _mars_mod.MARS_ECCENTRICITY
MARS_ORBITAL_PERIOD      = _mars_mod.MARS_ORBITAL_PERIOD
MARS_AXIAL_TILT          = _mars_mod.MARS_AXIAL_TILT
MARS_DEFAULT_COMPOSITION = _mars_mod.MARS_DEFAULT_COMPOSITION
