"""Dependency-boundary tests for the framework/celestial GCM split."""

import importlib.util
import inspect
import ast

from src.celestials.planets.mars import gcm as mars_gcm
from src.framework.gcm.body import BodyConstants
from src.framework.physics import gcm as framework_physics


def test_framework_physics_does_not_import_celestials():
    tree = ast.parse(inspect.getsource(framework_physics))
    imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    ]
    imports.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(name.startswith("src.celestials") for name in imports)


def test_removed_legacy_modules_do_not_return():
    assert importlib.util.find_spec("src.gcm3d") is None


def test_mars_owns_planet_specific_forcing_construction():
    forcing = mars_gcm.radiative_forcing()
    assert isinstance(forcing, framework_physics.RadiativeForcing)
    assert isinstance(mars_gcm.mars.MARS_BODY_3D, BodyConstants)
    assert forcing.rotation_period_s == float(mars_gcm.mars.MARS_ROTATION_PERIOD)
