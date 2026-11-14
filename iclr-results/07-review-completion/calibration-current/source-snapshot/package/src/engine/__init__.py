"""Engine package – single public API surface.

Usage::

    from src.engine import Accuracy, TimeController, Snapshot
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types

_PKG_DIR = os.path.dirname(__file__)


def _load(dotted_name: str, rel_path: str) -> types.ModuleType:
    """Load a ``.py`` file and register it under *dotted_name* in ``sys.modules``."""
    abs_path = os.path.join(_PKG_DIR, *rel_path.split("/"))
    spec = importlib.util.spec_from_file_location(dotted_name, abs_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Load sub-modules ─────────────────────────────────────────────────
# Ensure celestials is initialised (it registers framework.planet in
# sys.modules, which time_controller.py needs).
import src.celestials  # noqa: F401

_tc_mod  = _load("src.engine.time_controller",   "time_controller.py")
_btc_mod = _load("src.engine.batched_controller", "batched_controller.py")

# ── Public re-exports ────────────────────────────────────────────────
Accuracy              = _tc_mod.Accuracy
TimeController        = _tc_mod.TimeController
Snapshot              = _tc_mod.Snapshot
BatchedMars           = _btc_mod.BatchedMars
BatchedTimeController = _btc_mod.BatchedTimeController
