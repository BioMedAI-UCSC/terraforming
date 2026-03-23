"""Config loading, validation, and CLI-override merging.

Priority chain (highest → lowest):
    CLI flag  >  YAML file (--config)  >  built-in preset (--preset)  >  defaults
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from cli.presets import MARS_PRESET_NAMES

# Configs live alongside this file: cli/configs/
_CONFIGS_DIR = Path(__file__).parent / "configs"

# ── Default base config (mirrors Mars() constructor defaults) ─────────────────

_MARS_BASE: dict = {
    "preset": {
        "name": "custom",
        "description": "Custom configuration",
    },
    "planet": {
        "surface_temperature": 210.0,   # K
        "surface_pressure":    610.0,   # Pa
        "albedo":              0.25,
        "greenhouse_factor":   1.02,
        "ice_mass":            5.0e15,  # kg
        "latitude":            22.0,    # degrees N
        "longitude":           0.0,     # degrees E
        "elevation_m":         0.0,     # metres
        "initial_ls_deg":      251.0,   # solar longitude
        "composition":         None,    # None = use Mars defaults
    },
    "engine": {
        "dt":       3600.0,   # seconds
        "accuracy": "fast",   # "fast" | "accurate"
    },
    "experiment": {
        "type": "sol",    # "sol" | "year" | "multi"
        "sols": 1.0,
    },
    "output": {
        "save_csv":    True,
        "save_plot":   True,
        "ground_truth": False,
        "out_dir":     None,   # None = auto UTC timestamp (subfolder under outputs/)
        "output_path": None,   # full custom path; overrides out_dir when set
    },
}

_BASE_BY_PLANET = {"mars": _MARS_BASE}

# ── CLI kwarg → config path mapping ──────────────────────────────────────────

_CLI_TO_CONFIG: dict[str, tuple[str, str]] = {
    "surface_temp":      ("planet", "surface_temperature"),
    "pressure":          ("planet", "surface_pressure"),
    "albedo":            ("planet", "albedo"),
    "greenhouse_factor": ("planet", "greenhouse_factor"),
    "ice_mass":          ("planet", "ice_mass"),
    "lat":               ("planet", "latitude"),
    "lon":               ("planet", "longitude"),
    "elevation":         ("planet", "elevation_m"),
    "ls":                ("planet", "initial_ls_deg"),
    "accuracy":          ("engine", "accuracy"),
    "dt":                ("engine", "dt"),
    "exp_type":          ("experiment", "type"),
    "sols":              ("experiment", "sols"),
    "ground":            ("output", "ground_truth"),
    "output_dir":        ("output", "output_path"),
}

# Inverted flags: CLI flag set → config key becomes False
_INVERTED_FLAGS: dict[str, tuple[str, str]] = {
    "no_save": ("output", "save_csv"),
    "no_plot": ("output", "save_plot"),
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* onto a deep copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _coerce_numeric(cfg: dict) -> dict:
    """Cast YAML-loaded values to correct Python numeric types.

    PyYAML can read scientific notation like ``5.0e15`` as a string in some
    contexts.  This normalises all numeric config fields before use.
    """
    cfg = copy.deepcopy(cfg)
    for key in ("surface_temperature", "surface_pressure", "albedo",
                "greenhouse_factor", "ice_mass", "latitude", "longitude",
                "elevation_m", "initial_ls_deg"):
        v = cfg.get("planet", {}).get(key)
        if v is not None:
            cfg["planet"][key] = float(v)
    v = cfg.get("engine", {}).get("dt")
    if v is not None:
        cfg["engine"]["dt"] = float(v)
    v = cfg.get("experiment", {}).get("sols")
    if v is not None:
        cfg["experiment"]["sols"] = float(v)
    return cfg


def resolve_preset_path(planet: str, preset: str) -> Path:
    """Return absolute path to a built-in preset YAML."""
    names = MARS_PRESET_NAMES if planet == "mars" else []
    if preset not in names:
        raise ValueError(
            f"Unknown preset '{preset}' for planet '{planet}'.\n"
            f"Valid presets: {', '.join(names)}"
        )
    return _CONFIGS_DIR / f"{preset}.yaml"


def load(
    planet: str = "mars",
    preset: str | None = None,
    config: str | None = None,
) -> dict:
    """Resolve the full config dict.

    Resolution order:
      1. Planet default base.
      2. Merge built-in preset YAML (if --preset given).
      3. Merge user YAML (if --config given).
      4. Type-coerce all numeric fields.
    """
    result = copy.deepcopy(_BASE_BY_PLANET[planet])

    if preset is not None:
        path = resolve_preset_path(planet, preset)
        with open(path) as f:
            result = _deep_merge(result, yaml.safe_load(f) or {})

    if config is not None:
        with open(config) as f:
            result = _deep_merge(result, yaml.safe_load(f) or {})

    return _coerce_numeric(result)


def merge_overrides(cfg: dict, cli_kwargs: dict[str, Any]) -> dict:
    """Apply non-None CLI flags onto the resolved config.

    Only explicitly provided flags (non-None) participate in the merge,
    so preset/config values are preserved for anything the user omitted.
    """
    result = copy.deepcopy(cfg)

    for cli_key, (section, key) in _CLI_TO_CONFIG.items():
        val = cli_kwargs.get(cli_key)
        if val is not None:
            result[section][key] = val

    for cli_key, (section, key) in _INVERTED_FLAGS.items():
        if cli_kwargs.get(cli_key):
            result[section][key] = False

    name = cli_kwargs.get("name")
    if name is not None:
        result["output"]["out_dir"] = name

    return _coerce_numeric(result)


def validate(cfg: dict) -> list[str]:
    """Return a list of validation error strings (empty = valid)."""
    cfg = _coerce_numeric(cfg)
    errors: list[str] = []
    p, e, x = cfg.get("planet", {}), cfg.get("engine", {}), cfg.get("experiment", {})

    if (v := p.get("albedo")) is not None and not 0.0 <= v <= 1.0:
        errors.append(f"planet.albedo must be 0–1, got {v}")
    if (v := p.get("greenhouse_factor")) is not None and v < 1.0:
        errors.append(f"planet.greenhouse_factor must be ≥ 1.0, got {v}")
    if (v := p.get("surface_temperature")) is not None and v <= 0:
        errors.append(f"planet.surface_temperature must be > 0 K, got {v}")
    if (v := p.get("surface_pressure")) is not None and v <= 0:
        errors.append(f"planet.surface_pressure must be > 0 Pa, got {v}")
    if (v := p.get("ice_mass")) is not None and v < 0:
        errors.append(f"planet.ice_mass must be ≥ 0 kg, got {v}")
    if (v := p.get("latitude")) is not None and not -90.0 <= v <= 90.0:
        errors.append(f"planet.latitude must be -90 to 90°, got {v}")
    if (v := e.get("dt")) is not None and v <= 0:
        errors.append(f"engine.dt must be > 0 s, got {v}")
    if (v := e.get("accuracy")) is not None and v not in ("fast", "accurate"):
        errors.append(f"engine.accuracy must be 'fast' or 'accurate', got '{v}'")
    if (v := x.get("type")) is not None and v not in ("sol", "year", "multi"):
        errors.append(f"experiment.type must be 'sol', 'year', or 'multi', got '{v}'")
    if (v := x.get("sols")) is not None and v <= 0:
        errors.append(f"experiment.sols must be > 0, got {v}")

    return errors
