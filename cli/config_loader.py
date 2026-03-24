"""Config loading — YAML → SimConfig.

Priority chain (highest → lowest):
    CLI flags  >  YAML file (--config)  >  built-in preset (--preset)  >  defaults
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml
from pydantic import ValidationError

from cli.models import RunFlags, SimConfig
from cli.presets import MARS_PRESET_NAMES

_CONFIGS_DIR = Path(__file__).parent / "configs"


# ── Internal helpers ──────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* onto a deep copy of *base*."""
    result = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _preset_path(preset: str) -> Path:
    if preset not in MARS_PRESET_NAMES:
        raise ValueError(
            f"Unknown preset '{preset}'.\nValid presets: {', '.join(MARS_PRESET_NAMES)}"
        )
    return _CONFIGS_DIR / f"{preset}.yaml"


# ── Public API ────────────────────────────────────────────────────────────────

def load(
    planet: str = "mars",
    preset: str | None = None,
    config: str | None = None,
) -> SimConfig:
    """Return a fully resolved, validated SimConfig.

    Resolution order:
      1. SimConfig defaults (defined in models.py).
      2. Merge built-in preset YAML (if --preset given).
      3. Merge user YAML (if --config given).
      4. Validate via Pydantic — raises ValueError on bad values.
    """
    data: dict = {}

    if preset is not None:
        with open(_preset_path(preset)) as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})

    if config is not None:
        with open(config) as f:
            data = _deep_merge(data, yaml.safe_load(f) or {})

    return SimConfig.model_validate(data)


def merge_overrides(base: SimConfig, flags: RunFlags) -> SimConfig:
    """Apply non-None CLI flags onto *base* and return a new SimConfig."""
    return flags.apply(base)


def validate(cfg: SimConfig) -> list[str]:
    """Return validation error strings (empty list = valid).

    Pydantic already validates on construction; this re-validates and
    returns human-readable messages for the CLI error display.
    """
    try:
        SimConfig.model_validate(cfg.model_dump())
        return []
    except ValidationError as exc:
        return [f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}" for e in exc.errors()]


# ── Kept for mars_config_validate which reads raw user YAML ───────────────────

_MARS_BASE: dict = SimConfig().model_dump()
