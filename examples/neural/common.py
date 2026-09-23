"""Data/report helpers for the two generated-data examples, not framework APIs."""
import dataclasses
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from src.framework.gcm._dinosaur import jax, jnp
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import radiative_forcing
from src.framework.neural import ColumnInputs


def configuration(layers=4, ames=False):
    return tuple(np.linspace(0, 1, layers+1)), dataclasses.replace(
        radiative_forcing(), co2_radiation_enabled=True, ames_correlated_k_enabled=ames)


def generate_columns(seed, count, layers, *, extrapolation=False):
    """Independent profiles; entire profiles belong to a single split.

    Training/interpolation visible dust is [0,.8]; extrapolation is [1,1.5].
    Other ranges: air 170–250 K with smooth profiles, surface 185–265 K,
    pressure 450–850 Pa, cosine zenith 0–1 plus an explicit dark subset.
    """
    rng = np.random.default_rng(seed)
    sigma = np.linspace(.5/layers, 1-.5/layers, layers)[:, None, None]
    base = rng.uniform(190, 230, (count, 1))
    slope = rng.uniform(-20, 20, (count, 1))
    air = base[None] + slope[None]*(sigma-.5) + rng.normal(0, 2, (layers, count, 1))
    cosine = rng.uniform(.05, 1, (count, 1))
    cosine[::4] = 0
    visible = rng.uniform(1, 1.5, (count, 1)) if extrapolation else rng.uniform(0, .8, (count, 1))
    return ColumnInputs(*map(jnp.asarray, (
        air, rng.uniform(185, 265, (count, 1)), rng.uniform(450, 850, (count, 1)),
        visible, visible / 3, rng.uniform(20, 45, (count, 1)),
        590*cosine, 1/np.clip(cosine, .05, 1), rng.uniform(.15, .35, (count, 1)),
        rng.uniform(.9, 1, (count, 1)),
    )))


def synchronized_call(fn, *args):
    start = time.perf_counter()
    result = fn(*args)
    jax.block_until_ready(result)
    return result, time.perf_counter() - start


def write_report(out, report, title):
    """Strict JSON, preserving failures as null rather than silently dropping them."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    def clean(x):
        if isinstance(x, dict):
            return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, (float, np.floating)) and not np.isfinite(x):
            return None
        return x

    report = clean(report)
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(
        f"# {title}\n\nGenerated reference data; no observational Mars accuracy claim.\n\n"
        "Nonfinite metrics are null; failure counts are retained.\n\n```json\n"
        + json.dumps(report, indent=2) + "\n```\n")


def provenance(args):
    import platform
    import subprocess
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    root = Path(__file__).resolve().parents[2]
    sources = [*Path(__file__).parent.glob("*.py"),
               *(root / "package/src/framework/neural").glob("*.py"),
               root / "package/src/framework/physics/gcm.py",
               root / "package/src/framework/gcm/learning.py"]
    return {"arguments": vars(args), "python": platform.python_version(), "jax": jax.__version__,
            "numpy": np.__version__,
            "devices": [str(d) for d in jax.devices()], "x64": bool(jax.config.jax_enable_x64),
            "git_revision": revision, "working_tree_dirty": dirty,
            "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
            "reference": "generated using the framework radiation kernel", "seed": args.seed}
