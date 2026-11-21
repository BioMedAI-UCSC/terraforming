#!/usr/bin/env python3
"""Compare full air/wind fields at the annual runs' exact shared times."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import xarray as xr

from analyse_reviewer_timesteps import analyse as validate_campaign
from audit_paper_results import paired_timestep
from src.celestials.planets import mars
from src.framework.gcm import body, coordinates, specs
from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic
from src.framework.gcm.dynamics import reference_temperature
from src.framework.gcm.restart import load_restart


def checkpoint_fields(paths, seconds, coords, physics_specs, planet, hashes):
    grid = coords.horizontal
    velocity = float(physics_specs.dimensionalize(1., scales.units.meter / scales.units.second).magnitude)
    time_scale = 1. / float(physics_specs.nondimensionalize(1. * scales.units.second))
    reference = jnp.asarray(reference_temperature(coords, planet))[:, None, None]
    @jax.jit
    def convert(state):
        u, v = spherical_harmonic.vor_div_to_uv_nodal(grid, state.dynamics.vorticity, state.dynamics.divergence)
        return jnp.stack([grid.to_nodal(state.dynamics.temperature_variation) + reference, u * velocity, v * velocity])
    rows = []
    for path, second in zip(paths, seconds, strict=True):
        if str(path) not in hashes:
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        state = load_restart(path)
        clock = float(state.dynamics.sim_time) * time_scale
        if not np.isfinite(clock) or not np.isclose(clock, second, atol=.01, rtol=0):
            raise ValueError(f"Checkpoint clock differs from declared physical time: {path}")
        rows.append(np.asarray(convert(state)))
    values = np.stack(rows)
    return xr.Dataset({name: (("time", "sigma", "lat", "lon"), values[:, index].transpose(0, 1, 3, 2),
                              {"units": "K" if index == 0 else "m/s"})
        for index, name in enumerate(("air_temperature", "eastward_wind", "northward_wind"))},
        coords=dict(time=np.asarray(seconds) / planet.solar_day_seconds, sigma=np.asarray(coords.vertical.centers),
                    lat=np.degrees(grid.latitudes), lon=np.degrees(grid.longitudes)))


def analyse(studies, output):
    validate_campaign(studies)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Choose an empty output directory")
    jax.config.update("jax_enable_x64", True)
    configs, times = {}, {}
    for dt in (450, 300, 150):
        folder = studies / f"timestep-{dt}"
        config = json.loads((folder / "configuration.json").read_text())
        # Reject changed clocks/units before decoding dimensional restart fields.
        for relative, module in (("src/framework/gcm/body.py", body), ("src/framework/gcm/coordinates.py", coordinates),
                ("src/framework/gcm/specs.py", specs), ("src/celestials/planets/mars/__init__.py", mars)):
            if config["identity"]["source_sha256"]["package/" + relative] != hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest():
                raise ValueError(f"Installed decoder differs from archived source: {relative}")
        if config["jax"] != jax.__version__:
            raise ValueError("Use the archived JAX version for field decoding")
        configs[dt] = config
        rows = json.loads((folder / "diagnostics.json").read_text())
        times[dt] = {r["step"] * dt: r["step"] for r in rows}
    cfg = configs[450]["grid"]
    coords = coordinates.coordinate_system(cfg["truncation"], cfg["n_layers"])
    if not np.allclose(np.diff(coords.vertical.boundaries), 1. / cfg["n_layers"], rtol=0, atol=1e-7):
        raise ValueError("Equal-layer averaging requires uniformly spaced sigma boundaries")
    physics_specs = specs.physics_specs(mars.MARS_BODY_3D)
    output.mkdir(parents=True, exist_ok=True)
    hashes, comparisons = {}, []
    for first, second in itertools.combinations(times, 2):
        common = sorted(times[first].keys() & times[second].keys())
        datasets = []
        for dt in (first, second):
            folder = studies / f"timestep-{dt}"
            paths = [folder / ("initial.npz" if times[dt][t] == 0 else f"state-{times[dt][t]:09d}.npz") for t in common]
            datasets.append(checkpoint_fields(paths, common, coords, physics_specs, mars.MARS_BODY_3D, hashes))
        table, summary = paired_timestep(*datasets, first_dt=first, second_dt=second)
        summary["limitation"] = "Paired transient sensitivity of full 3-D air/u/v fields; Gaussian area and equal sigma-layer weights. Not observational error or equilibrium."
        table.to_csv(output / f"fields-{first}-{second}.csv", index=False)
        comparisons.append(summary)
        print(f"Full-field comparison {first}/{second}: {len(common)} exact shared samples", flush=True)
        del datasets
    scripts = (Path(__file__), Path(__file__).with_name("audit_paper_results.py"), Path(__file__).with_name("analyse_reviewer_timesteps.py"))
    result = dict(pairs=comparisons, checkpoint_sha256={str(Path(p).relative_to(studies)): h for p, h in hashes.items()},
        analysis_source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
        units=dict(air_temperature="K", eastward_wind="m/s", northward_wind="m/s"),
        jax=jax.__version__, backend=jax.default_backend())
    (output / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("studies", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyse(args.studies, args.output)
