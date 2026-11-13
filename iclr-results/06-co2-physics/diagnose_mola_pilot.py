"""Sample extrema in all saved pilot checkpoints; no reference skill claim."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from src.celestials.planets.mars import MARS_BODY_3D as body
from src.framework.gcm._dinosaur import jax, scales, spherical_harmonic
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import reference_temperature
from src.framework.gcm.restart import load_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics.gcm import co2_frost_point_k

jax.config.update("jax_enable_x64", True)
root = Path(__file__).resolve().parent / "ames-physics-mola-ls270-300s/convection"
config = json.loads((root / "config.json").read_text())
with zipfile.ZipFile(root / "executed_source.zip") as archive:
    for filename, expected in config["source_sha256"].items():
        assert hashlib.sha256(archive.read(filename)).hexdigest() == expected
coords = coordinate_system(config["truncation"], config["layers"])
grid = coords.horizontal
specs = physics_specs(body)
pressure_scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
speed_scale = float(specs.dimensionalize(1., scales.units.meter / scales.units.second).magnitude)
sigma = np.asarray(coords.vertical.centers, dtype=float)
gamma = body.cp_j_kg_k / (body.cp_j_kg_k - body.gas_constant_j_kg_k)
samples = []
for path in sorted((root / "checkpoints").glob("restart_*.npz")):
    state = load_restart(path)
    u, v = (np.asarray(x) * speed_scale for x in spherical_harmonic.vor_div_to_uv_nodal(
        grid, state.dynamics.vorticity, state.dynamics.divergence))
    speed = np.hypot(u, v)
    temperature = np.asarray(grid.to_nodal(state.dynamics.temperature_variation)) + np.asarray(
        reference_temperature(coords, body))[:, None, None]
    pressure = np.exp(np.asarray(grid.to_nodal(state.dynamics.log_surface_pressure))) * pressure_scale * sigma[:, None, None]
    mach = speed / np.sqrt(gamma * body.gas_constant_j_kg_k * temperature)
    dx = 2 * np.pi * body.radius_m / len(grid.longitudes) * np.cos(np.asarray(grid.latitudes))
    dy = body.radius_m * np.gradient(np.asarray(grid.latitudes))
    cfl = config["dt"] * (np.abs(u) / dx[None, None, :] + np.abs(v) / dy[None, None, :])
    assert np.isfinite(mach).all() and np.isfinite(cfl).all()
    k, i, j = np.unravel_index(np.argmax(speed), speed.shape)
    samples.append(dict(restart=path.name, max_speed_ms=float(speed[k, i, j]),
        location=dict(layer=int(k), sigma=float(sigma[k]),
            latitude_deg=float(np.degrees(grid.latitudes[j])), longitude_deg=float(np.degrees(grid.longitudes[i])),
            pressure_pa=float(pressure[k, i, j]), temperature_k=float(temperature[k, i, j]),
            mach=float(mach[k, i, j]), horizontal_advective_cfl=float(cfl[k, i, j])),
        max_mach=float(mach.max()), max_horizontal_advective_cfl=float(cfl.max()),
        max_frost_deficit_k=float(np.maximum(np.asarray(co2_frost_point_k(pressure)) - temperature, 0.).max())))
rows = list(csv.DictReader((root / "checkpoint_diagnostics.csv").open()))
mass = np.array([float(row["total_co2_mass_kg"]) for row in rows])
report = dict(configuration=config, sampled_extrema=samples,
    max_post_projection_relative_total_co2_mass_drift=float(np.max(np.abs(mass / mass[0] - 1.))),
    limitations="Five-sol cold-start pilot, sampled at one-sol checkpoints. CFL is a horizontal advective diagnostic, not a bound for the full spectral IMEX scheme. No reference-height comparison, seasonal equilibrium, spectral or full momentum closure is established.")
(root / "extrema-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: value for key, value in report.items() if key != "configuration"}, indent=2))
