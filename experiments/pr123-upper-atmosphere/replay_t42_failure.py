"""Replay the verified finite T42 step20 through public build_step once."""
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import shutil

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
os.environ['JAX_ENABLE_X64'] = 'true'
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm.restart import load_restart
from src.framework.gcm.dynamics import integrate
from src.framework.gcm.comparison_averaging import averaging_step, fields
from src.framework.physics.dinosaur_physics import build_step
from src.celestials.planets.mars.topography import regrid_to_nodal, mola_modal_orography
from run_gcm3d_ablation import reference_configuration, mean_anomaly_for_ls

source = Path('/scratch/source')
data = Path('/scratch/inputs')
root = Path('/work/pr123/69d6587cef02e5b17a82e28422b67aa3d5be55ad/t42-exact-cpu/T42/reference')
output = root.parents[2] / 't42-step20-replay'
output.mkdir(exist_ok=True)
config = json.loads((root / 'config.json').read_text())
assert config['truncation'] == 'T42' and config['layers'] == 56 and config['dt'] == 231.
for name, expected in config['source_sha256'].items():
    assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
for name, expected in config['reference_input_sha256'].items():
    path = data / ('mars_topo_mola16.nc' if name == 'mola' else name)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
paths = [root / f'checkpoints/restart_{step:09d}.npz' for step in (19, 20)]
earlier, last = map(load_restart, paths)
assert all(np.isfinite(np.asarray(value)).all() for value in jax.tree.leaves(last))
coords = coordinate_system('T42', 56, sigma_boundaries=np.asarray(last.auxiliary_fields['vertical_sigma_boundaries']))
grid = coords.horizontal
body, forcing, radiation, options, _ = reference_configuration(grid, data, config['dt'])
specs = physics_specs(body)
forcing = dataclasses.replace(forcing, init_orbital_angle_rad=mean_anomaly_for_ls(np.deg2rad(config['initial_ls']), forcing))
options = {key: value for key, value in options.items() if not key.startswith('initial_')}
options.update(config.get('co2_process_switches', {}))
options.update(config.get('surface_process_switches', {}))
options.update(config.get('reference_soil_controls', {}))
options.update(config.get('reference_process_controls', {}))
elevation = regrid_to_nodal(coords, mola_path=data / 'mars_topo_mola16.nc')
orography = mola_modal_orography(coords, specs, elevation_nodal_m=elevation)
step = build_step(coords, specs, body, forcing, config['dt'], radiation_component=radiation,
                  orography=orography, hyperdiffusion_tau_seconds=config['hyperdiffusion_tau_sols'] * body.solar_day_seconds, **options)
earlier, step = averaging_step(step, earlier, coords, specs, body, 1920, config['dt'])
advance = jax.jit(lambda value: integrate(step, value, 1))
replayed = advance(earlier)
jax.block_until_ready(replayed)
differences = {}
for (path, actual), (_, expected) in zip(jax.tree_util.tree_flatten_with_path(replayed)[0],
                                       jax.tree_util.tree_flatten_with_path(last)[0], strict=True):
    actual, expected = np.asarray(actual), np.asarray(expected)
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10, err_msg=jax.tree_util.keystr(path))
    differences[jax.tree_util.keystr(path)] = float(np.max(np.abs(actual - expected)))
candidate = advance(last)
jax.block_until_ready(candidate)


def location(shape, index):
    result = dict(index=list(map(int, index)))
    if len(shape) >= 2 and tuple(shape[-2:]) == tuple(grid.nodal_shape):
        result.update(longitude_deg=float(np.degrees(grid.longitudes[index[-2]])),
                      latitude_deg=float(np.degrees(grid.latitudes[index[-1]])))
    if len(shape) == 3 and shape[0] == config['layers']:
        result.update(layer=int(index[0]), sigma=float(coords.vertical.centers[index[0]]))
    if len(shape) >= 2 and tuple(shape[-2:]) == tuple(grid.modal_shape):
        result['horizontal_coordinates'] = 'spectral coefficient indices; no unique physical longitude/latitude'
    return result


invalid = []
for path, value in jax.tree_util.tree_flatten_with_path(candidate)[0]:
    value = np.asarray(value)
    mask = ~np.isfinite(value)
    if np.any(mask):
        index = np.argwhere(mask)[0]
        invalid.append(dict(field=jax.tree_util.keystr(path), dtype=str(value.dtype), shape=list(value.shape),
                            nonfinite_count=int(mask.sum()), first_nonfinite=location(value.shape, index)))
assert invalid, 'Step21 did not reproduce the recorded non-finite failure'
dimensional = {name: np.asarray(value) for name, value in fields(last, coords, specs, body).items()}
rows = list(csv.DictReader((root / 'checkpoint_diagnostics.csv').open()))
recorded = next(row for row in rows if int(row['step']) == 20)
np.testing.assert_allclose(dimensional['wind_speed'].max(), float(recorded['max_wind_speed_ms']), rtol=1e-12)
np.testing.assert_allclose(dimensional['air_temperature'].min(), float(recorded['min_air_temperature_k']), rtol=1e-12)
summary = dict(status='reproduced', source_revision='69d6587cef02e5b17a82e28422b67aa3d5be55ad',
    last_finite_step=20, first_nonfinite_step=21, dt_seconds=231., adjacent_step19_to20_matches=True,
    adjacent_maximum_absolute_differences=differences, invalid_fields=invalid,
    last_finite_min_temperature_k=float(dimensional['air_temperature'].min()),
    last_finite_min_temperature_location=location(dimensional['air_temperature'].shape, np.unravel_index(np.argmin(dimensional['air_temperature']), dimensional['air_temperature'].shape)),
    last_finite_max_wind_m_s=float(dimensional['wind_speed'].max()),
    last_finite_max_wind_location=location(dimensional['wind_speed'].shape, np.unravel_index(np.argmax(dimensional['wind_speed']), dimensional['wind_speed'].shape)),
    checkpoint_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    source_sha256=config['source_sha256'], input_sha256=config['reference_input_sha256'],
    replay_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cpu_affinity=sorted(os.sched_getaffinity(0)), cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text(),
    limitations=['First reported non-finite field follows PyTree order; it does not identify the first internal kernel that became invalid.',
                 'Spectral coefficients have global support, so a bad modal index does not locate the physical cause.'])
shutil.copyfile(paths[-1], output / 'last-finite.npz')
(output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
print(json.dumps(dict(status=summary['status'], first_nonfinite_step=21, invalid_fields=len(invalid))), flush=True)
