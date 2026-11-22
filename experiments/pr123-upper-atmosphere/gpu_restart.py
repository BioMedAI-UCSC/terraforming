"""Compare repeated GPU execution and save/load from a common T21 checkpoint."""
import concurrent.futures
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
os.environ.update(JAX_ENABLE_X64='true', JAX_PLATFORMS='cuda',
                  XLA_PYTHON_CLIENT_PREALLOCATE='false', XLA_PYTHON_CLIENT_MEM_FRACTION='.45')
revision = '09e8c10500072fa06b51596bc50350d535ac6c5f'
output = Path('/work/pr123') / revision / 'gpu-common-checkpoint-v4'
output.mkdir(exist_ok=True)


def worker(name):
    import numpy as np
    from src.framework.gcm._dinosaur import jax
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm.restart import load_restart, save_restart
    from src.framework.gcm.dynamics import integrate
    from src.framework.gcm.comparison_averaging import averaging_step
    from src.framework.physics.dinosaur_physics import build_step
    from src.celestials.planets.mars.topography import regrid_to_nodal, mola_modal_orography
    from run_gcm3d_ablation import reference_configuration, mean_anomaly_for_ls, reference_divergence_diffusivity

    source, data = Path('/scratch/source'), Path('/scratch/inputs')
    root = output.parent / 'd2-T21' / name / 'reference'
    target = output / name
    target.mkdir(exist_ok=True)
    config = json.loads((root / 'config.json').read_text())
    assert config['truncation'] == 'T21' and config['layers'] == 56 and config['dt'] == 231.
    for filename, expected in config['source_sha256'].items():
        assert hashlib.sha256((source / filename).read_bytes()).hexdigest() == expected, filename
    for filename, expected in config['reference_input_sha256'].items():
        path = data / ('mars_topo_mola16.nc' if filename == 'mola' else filename)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename
    checkpoint = root / 'checkpoints/restart_000000384.npz'
    state = load_restart(checkpoint)

    def compare(left, right):
        assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
        count = 0
        for (path, a), (_, b) in zip(jax.tree_util.tree_flatten_with_path(left)[0],
                                     jax.tree_util.tree_flatten_with_path(right)[0], strict=True):
            a, b = np.asarray(a), np.asarray(b)
            assert a.dtype == b.dtype and a.shape == b.shape
            assert np.isfinite(a).all() and np.isfinite(b).all()
            np.testing.assert_array_equal(a, b, err_msg=jax.tree_util.keystr(path))
            count += 1
        return count

    # A changed public state must fail the same exact comparator used below.
    changed = state._replace(surface_temperature=state.surface_temperature.at[0, 0, 0].add(1.))
    try:
        compare(state, changed)
    except AssertionError:
        pass
    else:
        raise AssertionError('Negative control did not detect a one-kelvin change')
    save_restart(state, target / 'input-roundtrip.npz')
    restored = load_restart(target / 'input-roundtrip.npz')
    input_leaves = compare(state, restored)
    coords = coordinate_system('T21', 56, sigma_boundaries=np.asarray(state.auxiliary_fields['vertical_sigma_boundaries']))
    body, forcing, radiation, options, _ = reference_configuration(coords.horizontal, data, config['dt'])
    specs = physics_specs(body)
    forcing = dataclasses.replace(forcing, init_orbital_angle_rad=mean_anomaly_for_ls(np.deg2rad(config['initial_ls']), forcing))
    options = {key: value for key, value in options.items() if not key.startswith('initial_')}
    for key in ('co2_process_switches', 'surface_process_switches', 'reference_soil_controls', 'reference_process_controls'):
        options.update(config.get(key, {}))
    profile, metadata = reference_divergence_diffusivity('T21', 56)
    assert metadata == config['native_top_divergence_damping']
    options['divergence_diffusivity_m2_s'] = profile
    elevation = regrid_to_nodal(coords, mola_path=data / 'mars_topo_mola16.nc')
    orography = mola_modal_orography(coords, specs, elevation_nodal_m=elevation)
    step = build_step(coords, specs, body, forcing, config['dt'], radiation_component=radiation,
        orography=orography, hyperdiffusion_tau_seconds=config['hyperdiffusion_tau_sols'] * body.solar_day_seconds, **options)
    # The checkpoint already has averaging state; keep its persisted counter dtype.
    _, step = averaging_step(step, state, coords, specs, body, 1920, config['dt'])
    compare(state, restored)
    started = time.time()
    advance = jax.jit(lambda value: integrate(step, value, 384)).lower(state).compile()
    first = jax.block_until_ready(advance(state))
    repeated = jax.block_until_ready(advance(state))
    reloaded = jax.block_until_ready(advance(restored))
    repeat_leaves, reload_leaves = compare(first, repeated), compare(first, reloaded)
    save_restart(first, target / 'step768.npz')
    mid = load_restart(target / 'step768.npz')
    mid_leaves = compare(first, mid)
    continued = jax.block_until_ready(advance(first))
    continued_loaded = jax.block_until_ready(advance(mid))
    continuation_leaves = compare(continued, continued_loaded)
    save_restart(continued, target / 'step1152.npz')
    result = dict(status='pass', source_revision=revision, case=name, checkpoint_step=384,
        steps_per_compiled_call=384, final_step=1152, compilation_count=1,
        same_input_repeat_exact_leaves=repeat_leaves, input_roundtrip_exact_leaves=input_leaves,
        reloaded_input_advance_exact_leaves=reload_leaves, midpoint_roundtrip_exact_leaves=mid_leaves,
        reloaded_midpoint_continuation_exact_leaves=continuation_leaves, negative_control='one-kelvin change detected',
        elapsed_compile_and_five_calls_seconds=time.time()-started,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        source_sha256=config['source_sha256'], input_sha256=config['reference_input_sha256'],
        initial_pressure_scheme=config['initial_pressure_scheme'], native_top_divergence_damping=metadata,
        device=str(jax.devices()[0]), cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text(),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        limitations=['One compiled executable and one common saved state per case; does not establish equality across independent cold initializations or compilations.'])
    (target / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print(json.dumps(result), flush=True)


if len(sys.argv) == 2:
    worker(sys.argv[1])
else:
    stop = threading.Event()

    def monitor():
        with (output / 'gpu.csv').open('w') as gpu, (output / 'resources.jsonl').open('w') as resources:
            while not stop.is_set():
                result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu',
                                         '--format=csv,noheader,nounits'], capture_output=True, text=True)
                gpu.write(result.stdout or result.stderr)
                gpu.flush()
                resources.write(json.dumps(dict(unix_time=time.time(), cgroup={key: Path('/sys/fs/cgroup', key).read_text()
                    for key in ('memory.current', 'memory.peak', 'cpu.stat')}))+'\n')
                resources.flush()
                stop.wait(1.)

    def run(name):
        with (output / f'{name}.log').open('w') as log:
            result = subprocess.run([sys.executable, __file__, name], stdout=log, stderr=subprocess.STDOUT)
        return dict(case=name, returncode=result.returncode)

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(run, ('native-d2', 'balanced-d2')))
        (output / 'summary.json').write_text(json.dumps(results, indent=2))
        assert all(row['returncode'] == 0 for row in results), results
    finally:
        stop.set()
        thread.join()
