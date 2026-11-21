"""Run source-pinned native-D2 controls; arguments: T21, or T42 and one case."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

allowed = sorted(os.sched_getaffinity(0))
os.sched_setaffinity(0, allowed[:8])

import numpy as np

revision = '09e8c10500072fa06b51596bc50350d535ac6c5f'
resolution = sys.argv[1]
assert resolution in ('T21', 'T42')
if resolution == 'T21':
    assert len(sys.argv) == 2
    names = ('native-d2', 'balanced-d2', 'native-d2-resumed', 'balanced-d2-resumed')
else:
    assert len(sys.argv) == 3 and sys.argv[2] in ('native-d2', 'balanced-d2')
    names = (sys.argv[2],)
output = Path('/work/pr123') / revision / ('d2-' + '-'.join(sys.argv[1:]))
output.mkdir(exist_ok=True)
source = Path('/scratch/source')
stop = threading.Event()
(output / 'launcher.json').write_text(json.dumps(dict(source_revision=revision, resolution=resolution,
    cases=names, concurrency=2 if resolution == 'T21' else 1, original_affinity=allowed,
    cpu_affinity=sorted(os.sched_getaffinity(0)), os_cpu_count=os.cpu_count(),
    cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text()), indent=2))
(output / 'device.csv').write_text(subprocess.check_output(['nvidia-smi',
    '--query-gpu=name,uuid,driver_version,memory.total', '--format=csv'], text=True))


def monitor():
    with (output / 'gpu.csv').open('w') as gpu, (output / 'resources.jsonl').open('w') as resources:
        gpu.write('timestamp,gpu,memory_used_mib,memory_total_mib,gpu_utilization_percent,memory_utilization_percent\n')
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            gpu.write(result.stdout or result.stderr)
            gpu.flush()
            resources.write(json.dumps(dict(unix_time=time.time(), cgroup={key: Path('/sys/fs/cgroup', key).read_text()
                for key in ('memory.current', 'memory.peak', 'memory.stat', 'cpu.stat')})) + '\n')
            resources.flush()
            stop.wait(1.)


def case(name):
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION='.4' if resolution == 'T21' else '.85')
    common = [sys.executable, str(source / 'scripts/run_gcm3d_ablation.py'), '/scratch/inputs',
              '--config', 'reference', '--native-top-divergence-damping',
              '--mola-path', '/scratch/inputs/mars_topo_mola16.nc', '--truncation', resolution,
              '--layers', '56', '--dt', '231', '--chunk-sols', '1', '--cooldown-seconds', '0',
              '--output-dir', str(output / name)]
    if name.startswith('balanced'):
        common += ['--initial-pressure-scheme', 'sigma-balanced']
    commands = ([common + ['--sols', '1'], common + ['--sols', '5', '--resume']]
                if name.endswith('resumed') else [common + ['--sols', '5']])
    started = time.time()
    with (output / f'{name}.log').open('w') as log:
        for command in commands:
            result = subprocess.run(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                break
    return dict(case=name, returncode=result.returncode, started_unix=started, ended_unix=time.time(), commands=commands)


thread = threading.Thread(target=monitor)
thread.start()
records = []
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 if resolution == 'T21' else 1) as pool:
        futures = [pool.submit(case, name) for name in names]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            (output / 'summary.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(records[-1]), flush=True)
    if any(row['returncode'] for row in records):
        raise SystemExit(1)
    if resolution == 'T21':
        for name in ('native-d2', 'balanced-d2'):
            with np.load(output / name / 'reference/restart.npz', allow_pickle=False) as full, \
                    np.load(output / f'{name}-resumed/reference/restart.npz', allow_pickle=False) as resumed:
                assert set(full.files) == set(resumed.files)
                for key in full.files:
                    np.testing.assert_array_equal(full[key], resumed[key], err_msg=f'{name}:{key}')
        (output / 'restart-exact.json').write_text(json.dumps(dict(status='pass', cases=['native-d2', 'balanced-d2'], steps=1920)))
finally:
    stop.set()
    thread.join()
