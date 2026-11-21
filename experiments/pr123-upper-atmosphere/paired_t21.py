"""Run matched T21 baseline/treatment cases and exact restarts on one GPU."""
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import numpy as np

output = Path('/work/pr123/paired-t21-69d6587-43fc88b')
output.mkdir(exist_ok=True)
stop = threading.Event()


def monitor():
    with (output / 'gpu.csv').open('w') as stream:
        stream.write('timestamp,gpu,memory_used_mib,memory_total_mib,gpu_utilization_percent,memory_utilization_percent\n')
        while not stop.is_set():
            result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,memory.used,memory.total,utilization.gpu,utilization.memory',
                                     '--format=csv,noheader,nounits'], capture_output=True, text=True)
            stream.write(result.stdout or result.stderr)
            stream.flush()
            stop.wait(1.)


def case(name):
    source = Path('/scratch/treatment/source' if name.startswith('balanced') else '/scratch/source')
    env = dict(os.environ, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION='.4',
               PYTHONPATH=':'.join(str(source / path) for path in ('package', '.', 'scripts', 'apps/nautalis')))
    common = [sys.executable, str(source / 'scripts/run_gcm3d_ablation.py'), '/scratch/inputs',
              '--config', 'reference', '--mola-path', '/scratch/inputs/mars_topo_mola16.nc',
              '--truncation', 'T21', '--layers', '56', '--dt', '231', '--chunk-sols', '1',
              '--cooldown-seconds', '0', '--output-dir', str(output / name)]
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
    return dict(case=name, returncode=result.returncode, started_unix=started, ended_unix=time.time())


device = subprocess.check_output(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                                  '--format=csv'], text=True)
(output / 'device.csv').write_text(device)
thread = threading.Thread(target=monitor)
thread.start()
records = []
try:
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(case, name) for name in ('baseline', 'balanced', 'baseline-resumed', 'balanced-resumed')]
        for future in concurrent.futures.as_completed(futures):
            records.append(future.result())
            (output / 'summary.json').write_text(json.dumps(records, indent=2))
            print(json.dumps(records[-1]), flush=True)
    if any(row['returncode'] for row in records):
        raise SystemExit(1)
    for name in ('baseline', 'balanced'):
        with np.load(output / name / 'reference/restart.npz', allow_pickle=False) as full, \
                np.load(output / f'{name}-resumed/reference/restart.npz', allow_pickle=False) as resumed:
            assert set(full.files) == set(resumed.files)
            for key in full.files:
                np.testing.assert_array_equal(full[key], resumed[key], err_msg=f'{name}:{key}')
    (output / 'restart-exact.json').write_text(json.dumps(dict(status='pass', cases=['baseline', 'balanced'], steps=1920)))
finally:
    stop.set()
    thread.join()
