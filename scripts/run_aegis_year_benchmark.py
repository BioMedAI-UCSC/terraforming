#!/usr/bin/env python3
"""Launch a fresh year/decade rollout or summarize an existing completed run."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', 'outputs/mpl-cache')
ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--years', type=int, choices=[1, 10], default=1)
    parser.add_argument('--surface', type=Path, default=Path('data/tes/mgs_tes_surface_1deg.nc'))
    parser.add_argument('--dust', type=Path, default=Path('data/ames/fv3betaout1/ames_surface_reference.nc'))
    parser.add_argument('--analyze-only', action='store_true')
    parser.add_argument('--require-gpu', action='store_true')
    args = parser.parse_args()
    if not args.analyze_only:
        if args.run_dir.exists():
            parser.error('Use a fresh run directory; --analyze-only reads completed runs')
        if args.require_gpu:
            os.environ['JAX_PLATFORMS'] = 'cuda'
        command = [sys.executable, str(ROOT/'scripts/run_gcm3d_ablation.py'), str(args.surface),
            '--output-dir', str(args.run_dir), '--config', 'convection', '--truncation', 'T21',
            '--layers', '12', '--dt', '450', '--initial-ls', '0', '--diurnal', '--sols', str(668.6*args.years),
            '--chunk-sols', '10', '--performance-mode', '--precision', 'float64',
            '--physics-evaluation', 'stage', '--ames-dust-reference', str(args.dust)]
        started = time.perf_counter()
        result = subprocess.run(command)
        args.run_dir.mkdir(parents=True, exist_ok=True)
        (args.run_dir/'launcher.json').write_text(json.dumps({'command': command, 'returncode': result.returncode,
            'whole_process_seconds': time.perf_counter()-started}, indent=2))
        if result.returncode:
            return result.returncode
    import numpy as np
    import matplotlib.pyplot as plt
    manifest = json.loads((args.run_dir/'manifest.json').read_text())
    timing = manifest['runs']['convection']
    with (args.run_dir/'convection/checkpoint_diagnostics.csv').open() as f:
        data = list(csv.DictReader(f))
    sols = np.array([float(r['elapsed_sols']) for r in data])
    mass = np.array([float(r['atmosphere_plus_surface_reservoir_mass_kg']) for r in data])
    if len(data) < 2 or not np.isfinite([[float(v) for v in r.values()] for r in data]).all():
        raise ValueError('Insufficient or nonfinite diagnostics')
    if abs(sols[-1]-manifest['sols']) > manifest['dt_seconds']/88775.244*2:
        raise ValueError('Incomplete rollout; refusing completed-year performance claim')
    if timing.get('invocation_start_step', 0) != 0:
        raise ValueError('Resumed invocation timing cannot represent the complete rollout')
    seconds = timing['invocation_elapsed_seconds']
    summary = {'sols': float(sols[-1]), 'steps': timing['steps'], 'wall_seconds': seconds,
        'sols_per_hour': float(sols[-1]/seconds*3600), 'mars_years_per_day': float(sols[-1]/668.6/seconds*86400),
        'maximum_relative_co2_drift': float(np.max(np.abs(mass/mass[0]-1))),
        'timed_region': timing.get('timed_region'), 'ames_legacy_year_hours_approximate': 668.6/20,
        'ames_source': 'https://github.com/nasa/legacy-mars-global-climate-model#running-the-model',
        'limitations': 'Ames tutorial hardware unspecified, different resolution/physics; no matched speedup. Finite checkpoints do not establish equilibrium.'}
    (args.run_dir/'year_benchmark.json').write_text(json.dumps(summary, indent=2))
    with (args.run_dir/'year_benchmark.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary)); writer.writeheader(); writer.writerow(summary)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
    axes[0].bar(['AEGIS measured\nnormalized per Mars year', 'Ames Legacy tutorial\nextrapolated; hardware unknown'],
                [seconds/3600/(sols[-1]/668.6), 668.6/20], color=['#2962a3', '#999999'])
    axes[0].set_ylabel('Wall hours per Mars year'); axes[0].tick_params(axis='x', labelsize=8)
    axes[1].semilogy(sols, np.maximum(np.abs(mass/mass[0]-1), 1e-16))
    axes[1].set(xlabel='Elapsed sols', ylabel='Relative total CO₂ inventory drift')
    fig.suptitle('Throughput context and conservation (unmatched configurations)')
    fig.savefig(args.run_dir/'year_benchmark.pdf'); fig.savefig(args.run_dir/'year_benchmark.png', dpi=180)
    print(json.dumps(summary, indent=2))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
