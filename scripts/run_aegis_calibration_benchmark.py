#!/usr/bin/env python3
"""Paired autodiff/Powell recovery benchmark; optional external simulator adapter.

An external command receives REQUEST.json RESPONSE.npz as its final arguments.
It must return `fields` in Model.observe units/shape (time, T/u/v, level, lon, lat).
The request supplies physical parameters, times, dt, inputs and identical restart.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', 'outputs/mpl-cache')
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'package'), str(ROOT / 'apps/mars-calibration')]
import numpy as np
from mars_calibration import paper as p
from mars_calibration import driver as d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT / 'apps/mars-calibration/paper-experiment.json')
    parser.add_argument('--require-gpu', action='store_true')
    parser.add_argument('--external-command', help='Executable and arguments, parsed without a shell')
    parser.add_argument('--external-name', default='External simulator')
    parser.add_argument('--external-timeout', type=float, default=3600)
    parser.add_argument('--loss-threshold', type=float, default=1e-6)
    args = parser.parse_args()
    if args.loss_threshold <= 0 or args.external_timeout <= 0:
        parser.error('threshold and timeout must be positive')
    config = p.validate_config(json.loads(args.config.read_text()))
    manifest = p.validate_inputs(args.inputs)
    d.jax.config.update('jax_enable_x64', True)
    if args.require_gpu and d.jax.default_backend() != 'gpu':
        raise RuntimeError('GPU required')
    os.environ['MOLA_PATH'] = str((args.inputs / 'mola.img').resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    p.dump(args.output_dir / 'config.json', config)
    p.dump(args.output_dir / 'inputs.json', manifest)
    p.dump(args.output_dir / 'environment.json', p.environment())
    model = p.Model(args.inputs)
    def emit(row):
        print(json.dumps(row), flush=True)
        with (args.output_dir / 'progress.jsonl').open('a') as f:
            f.write(json.dumps(row) + '\n')
    report = p.recovery(model, config, args.output_dir, emit)
    results = report['results']
    if args.external_command:
        target = np.load(args.output_dir / 'synthetic-targets.npz')
        ntrain = len(config['train_seconds'])
        counter = 0
        for active in config['active_sets']:
            for start_id, physical_start in enumerate(config['starts']):
                def external(x):
                    nonlocal counter
                    counter += 1
                    physical = np.array(config['truth'], dtype=float)
                    physical[active] = d.LOWER_BOUNDS[active] + np.asarray(x) * (d.UPPER_BOUNDS-d.LOWER_BOUNDS)[active]
                    request = args.output_dir / f'external-{counter:05d}.json'
                    response = request.with_suffix('.npz')
                    p.dump(request, {'parameters': physical.tolist(), 'parameter_names': list(d.PARAMETER_NAMES),
                        'seconds': config['train_seconds'], 'dt_seconds': config['dt_seconds'],
                        'inputs': str(args.inputs.resolve()), 'restart': str((args.inputs/'restart.npz').resolve()),
                        'fields_contract': 'time, [temperature K, u m/s, v m/s], level, lon, lat; T21/L12'})
                    subprocess.run(shlex.split(args.external_command) + [str(request.resolve()), str(response.resolve())],
                                   check=True, timeout=args.external_timeout)
                    with np.load(response) as data:
                        fields = data['fields']
                    truth = target['fields'][:ntrain]
                    if fields.shape != truth.shape or not np.isfinite(fields).all():
                        raise ValueError('External fields have incorrect shape or nonfinite values')
                    return model.loss(d.jnp.asarray(fields), d.jnp.asarray(truth), d.jnp.asarray(target['normalization']))
                start = (np.asarray(physical_start)[active]-d.LOWER_BOUNDS[active])/(d.UPPER_BOUNDS-d.LOWER_BOUNDS)[active]
                result = p.optimize(external, None, start, 'Powell', config['max_evaluations'], emit)
                results.append({**result, 'active': active, 'start_id': start_id, 'method': args.external_name + ' Powell',
                                'forward_preparation': {}, 'gradient_preparation': {}})
    rows = []
    for r in results:
        hit = next((v for v in r['trace'] if v['loss'] <= args.loss_threshold), None)
        prep = r['gradient_preparation'] if r['method'] == 'L-BFGS-B' else r['forward_preparation']
        rows.append({'active': str(r['active']), 'start_id': r['start_id'], 'method': r['method'],
                     'threshold': args.loss_threshold, 'reached': hit is not None,
                     'seconds_to_threshold': hit['cumulative_wall_seconds'] if hit else None,
                     'total_optimization_seconds': r['wall_seconds'], 'evaluations': r['evaluations'],
                     'best_loss': r['best']['loss'], 'preparation_seconds': sum(prep.values())})
    p.write_csv(args.output_dir / 'time-to-target.csv', rows)
    p.dump(args.output_dir / 'benchmark.json', {'results': rows, 'external_measured': bool(args.external_command),
        'aegis_scientific_acceptance_pass': report['scientific_acceptance_pass'],
        'limitations': ['Synthetic identical-model targets favor AEGIS; external model mismatch can prevent threshold attainment.',
                       'Unreached thresholds are censored, not speedup measurements.',
                       'Default Powell uses AEGIS, not Ames. External timings include adapter process and IO.',
                       'Single- and three-scalar experiments do not establish vector-field scaling.']})
    p.plot_recovery(args.output_dir, results)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    labels = [f"{r['method']}\n{r['active']} / start {r['start_id']}" for r in rows]
    ax.bar(range(len(rows)), [r['seconds_to_threshold'] if r['reached'] else r['total_optimization_seconds'] for r in rows],
           color=['#2962a3' if r['reached'] else '#999999' for r in rows])
    ax.set_xticks(range(len(rows)), labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel('Synchronized wall seconds (excluding preparation)')
    ax.set_title(f'Time to loss ≤ {args.loss_threshold:g}; grey bars: threshold not reached')
    fig.savefig(args.output_dir/'calibration_timing.pdf'); fig.savefig(args.output_dir/'calibration_timing.png', dpi=180)
    plt.close(fig)
    return 0 if report['scientific_acceptance_pass'] else 10

if __name__ == '__main__':
    raise SystemExit(main())
