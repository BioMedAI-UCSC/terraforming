#!/usr/bin/env python3
"""Run the original 13-case ablation suite and reproduce its RMSE bar plot."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLBACKEND', 'Agg')
os.environ.setdefault('MPLCONFIGDIR', 'outputs/mpl-cache')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sols', type=int, choices=[30, 338], required=True)
    p.add_argument('--inputs', type=Path, default=Path('outputs/nautilus-calibration-inputs'))
    p.add_argument('--output-dir', type=Path, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--prepare-only', action='store_true')
    group.add_argument('--plot-only', action='store_true')
    p.add_argument('--require-gpu', action='store_true')
    args = p.parse_args()
    config = json.loads((ROOT/'apps/mars-calibration/paper-experiment.json').read_text())
    # Five equally spaced instantaneous observations; all align to dt and dt/2.
    end_steps = round(args.sols * 88775.244 / 300)
    times = [round(end_steps * i / 5) * 300 for i in range(1, 6)]
    config.update(train_seconds=times[:3], validation_seconds=times[3:],
                  profile=f'paired-ablations-{args.sols}sols-five-samples')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        p.error('Existing duration config differs; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2)+'\n')
    run = args.output_dir/'run'
    if args.prepare_only:
        print(f'Prepared {config_path}: sample seconds {times}; actual duration {times[-1]/88775.244:.6f} sols')
        return 0
    if not args.plot_only:
        command = [sys.executable, str(ROOT/'scripts/run_mars_paper_experiments.py'),
                   '--inputs', str(args.inputs), '--config', str(config_path),
                   '--output', str(run), '--task', 'ablations']
        if args.require_gpu:
            command.append('--require-gpu')
        subprocess.run(command, check=True)
    actual = json.loads((run/'config.json').read_text())
    if actual != config:
        raise ValueError('Run config does not match requested horizon')
    report = json.loads((run/'ablations.json').read_text())
    with (run/'ablations.csv').open() as f:
        rows = list(csv.DictReader(f))
    if report.get('status') != 'complete' or len(rows) != 13 or any(r['status'] != 'finite' for r in rows):
        raise ValueError('Expected all 13 finite completed cases; inspect ablations.json for failures')
    sys.path.insert(0, str(ROOT/'iclr-results/figures'))
    import plotstyle as ps
    from make_figures import fig_ablations
    ps.apply_style()
    out = args.output_dir/'plots'
    out.mkdir(exist_ok=True)
    fig_ablations(ROOT/'iclr-results', out, csv_path=run/'ablations.csv',
                  title=f'{args.sols}-sol physics ablation suite (five sampled times)')
    print(out/'ablations_rmse.png')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
