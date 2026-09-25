#!/usr/bin/env python3
"""Plot new experiment artifacts using the existing paper palette and export style."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotstyle as ps


def annual_ablations(roots, out):
    """Combine compatible annual partitions without treating GPUs as replicates."""
    frames, contract = [], None
    for root in roots:
        read = lambda name: json.loads((root / name).read_text())
        config, report = read("config.json"), read("report.json")
        if report.get("execution_status") != "complete" or report.get("tasks", {}).get("ablations") != "complete":
            raise ValueError(f"Incomplete annual partition: {root}")
        if config["validation_seconds"][-1] != 59356800:
            raise ValueError(f"Expected a complete 59,356,800-second Mars year: {root}")
        identity = (config,
                    {k: v["sha256"] for k, v in read("input-manifest.json")["files"].items()},
                    sorted(read("source-hashes.json").values()))
        if contract is not None and identity != contract:
            raise ValueError(f"Incompatible annual configuration, inputs or source: {root}")
        contract = identity
        df = pd.read_csv(root / "ablations.csv")
        if not df.status.eq("finite").all() or df['case'].duplicated().any():
            raise ValueError(f"Nonfinite or duplicate case: {root}")
        df["source_directory"] = str(root)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    full = combined[combined['case'] == 'full']
    numeric = [c for c in frames[0].select_dtypes(include=np.number).columns
               if c not in ('wall_seconds', 'compile_seconds')]
    if len(full) != len(roots) or not np.allclose(full[numeric], full[numeric].iloc[0], rtol=1e-10, atol=1e-12):
        raise ValueError("Partition baselines do not match")
    cases = combined[combined['case'] != 'full']
    expected = {'no_pbl', 'no_convection', 'no_regolith', 'no_co2_exchange',
                'weaker_diffusion', 'half_timestep'} | {
                    f'{parameter}_{factor}' for parameter in
                    ('co2_longwave_opacity_scale', 'dust_longwave_opacity_scale', 'surface_exchange_multiplier')
                    for factor in ('0.8', '1.2')}
    if cases['case'].duplicated().any() or set(cases['case']) != expected:
        raise ValueError("Expected all 12 distinct annual ablation cases")
    cases = cases.sort_values('temperature_rmse_vs_full_k')
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
    labels = [ps.prettify_case(c) for c in cases['case']]
    for ax, column, title, unit, color in zip(axes,
            ('temperature_rmse_vs_full_k', 'u_rmse_vs_full_ms', 'v_rmse_vs_full_ms'),
            ('Temperature', 'Zonal wind', 'Meridional wind'), ('K', 'm s$^{-1}$', 'm s$^{-1}$'),
            ('blue', 'orange', 'green')):
        ax.barh(np.arange(len(cases)), cases[column], color=ps.PALETTE[color])
        ax.set(title=title, xlabel=f'RMSE vs full physics ({unit})')
    axes[0].set_yticks(np.arange(len(cases)), labels)
    fig.suptitle('One-Mars-year physical ablation sensitivity', weight='bold')
    ps.annotate_provenance(fig, 'GPU0–3 partitions; one common restart. Sample-averaged transient differences; not observational error or ensemble uncertainty.')
    fig.tight_layout(rect=(0, .04, 1, .95))
    ps.save(fig, out, 'year-ablations')
    pd.concat([full.iloc[:1], cases]).to_csv(out / 'year-ablations.csv', index=False)
    (out / 'year-ablations-provenance.json').write_text(json.dumps({
        'source_directories': list(map(str, roots)), 'cases_including_baseline': 13,
        'duration_seconds': 59356800, 'independent_initial_states': 1,
        'repeated_baselines': len(full), 'config': contract[0],
        'scope': 'Sample-averaged transient sensitivity; no equilibrium or observational accuracy claim'}, indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path, default=[])
    ap.add_argument("--year-runs", nargs="+", type=Path, default=[],
                    help="Complete compatible annual ablation partitions, including GPU0")
    ap.add_argument("--out", type=Path, default=Path("iclr-results/figures/out"))
    args = ap.parse_args()
    if not args.runs and not args.year_runs:
        ap.error("Supply --runs or --year-runs")
    ps.apply_style()
    if args.year_runs:
        annual_ablations(args.year_runs, args.out)
    for root in args.runs:
        found = False
        for name in ("gradients", "training", "skill", "ablations", "climate", "observations", "scaling"):
            path = root / f"{name}.csv"
            if not path.exists():
                continue
            found = True
            df = pd.read_csv(path)
            fig, ax = plt.subplots(figsize=(9, 5))
            caveat = ""
            if name == "gradients":
                for (direction, eps), sub in df.groupby(["direction", "epsilon"]):
                    ax.plot(sub.seconds, np.maximum(sub.relative_error, 1e-16), marker="o", label=f"parameter {direction}, epsilon={eps:g}")
                ax.set(xlabel="Rollout horizon (s)", ylabel="Gradient relative discrepancy", yscale="log")
                caveat = "See CSV for absolute errors, gradient magnitudes, pass flags and timings."
            elif name == "training":
                ax.plot(df['update'], df.train_loss_before_update, label="Training before update")
                ax.plot(df['update'], df.validation_loss, label="Validation after update")
                ax.set(xlabel="Update", ylabel="Normalized trajectory loss")
            elif name == "skill":
                for (split, model), sub in df.groupby(["split", "model"]):
                    means = sub.groupby("seconds").temperature_rmse_k.apply(lambda x: np.sqrt(np.mean(x**2)))
                    ax.plot(means.index, means, marker={"train": "s", "validation": "^", "test": "o"}[split],
                            color=ps.PALETTE[{"physical": "blue", "neural": "orange", "calibrated": "green"}[model]],
                            linestyle={"train": ":", "validation": "--", "test": "-"}[split], label=f"{split}: {model}")
                ax.set(xlabel="Lead time (s)", ylabel="Temperature RMSE (K)")
                ax.set_xticks(sorted(df.seconds.unique()))
                caveat = "Reference: " + ", ".join(df.reference_kind.unique()) + "; equal window weights, no inferred significance."
            elif name == "ablations":
                sub = df[df['case'] != 'full']
                groups = list(sub.groupby("case"))
                for i, (case, values) in enumerate(groups):
                    v = values.temperature_rmse_k
                    ax.scatter(v, np.full(len(v), i), s=20, color=ps.PALETTE['sky'])
                    ax.errorbar(v.mean(), i, xerr=v.std(ddof=1) if len(v)>1 else 0,
                                fmt="o", color=ps.PALETTE['blue'])
                ax.set_yticks(range(len(groups)), [ps.prettify_case(k) for k, _ in groups])
                ax.set_xlabel("Temperature RMSE vs paired full physics (K)")
                caveat = "Dots: starts; mean +/- SD across starts, not confidence intervals or independent climates."
            elif name == "climate":
                df = df[df.field == "mean_surface_pressure_pa"]
                for year, sub in df.groupby("year"):
                    ax.plot(sub.phase_sol, sub['mean'], label=f"Year {year}")
                ax.set(xlabel="Orbital phase (sols since initialization)", ylabel="Global mean surface pressure (Pa)")
                caveat = "Internal annual repeatability; not a Viking station comparison."
            else:
                for field, sub in df.groupby("field"):
                    if df.field.nunique() > 1:
                        plt.close(fig)
                        fig, ax = plt.subplots(figsize=(9, 5))
                    if name == "scaling":
                        ax.scatter(sub.median_seconds, sub.rmse)
                        for _, r in sub.iterrows():
                            ax.annotate(r.configuration, (r.median_seconds, r.rmse))
                        ax.set_xlabel("Matched forecast runtime (s)")
                    else:
                        ax.bar(sub.configuration, sub.rmse)
                    ax.set_ylabel(f"{field} RMSE ({sub.units.iloc[0]})")
                    ps.annotate_provenance(fig, str(path))
                    fig.tight_layout(rect=(0, .04, 1, 1))
                    ps.save(fig, args.out, f"{root.name}-{name}-{field}")
                continue
            ax.set_title(name.capitalize())
            if name != "ablations": ax.legend(fontsize=8)
            ps.annotate_provenance(fig, f"{path} — {caveat}")
            fig.tight_layout(rect=(0, .04, 1, 1))
            ps.save(fig, args.out, f"{root.name}-{name}")
        if not found:
            raise ValueError(f"No supported experiment artifacts in {root}")


if __name__ == "__main__":
    main()
