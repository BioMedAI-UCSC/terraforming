#!/usr/bin/env python3
"""Plot new experiment artifacts using the existing paper palette and export style."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotstyle as ps


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("iclr-results/figures/out"))
    args = ap.parse_args()
    ps.apply_style()
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
