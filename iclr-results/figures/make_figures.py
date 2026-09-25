#!/usr/bin/env python3
"""Generate beautified, consistent paper figures from the iclr-results archive.

Every figure is written as PNG (preview) and PDF (vector for LaTeX) into
`iclr-results/figures/out/`. The script reads only the curated archive by
default; nothing under it is modified.

Usage:
    python iclr-results/figures/make_figures.py all
    python iclr-results/figures/make_figures.py ablations year benchmarks timing neural
    python iclr-results/figures/make_figures.py --root iclr-results --out iclr-results/figures/out all

Groups:
    ablations   short-paper physics ablation suite (RMSE / energy / AAM)
    year        one-Mars-year ablation partitions (gpu1/gpu2/gpu3)
    benchmarks  reference comparison (Ames / MCD / ARCO) skill vs season
    timing      throughput / wall-clock performance
    neural      radiation training curves, held-out generalization,
                parameter recovery and inverse-control descent
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotstyle as ps

CAVEAT = ("Diagnostic / transient run; RMSE is relative to each run's full case, "
          "not observational validation.")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        print(f"  ! skip (missing): {path}")
        return None
    return pd.read_csv(path)


def _bar_labels(ax, bars, fmt="{:.2f}", pad=3):
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt.format(w), (w, b.get_y() + b.get_height() / 2),
                    xytext=(pad, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=8, color=ps.PALETTE["ink"])


# --------------------------------------------------------------------------- #
# ablations
# --------------------------------------------------------------------------- #
def fig_ablations(root: Path, out: Path, *, csv_path=None,
                  title="Short-paper physics ablation suite"):
    source = Path(csv_path) if csv_path is not None else root / "02-ablations/short-paper-suite/ablations.csv"
    df = _read_csv(source)
    if df is None:
        return
    df = df[df["case"] != "full"].copy()
    df["label"] = df["case"].map(ps.prettify_case)
    df["wind_rmse"] = np.hypot(df["u_rmse_vs_full_ms"], df["v_rmse_vs_full_ms"])
    df = df.sort_values("temperature_rmse_vs_full_k")

    # Figure A: temperature and wind sensitivity side by side.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    y = np.arange(len(df))
    b1 = axes[0].barh(y, df["temperature_rmse_vs_full_k"],
                      color=ps.PALETTE["blue"], edgecolor="white")
    axes[0].set_yticks(y, df["label"])
    axes[0].set_xlabel("Temperature RMSE vs full (K)")
    axes[0].set_title("Physics ablation: temperature sensitivity")
    _bar_labels(axes[0], b1)

    b2 = axes[1].barh(y, df["wind_rmse"], color=ps.PALETTE["orange"],
                      edgecolor="white")
    axes[1].set_xlabel("Wind-vector RMSE vs full (m s$^{-1}$)")
    axes[1].set_title("Physics ablation: wind sensitivity")
    _bar_labels(axes[1], b2)
    fig.suptitle(title, fontsize=13, weight="bold")
    ps.annotate_provenance(fig, "02-ablations/short-paper-suite/ablations.csv — " + CAVEAT if csv_path is None else source.name + " — " + CAVEAT)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    print("  wrote", *[p.name for p in ps.save(fig, out, "ablations_rmse")])

    # Figure B: diverging energy / AAM change.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)
    dfe = df.sort_values("energy_change_j_m2_sr")
    ye = np.arange(len(dfe))
    cols = [ps.PALETTE["red"] if v < 0 else ps.PALETTE["green"]
            for v in dfe["energy_change_j_m2_sr"]]
    axes[0].barh(ye, dfe["energy_change_j_m2_sr"] / 1e3, color=cols, edgecolor="white")
    axes[0].set_yticks(ye, dfe["label"])
    axes[0].axvline(0, color=ps.PALETTE["ink"], lw=1)
    axes[0].set_xlabel("Energy change (kJ m$^{-2}$ sr$^{-1}$)")
    axes[0].set_title("Energy budget change")

    cols2 = [ps.PALETTE["red"] if v < 0 else ps.PALETTE["green"]
             for v in dfe["aam_change_kg_m_s_sr"]]
    axes[1].barh(ye, dfe["aam_change_kg_m_s_sr"] / 1e9, color=cols2, edgecolor="white")
    axes[1].axvline(0, color=ps.PALETTE["ink"], lw=1)
    axes[1].set_xlabel("AAM change ($10^9$ kg m s$^{-1}$ sr$^{-1}$)")
    axes[1].set_title("Angular-momentum change")
    fig.suptitle("Ablation conservation diagnostics", fontsize=13, weight="bold")
    ps.annotate_provenance(fig, "02-ablations/short-paper-suite/ablations.csv — not budget closure.")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    print("  wrote", *[p.name for p in ps.save(fig, out, "ablations_conservation")])


# --------------------------------------------------------------------------- #
# one-Mars-year ablations
# --------------------------------------------------------------------------- #
def fig_year(root: Path, out: Path):
    frames = []
    for part in ("mars-year-gpu1", "mars-year-gpu2", "mars-year-gpu3"):
        df = _read_csv(root / f"02-ablations/{part}/ablations.csv")
        if df is None:
            continue
        df = df[df["case"] != "full"].copy()
        df["partition"] = part.replace("mars-year-", "")
        frames.append(df)
    if not frames:
        return
    allf = pd.concat(frames, ignore_index=True)
    allf["label"] = allf["case"].map(ps.prettify_case)

    order = (allf.groupby("label")["temperature_rmse_vs_full_k"].max()
             .sort_values().index.tolist())
    parts = sorted(allf["partition"].unique())
    pcol = {p: ps.SEQ[i] for i, p in enumerate(parts)}

    fig, ax = plt.subplots(figsize=(10, 5.6))
    y = np.arange(len(order))
    h = 0.8 / max(len(parts), 1)
    for i, p in enumerate(parts):
        sub = allf[allf["partition"] == p].set_index("label")
        vals = [sub["temperature_rmse_vs_full_k"].get(lbl, np.nan) for lbl in order]
        ax.barh(y + i * h, vals, height=h, label=f"partition {p}",
                color=pcol[p], edgecolor="white")
    ax.set_yticks(y + h * (len(parts) - 1) / 2, order)
    ax.set_xlabel("Temperature RMSE vs full (K)")
    ax.set_title("One-Mars-year ablation sensitivity (transient from restart)")
    ax.legend(title="GPU run")
    ps.annotate_provenance(fig, "02-ablations/mars-year-gpu{1,2,3} — separate complete "
                           "runs; do not pool repeated full rows.")
    fig.tight_layout()
    print("  wrote", *[p.name for p in ps.save(fig, out, "year_ablations")])


# --------------------------------------------------------------------------- #
# benchmarks / reference comparison
# --------------------------------------------------------------------------- #
def fig_benchmarks(root: Path, out: Path):
    df = _read_csv(root / "01-mola-topography-comparisons/metrics.csv")
    if df is None:
        return
    fields = list(dict.fromkeys(df["field"]))
    refs = list(dict.fromkeys(df["reference"]))

    # RMSE vs season, small multiples per field.
    n = len(fields)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(11, 3.4 * nrow), squeeze=False)
    for idx, field in enumerate(fields):
        ax = axes[idx // ncol][idx % ncol]
        sub = df[df["field"] == field]
        for ref in refs:
            s = sub[sub["reference"] == ref].sort_values("Ls")
            if s.empty:
                continue
            ax.plot(s["Ls"], s["rmse"], marker="o",
                    color=ps.REFERENCE_COLORS.get(ref, ps.PALETTE["grey"]), label=ref)
        units = sub["units"].iloc[0] if not sub.empty else ""
        ax.set_title(field)
        ax.set_xlabel("Solar longitude $L_s$ (deg)")
        ax.set_ylabel(f"RMSE ({units})")
        ax.set_xticks(sorted(sub["Ls"].unique()))
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=len(refs), loc="upper center",
               bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Model vs reference RMSE by season", fontsize=13, weight="bold", y=1.02)
    ps.annotate_provenance(fig, "01-mola-topography-comparisons/metrics.csv — "
                           "reference diagnostics, not observational truth.")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    print("  wrote", *[p.name for p in ps.save(fig, out, "benchmark_rmse_by_season")])

    # Correlation heatmap (field x reference), averaged over Ls.
    piv = (df.groupby(["field", "reference"])["correlation"].mean()
           .unstack("reference").reindex(index=fields, columns=refs))
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(refs), 0.7 * len(fields) + 1.5))
    im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(refs)), refs, rotation=20, ha="right")
    ax.set_yticks(range(len(fields)), fields)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=9)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Mean pattern correlation")
    ax.set_title("Model–reference spatial correlation")
    ps.annotate_provenance(fig, "01-mola-topography-comparisons/metrics.csv (Ls-averaged).")
    fig.tight_layout()
    print("  wrote", *[p.name for p in ps.save(fig, out, "benchmark_correlation")])


# --------------------------------------------------------------------------- #
# timing / performance
# --------------------------------------------------------------------------- #
def fig_timing(root: Path, out: Path):
    path = root / "03-benchmarks/performance-timing/timing.json"
    if not path.exists():
        print(f"  ! skip (missing): {path}")
        return
    data = json.loads(path.read_text())
    m = data["measurements"]
    names = [x["workload"].replace("_", " ") for x in m]
    thr = [x["simulated_sols_per_wall_hour"] for x in m]
    med = [x["median_seconds"] for x in m]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    y = np.arange(len(names))
    b = axes[0].barh(y, thr, color=ps.PALETTE["blue"], edgecolor="white")
    axes[0].set_yticks(y, names)
    axes[0].set_xlabel("Simulated sols / wall-clock hour")
    axes[0].set_title("Throughput")
    _bar_labels(axes[0], b, fmt="{:.0f}")

    b2 = axes[1].barh(y, med, color=ps.PALETTE["orange"], edgecolor="white")
    axes[1].set_yticks(y, names)
    axes[1].set_xlabel("Median step time (s)")
    axes[1].set_title("Warm wall-clock per workload")
    _bar_labels(axes[1], b2, fmt="{:.2f}")
    fig.suptitle("Differentiable GCM performance", fontsize=13, weight="bold")
    ps.annotate_provenance(fig, "03-benchmarks/performance-timing/timing.json — "
                           "single-GPU warm calls; no external speedup claimed.")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    print("  wrote", *[p.name for p in ps.save(fig, out, "performance_timing")])


# --------------------------------------------------------------------------- #
# neural experiments
# --------------------------------------------------------------------------- #
def _load(path: Path):
    if not path.exists():
        print(f"  ! skip (missing): {path}")
        return None
    return json.loads(path.read_text())


def fig_neural(root: Path, out: Path):
    rad = _load(root / "04-neural-experiments/radiation/report.json")
    if rad is not None:
        # Training curves.
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in ("local", "trajectory", "continued_local"):
            hist = rad["training"][method]["history"]
            upd = [h["update"] for h in hist]
            val = [h["validation_after_update"] for h in hist]
            ax.plot(upd, val, color=ps.METHOD_COLORS[method],
                    label=ps.METHOD_LABELS[method])
        ax.set_yscale("log")
        ax.set_xlabel("Update")
        ax.set_ylabel("Validation loss (log)")
        ax.set_title("Neural radiation: training convergence")
        ax.grid(True, which="both", axis="both")
        ax.legend()
        ps.annotate_provenance(fig, "04-neural-experiments/radiation/report.json — "
                               "generated data; no Mars forecast skill.")
        fig.tight_layout()
        print("  wrote", *[p.name for p in ps.save(fig, out, "neural_radiation_training")])

        # Held-out generalization: test vs extrapolation air RMSE.
        splits = ["test", "extrapolation"]
        methods = ["local", "trajectory", "continued_local"]
        vals = {sp: [] for sp in splits}
        for sp in splits:
            cols = rad["column_evaluation"][sp]
            # keep only per-column entries that carry a per-method 'models' block
            entries = [v for v in cols.values()
                       if isinstance(v, dict) and "models" in v]
            for meth in methods:
                per = [e["models"][meth]["air_rmse_k"] for e in entries]
                vals[sp].append(float(np.mean(per)) if per else np.nan)
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(methods))
        w = 0.38
        for i, sp in enumerate(splits):
            bars = ax.bar(x + (i - 0.5) * w, vals[sp], width=w,
                          label=sp.capitalize(),
                          color=[ps.PALETTE["blue"], ps.PALETTE["orange"]][i],
                          edgecolor="white")
            for b in bars:
                ax.annotate(f"{b.get_height():.3f}",
                            (b.get_x() + b.get_width() / 2, b.get_height()),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8)
        ax.set_xticks(x, [ps.METHOD_LABELS[m] for m in methods])
        ax.set_ylabel("Mean air-temperature RMSE (K)")
        ax.set_title("Neural radiation: held-out generalization")
        ax.legend(title="Split")
        ps.annotate_provenance(fig, "04-neural-experiments/radiation/report.json "
                               "(held-out dust splits).")
        fig.tight_layout()
        print("  wrote", *[p.name for p in ps.save(fig, out, "neural_radiation_generalization")])

    # Inverse control descent.
    ctl = _load(root / "04-neural-experiments/control/report.json")
    if ctl is not None and "history" in ctl:
        it = [h["iteration"] for h in ctl["history"]]
        loss = [h["loss_before_update"] for h in ctl["history"]]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(it, loss, color=ps.PALETTE["purple"], marker="o")
        ax.set_xlabel("Optimization iteration")
        ax.set_ylabel("Control objective")
        ax.set_title("Neural inverse control: gradient descent through the GCM")
        txt = (f"init obj {ctl['initial_objective']:.4f} -> "
               f"final {ctl['final_objective']:.4f}\n"
               f"max |heating| {ctl['max_abs_policy_heating_k_s']:.2e} K/s "
               f"(bound {ctl['configuration']['maximum_heating_k_s']:.0e})")
        ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=8.5, color=ps.PALETTE["ink"])
        ps.annotate_provenance(fig, "04-neural-experiments/control/report.json — "
                               "synthetic integration demonstration.")
        fig.tight_layout()
        print("  wrote", *[p.name for p in ps.save(fig, out, "neural_control_descent")])

    # Parameter recovery (optional; reads GPU csv fallback if archived csv absent).
    rec_csv = None
    for cand in (root / "04-neural-experiments/parameter-recovery/parameter-recovery.csv",
                 Path("outputs/paper-recovery-gpu/parameter-recovery.csv")):
        if cand.exists():
            rec_csv = cand
            break
    if rec_csv is not None:
        df = pd.read_csv(rec_csv)
        df["run"] = df["start_id"].astype(str) + "/" + df["method"]
        df = df.sort_values("max_parameter_relative_error")
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = [ps.PALETTE["green"] if p else ps.PALETTE["red"]
                  for p in df["acceptance_pass"]]
        y = np.arange(len(df))
        ax.barh(y, df["max_parameter_relative_error"].clip(lower=1e-30),
                color=colors, edgecolor="white")
        ax.set_yticks(y, df["run"])
        ax.set_xscale("log")
        ax.set_xlabel("Max parameter relative error (log)")
        ax.set_title("Physical parameter recovery accuracy")
        ax.text(0.97, 0.05, "green = acceptance pass", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=ps.PALETTE["green"])
        ps.annotate_provenance(fig, f"{rec_csv} — generated-data inverse recovery.")
        fig.tight_layout()
        print("  wrote", *[p.name for p in ps.save(fig, out, "parameter_recovery")])


# --------------------------------------------------------------------------- #
GROUPS = {
    "ablations": fig_ablations,
    "year": fig_year,
    "benchmarks": fig_benchmarks,
    "timing": fig_timing,
    "neural": fig_neural,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("groups", nargs="+", choices=["all", *GROUPS],
                    help="figure groups to build")
    ap.add_argument("--root", default="iclr-results", type=Path,
                    help="archive root (default: iclr-results)")
    ap.add_argument("--out", default=None, type=Path,
                    help="output dir (default: <root>/figures/out)")
    args = ap.parse_args()

    out = args.out or (args.root / "figures/out")
    ps.apply_style()
    todo = list(GROUPS) if "all" in args.groups else args.groups
    for g in todo:
        print(f"[{g}]")
        GROUPS[g](args.root, out)
    print(f"\nDone. Figures in {out}/ (PNG + PDF).")


if __name__ == "__main__":
    main()
