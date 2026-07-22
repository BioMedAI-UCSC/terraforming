"""Show the present-day Mars baseline: is the *unforced* model a stable state?

Runs the unforced model for several Mars years from CO₂-budget-consistent
present-day initial conditions and writes:

  docs/validation/baseline_stability.png   — multi-year P/T/ice time series
  docs/validation/baseline_metrics.md      — mean pressure, swing, drift table

Usage (from package/):  PYTHONPATH=. python scripts/baseline_stability.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.calibration.baseline import run_unforced, summarize_baseline

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "validation"
N_YEARS, DT, SPINUP = 5, 3600.0, 1


def _daily_mean(t: np.ndarray, y: np.ndarray, day_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Bin a series into per-sol means (declutters the diurnal cycle)."""
    day = np.floor(t / day_s).astype(int)
    days = np.unique(day)
    return days, np.array([y[day == d].mean() for d in days])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run = run_unforced(n_years=N_YEARS, dt=DT)
    s = summarize_baseline(run, spinup_years=SPINUP)

    day_s = 88_775.244
    yrs = run.time_s / run.year_s
    dP_x, dP = _daily_mean(run.time_s, run.pressure_pa, day_s)
    dT_x, dT = _daily_mean(run.time_s, run.temperature_k, day_s)
    dI_x, dI = _daily_mean(run.time_s, run.ice_kg, day_s)
    x_years = dP_x * day_s / run.year_s

    fig, (axp, axt, axi) = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)

    axp.plot(x_years, dP, "C0-", lw=1.2, label="daily-mean surface pressure")
    axp.axhline(600.0, color="k", ls=":", lw=1, label="6.0 mb (present-day global mean)")
    axp.axhline(s.mean_pressure_pa, color="C3", ls="--", lw=1,
                label=f"model mean {s.mean_pressure_mb:.2f} mb")
    axp.axvspan(0, SPINUP, color="0.85", label="spin-up (excluded)")
    axp.set_ylabel("Pressure (Pa)")
    axp.set_title(
        f"Unforced Mars baseline — mean {s.mean_pressure_mb:.2f} mb, "
        f"seasonal swing {s.seasonal_swing_pa:.0f} Pa ({s.seasonal_swing_pct:.0f}%), "
        f"drift {s.pressure_drift_pa_per_year:+.2f} Pa/yr"
    )
    axp.legend(fontsize=8, loc="lower right", ncol=2)
    axp.grid(alpha=0.3)

    axt.plot(x_years, dT, "C1-", lw=1.2)
    axt.axhline(s.mean_temperature_k, color="C3", ls="--", lw=1,
                label=f"mean {s.mean_temperature_k:.1f} K")
    axt.set_ylabel("Surface T (K)")
    axt.legend(fontsize=8, loc="lower right")
    axt.grid(alpha=0.3)

    axi.plot(x_years, dI / 1e15, "C2-", lw=1.2)
    axi.set_ylabel("Ice (10¹⁵ kg)")
    axi.set_xlabel("Time (Mars years)")
    axi.set_xlim(0, N_YEARS)
    axi.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "baseline_stability.png", dpi=130)
    print(f"wrote {OUT / 'baseline_stability.png'}")

    stable = s.is_stable()
    lines = [
        "# Unforced present-day baseline — stability",
        "",
        f"Initial conditions: {run.pressure_pa[0]:.0f} Pa atmosphere, "
        f"{run.ice_kg[0]:.2e} kg cap CO₂, no forcing "
        f"(caps relax to a ~{s.mean_ice_kg:.2e} kg seasonal-mean attractor). "
        f"{N_YEARS} Mars years, first {SPINUP} discarded as spin-up.",
        "",
        "| Quantity | Model | Present-day Mars |",
        "|---|---|---|",
        f"| Mean surface pressure | {s.mean_pressure_mb:.2f} mb | ~6.1 mb |",
        f"| Seasonal swing | {s.seasonal_swing_pa:.0f} Pa ({s.seasonal_swing_pct:.0f}%) | ~25–30 % (Viking) |",
        f"| Swing repeatability (σ across years) | {s.swing_repeatability_pa:.2f} Pa | — |",
        f"| Pressure drift | {s.pressure_drift_pa_per_year:+.3f} Pa/yr "
        f"({s.pressure_drift_pct_per_century:+.2f} %/century) | — |",
        f"| Temperature drift | {s.temperature_drift_k_per_year:+.4f} K/yr | — |",
        f"| Ice drift | {s.ice_drift_kg_per_year:+.2e} kg/yr | — |",
        f"| **Stable limit cycle?** | **{'YES' if stable else 'NO'}** | — |",
        "",
    ]
    (OUT / "baseline_metrics.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'baseline_metrics.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
