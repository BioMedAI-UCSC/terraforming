"""Model vs. observed Mars surface-pressure climatology (answers reviewer Q5).

Overlays the simulator's **seasonal surface-pressure cycle** on the observed
annual pressure climatology at the Viking Lander sites — the canonical Mars
CO2-cycle benchmark, reproduced by the Mars Climate Database (MCD 6.1). Writes:

    docs/validation/mcd_pressure_comparison.png   — the side-by-side figure
    docs/validation/mcd_pressure_comparison.md    — the metrics table

What was calibrated, and how many parameters:
  * Surface-*temperature* diurnal cycle → REMS Gale-Crater data (Vasavada 2017):
    thermal inertia, diurnal-swing amplitude, thermal-tide amplitude + phase
    (~4 hand-set constants).
  * Seasonal surface-*pressure* cycle (this figure) → the observed annual
    pressure curve: a SINGLE parameter, the effective polar-cap area
    ``MARS_POLAR_CAP_FRACTION`` (0.04), set so the peak-to-peak swing matches the
    observed ~25-30 %.

Honesty note: MCD 6.1 is gated behind its own access tools and is not fetched
here. The reference curves below are the Viking Lander daily-mean pressure
cycles (Hess et al. 1980; Tillman et al. 1993) — the observational ground truth
MCD 6.1 is itself validated against (Guo et al. 2009). A direct MCD 6.1 seasonal
extraction drops in with the same ``(ls, pressure)`` shape.

Usage (from package/):  PYTHONPATH=. python scripts/mcd_pressure_comparison.py
"""

from __future__ import annotations

import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.celestials.planets.mars import MARS_LS_PERIHELION, Mars  # noqa: E402
from src.engine.time_controller import Accuracy, TimeController  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "validation"
_SOL = 88_775.244
_YEAR = 669 * _SOL
_LS = np.arange(0.0, 360.0, 30.0)

# Observed daily-mean surface pressure (Pa) vs Ls, digitised from the published
# Viking Lander curves (Hess et al. 1980; Tillman et al. 1993). VL1 sits at
# 22.3 N — matching the model's default site latitude.
SITES = {
    "Viking Lander 1  (22.3°N, -3.6 km)": dict(
        lat=22.3, elev=-3627.0,
        obs=np.array([760, 780, 790, 770, 730, 690, 710, 760, 840, 890, 850, 790.0]),
    ),
    "Viking Lander 2  (47.6°N, -4.5 km)": dict(
        lat=47.6, elev=-4505.0,
        obs=np.array([880, 910, 900, 850, 790, 730, 770, 850, 980, 1080, 1010, 920.0]),
    ),
}


def model_pressure_cycle(lat: float, elev: float) -> np.ndarray:
    """Spun-up daily-mean seasonal pressure cycle from the model at a site."""
    mars = Mars(device="cpu", latitude=lat, elevation_m=elev, initial_ls_deg=0.0)
    hist = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST).run(duration=3 * _YEAR)
    t = np.array([float(s.time) for s in hist])
    P = np.array([float(s.surface_pressure) for s in hist])
    oa = np.array([float(s.orbital_angle) for s in hist])
    ls = (np.degrees(oa + float(MARS_LS_PERIHELION))) % 360.0
    last = t >= 2 * _YEAR  # discard 2 years of spin-up
    idx = np.floor(ls / 30.0).astype(int)
    return np.array([P[last & (idx == k)].mean() for k in range(12)])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(SITES), figsize=(12, 4.6), sharey=False)
    rows = []
    for ax, (name, s) in zip(axes, SITES.items()):
        model = model_pressure_cycle(s["lat"], s["elev"])
        obs = s["obs"]
        rmse = float(np.sqrt(np.mean((model - obs) ** 2)))
        m_sw, o_sw = model.max() - model.min(), obs.max() - obs.min()
        ax.plot(_LS, obs, "ko-", lw=2, label="observed (Viking / MCD-consistent)")
        ax.plot(_LS, model, "C0^-", label=f"model (RMSE {rmse:.0f} Pa)")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Solar longitude  Ls (deg)")
        ax.set_ylabel("Surface pressure (Pa)")
        ax.set_xlim(0, 330)
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.3)
        rows.append(
            f"| {name} | {model.mean():.0f} / {obs.mean():.0f} | "
            f"{m_sw:.0f} ({100*m_sw/model.mean():.0f}%) / {o_sw:.0f} ({100*o_sw/obs.mean():.0f}%) "
            f"| {rmse:.0f} |"
        )
    fig.suptitle(
        "Model seasonal surface-pressure cycle vs observed climatology "
        "(1 calibrated parameter: polar-cap area)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT / "mcd_pressure_comparison.png", dpi=130)
    print(f"wrote {OUT / 'mcd_pressure_comparison.png'}")

    lines = [
        "# Model vs observed surface-pressure climatology",
        "",
        "Model seasonal cycle (no per-site tuning) vs the observed Viking-Lander "
        "daily-mean pressure curves (Hess 1980; Tillman 1993), the benchmark MCD "
        "6.1 reproduces. **One** calibrated parameter: `MARS_POLAR_CAP_FRACTION`.",
        "",
        "| Site | mean model / obs (Pa) | seasonal swing model / obs | RMSE (Pa) |",
        "|---|---|---|---|",
        *rows,
        "",
    ]
    (OUT / "mcd_pressure_comparison.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'mcd_pressure_comparison.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
