"""Generate the model-vs-observed climatology side-by-side + calibration report.

Runs the 0-D Mars model at the Viking Lander 1 site (22°N — the model's default
latitude), overlays its seasonal surface-pressure cycle on the VL1 observed cycle
(the canonical CO₂-cycle benchmark reproduced by MCD 6.1), and shows the effect of
calibrating the single dominant free parameter (polar cap fraction) to that cycle.
Writes:

  docs/validation/mcd_vs_model.png          — the side-by-side figure
  docs/validation/calibration_metrics.md    — the metrics table (before/after)

The calibrated value ``CALIBRATED_CAP_FRACTION`` is the optimum found by
``src.calibration.calibrate`` (SciPy Nelder-Mead, 1 parameter, tuned to the VL1
pressure cycle). Re-derive it with ``--recalibrate`` (slow: runs the optimiser).

Usage (from package/):  PYTHONPATH=. python scripts/mcd_validation.py
"""

from __future__ import annotations

import argparse
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.calibration import calibrate, evaluate, get_reference, simulate_seasonal_cycle

OUT = pathlib.Path(__file__).resolve().parents[2] / "docs" / "validation"
DT, N_YEARS = 3600.0, 2
DEFAULT_CAP = 0.01           # the REMS-temperature-tuned default
CALIBRATED_CAP_FRACTION = 0.0506  # calibrate() optimum vs VL1 pressure (RMSE 50 Pa)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recalibrate", action="store_true",
                    help="re-run the SciPy optimiser (slow) instead of the stored optimum")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    ref = get_reference("vl1")

    default_cycle = simulate_seasonal_cycle(
        ref, overrides={"polar_cap_fraction": DEFAULT_CAP}, dt=DT, n_years=N_YEARS
    )
    default_m = evaluate(default_cycle, ref, "pressure")

    cap = CALIBRATED_CAP_FRACTION
    if args.recalibrate:
        print("Recalibrating polar_cap_fraction to VL1 pressure (slow) ...")
        result = calibrate(
            ref, ["polar_cap_fraction"], x0=[0.02], field="pressure",
            bounds=[(0.002, 0.3)], maxiter=25, dt=7200.0, n_years=2,
        )
        cap = result.tuned_parameters["polar_cap_fraction"]
        print(f"  optimum cap={cap:.4f}, RMSE {result.rmse_before:.0f} -> {result.rmse_after:.0f}")

    tuned_cycle = simulate_seasonal_cycle(
        ref, overrides={"polar_cap_fraction": cap}, dt=DT, n_years=N_YEARS
    )
    tuned_m = evaluate(tuned_cycle, ref, "pressure")

    # ── Figure: pressure overlay (left) + model temperature diagnostic (right) ──
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(12, 4.6))

    axp.plot(ref.ls_deg, ref.pressure_pa, "ko-", lw=2, label="VL1 observed (Hess 1980)")
    axp.plot(default_cycle.ls_deg, default_cycle.pressure_pa, "C3s--",
             label=f"model default (cap={DEFAULT_CAP:.2f})  RMSE {default_m.rmse:.0f} Pa")
    axp.plot(tuned_cycle.ls_deg, tuned_cycle.pressure_pa, "C0^-",
             label=f"model calibrated (cap={cap:.3f})  RMSE {tuned_m.rmse:.0f} Pa")
    axp.set_xlabel("Solar longitude  Lₛ (deg)")
    axp.set_ylabel("Surface pressure (Pa)")
    axp.set_title("Surface-pressure annual cycle — model vs VL1 (22°N)")
    axp.set_xlim(0, 330)
    axp.legend(fontsize=8, loc="upper left")
    axp.grid(alpha=0.3)

    axt.plot(tuned_cycle.ls_deg, tuned_cycle.temperature_k, "C2o-",
             label="model daily-mean surface T")
    axt.set_xlabel("Solar longitude  Lₛ (deg)")
    axt.set_ylabel("Surface temperature (K)")
    axt.set_title("Surface temperature (model diagnostic — no VL1 T target)")
    axt.set_xlim(0, 330)
    axt.legend(fontsize=8, loc="upper left")
    axt.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "mcd_vs_model.png", dpi=130)
    print(f"wrote {OUT / 'mcd_vs_model.png'}")

    # ── Metrics table ──
    lines = [
        "# VL1 pressure calibration — metrics",
        "",
        f"Site: {ref.name} ({ref.latitude_deg}°N, {ref.elevation_m:.0f} m). "
        f"Reference: {ref.citation}. Tuned parameters: **1** "
        f"(`polar_cap_fraction`: {DEFAULT_CAP:.3f} → {cap:.3f}).",
        "",
        f"| Metric | Default (cap={DEFAULT_CAP:.2f}) | Calibrated (cap={cap:.3f}) | VL1 observed |",
        "|---|---|---|---|",
        f"| Annual mean (Pa) | {default_m.model_mean:.0f} | {tuned_m.model_mean:.0f} | {default_m.reference_mean:.0f} |",
        f"| Seasonal amplitude (Pa) | {default_m.model_amplitude:.0f} | {tuned_m.model_amplitude:.0f} | {default_m.reference_amplitude:.0f} |",
        f"| Amplitude ratio | {default_m.amplitude_ratio:.2f} | {tuned_m.amplitude_ratio:.2f} | 1.00 |",
        f"| Phase lag (° Lₛ) | {default_m.phase_lag_deg:+.0f} | {tuned_m.phase_lag_deg:+.0f} | 0 |",
        f"| RMSE (Pa) | {default_m.rmse:.0f} | {tuned_m.rmse:.0f} | — |",
        "",
    ]
    (OUT / "calibration_metrics.md").write_text("\n".join(lines))
    print(f"wrote {OUT / 'calibration_metrics.md'}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
