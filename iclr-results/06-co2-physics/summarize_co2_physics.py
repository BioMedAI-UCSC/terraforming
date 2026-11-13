"""Verify local artifacts and draw the paired CO2 mechanism result."""
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parent
reports = {}
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for dt, style in ((300, "-"), (150, "--")):
    folder = root / f"co2-polar-night-verified-{dt}s"
    report = json.loads((folder / "report.json").read_text())
    assert report["status"] == "pass" and not report["source_changed_during_run"]
    for filename, expected in report["artifact_sha256"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == expected
    with zipfile.ZipFile(folder / "executed_source.zip") as archive:
        for filename, expected in report["source_sha256"].items():
            assert hashlib.sha256(archive.read(filename)).hexdigest() == expected
    reports[dt] = report
    duration = report["configuration"]["elapsed_sols"]
    for case, color, label in (("surface_only", "#b64a32", "Surface only"),
                               ("atmospheric_condensation", "#176e9e", "Atmospheric condensation")):
        with np.load(folder / f"{case}_history.npz", allow_pickle=False) as history:
            time = np.arange(1, len(history["frost_deficit_k"]) + 1) * duration / len(history["frost_deficit_k"])
            axes[0].plot(time, history["frost_deficit_k"], style, color=color, label=f"{label}, {dt} s")
            drift = history["corrected_total_mass_kg"] / report["cases"][case]["initial"]["total_co2_mass_kg"] - 1.
            axes[1].plot(time, drift * 1.e13, style, color=color)
axes[0].set(xlabel="Elapsed Mars sols", ylabel="Maximum frost-point deficit (K)", ylim=(0, None))
axes[0].legend(fontsize=8)
axes[1].set(xlabel="Elapsed Mars sols", ylabel="Post-projection CO₂ mass drift (×10⁻¹³)")
for ax in axes:
    ax.grid(alpha=.2)
fig.suptitle("Paired synthetic polar night · T21/L4 · float64 · identical initial state")
fig.savefig(root / "co2-physics-comparison.png", dpi=200)
fig.savefig(root / "co2-physics-comparison.pdf")
plt.close(fig)

with np.load(root / "co2-polar-night-verified-300s/initial.npz", allow_pickle=False) as first, \
     np.load(root / "co2-polar-night-verified-150s/initial.npz", allow_pickle=False) as second:
    assert first.files == second.files
    for field in first.files:
        np.testing.assert_array_equal(first[field], second[field])
assert reports[300]["source_sha256"] == reports[150]["source_sha256"]
assert reports[300]["configuration"]["elapsed_sols"] == reports[150]["configuration"]["elapsed_sols"]
summary = {"status": "pass", "duration_sols": reports[300]["configuration"]["elapsed_sols"],
    "cases": {}, "timestep_endpoint_differences_300_minus_150": {},
    "limitations": reports[300]["limitations"],
    "source_snapshots_verified": True, "initial_states_identical": True}
for case in ("surface_only", "atmospheric_condensation"):
    summary["cases"][case] = {str(dt): {key: report["cases"][case][key] for key in (
        "final_frost_deficit_k", "max_relative_total_co2_mass_drift", "max_raw_step_mass_error_kg",
        "max_projection_energy_change_j", "finite", "valid_inventory")}
        for dt, report in reports.items()}
    summary["timestep_endpoint_differences_300_minus_150"][case] = {
        key: reports[300]["cases"][case]["final"][key] - reports[150]["cases"][case]["final"][key]
        for key in ("mean_surface_pressure_pa", "mean_surface_temperature_k", "min_air_temperature_k",
                    "mean_co2_cloud_ice_pa", "max_wind_speed_ms")}
for dt, report in reports.items():
    summary["cases"]["atmospheric_condensation"][str(dt)]["deficit_reduction_percent"] = 100. * (
        1. - report["cases"]["atmospheric_condensation"]["final_frost_deficit_k"]
        / report["cases"]["surface_only"]["final_frost_deficit_k"])
(root / "co2-physics-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
