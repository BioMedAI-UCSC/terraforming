from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compare_gcm_performance_runs.py"
SPEC = importlib.util.spec_from_file_location("compare_gcm_performance_runs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def rows(temperature_offset=0.0, pressure_scale=1.0, mass_drift=1e-9):
    result = []
    for i, sol in enumerate((0.0, 300.0, 620.0)):
        result.append({
            "elapsed_sols": sol,
            "mean_surface_temperature_k": 200.0 + i + temperature_offset,
            "mean_surface_pressure_pa": (600.0 + 10.0 * i) * pressure_scale,
            "atmosphere_plus_surface_reservoir_mass_kg": 1e16 * (1.0 + mass_drift * i),
            "solar_longitude_deg": (i * 170.0) % 360.0,
        })
    return result


def test_equal_runs_pass_all_gates():
    report = MODULE.compare(rows(), rows(), MODULE.DEFAULT_GATES)
    assert report["status"] == "pass"
    assert all(check["pass"] for check in report["checks"].values())


def test_temperature_failure_is_reported():
    report = MODULE.compare(rows(), rows(temperature_offset=2.0), MODULE.DEFAULT_GATES)
    assert report["status"] == "fail"
    assert not report["checks"]["mean_surface_temperature_k"]["pass"]


def test_short_run_omits_seasonal_phase_gate():
    reference = rows()[:2]
    candidate = rows()[:2]
    report = MODULE.compare(reference, candidate, MODULE.DEFAULT_GATES)
    assert "seasonal_peak_ls_deg" not in report["checks"]
