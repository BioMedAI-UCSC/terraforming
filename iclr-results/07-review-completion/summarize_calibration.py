"""Check and summarize the completed current-physics calibration artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root):
    read = lambda name: json.loads((root / name).read_text())
    assert read("complete.json")["status"] == "complete"
    provenance, optimization = read("provenance.json"), read("optimization.json")
    for source in provenance["source_snapshot"]:
        assert hashlib.sha256((root / source["snapshot"]).read_bytes()).hexdigest() == source["sha256"], source["path"]
    checks = pd.read_csv(root / "gradient-checks.csv")
    assert len(checks) == 9 and checks.pass_check.all()
    assert optimization["starts_completed"] == optimization["starts_requested"] == 3
    assert all(row["stationary"] and row["projected_gradient_inf"] <= optimization["gtol"]
               for row in optimization["terminations"])
    frame = pd.read_csv(root / "skill.csv")
    keys = ["window", "split", "block", "seconds"]
    assert not frame.duplicated(keys + ["model"]).any()
    baseline = frame[frame.model == "physical"].set_index(keys)
    fitted = frame[frame.model == "calibrated"].set_index(keys)
    assert baseline.index.equals(fitted.index)
    fields = ["temperature_rmse_k", "u_rmse_ms", "v_rmse_ms"]
    assert np.isfinite(frame[fields]).all().all()
    by_split = {split: {model: {field: float(np.sqrt(np.mean(rows[field]**2))) for field in fields}
                for model, rows in group.groupby("model")} for split, group in frame.groupby("split")}
    uncertainty = read("uncertainty.json")
    gain = by_split["test"]["physical"][fields[0]] - by_split["test"]["calibrated"][fields[0]]
    np.testing.assert_allclose(gain, uncertainty["gain_k"], rtol=1e-12, atol=1e-12)
    test = baseline.index.get_level_values("split") == "test"
    improved = int((fitted.loc[test, fields[0]] < baseline.loc[test, fields[0]]).sum())
    return dict(status="verified_current_physics_short_lead_calibration", by_split=by_split,
        test_temperature_gain_k=gain, test_temperature_gain_percent=100. * gain / by_split["test"]["physical"][fields[0]],
        test_windows=int(test.sum()), temperature_windows_improved=improved, uncertainty=uncertainty,
        hardware=provenance["environment"]["hardware"], source_files_verified=len(provenance["source_snapshot"]),
        gradient_checks=len(checks), optimization=optimization,
        limitations=["Native 7397.937-second lead; not an annual climate or equilibrium evaluation.",
            "Training endpoints are stationary; the validation-selected checkpoint is not stationary.",
            "The selected CO2 opacity control is at its lower bound; fitted controls are not validated physical constants.",
            "Four held-out windows in two Mars years; the interval is conditional, not population skill or optimizer uncertainty.",
            "Individual windows and wind components can worsen. No matched performance claim."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.root)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("test_temperature_gain_k", "test_temperature_gain_percent", "test_windows", "temperature_windows_improved")}))
