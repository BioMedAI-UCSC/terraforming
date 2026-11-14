"""The comparison changes dt while keeping the same physical duration."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from run_reviewer_stability import configuration
from analyse_reviewer_timesteps import FIELDS, analyse, paired_diagnostics


def test_timestep_cases_share_physical_duration_and_preserve_ultra_defaults():
    cases = [configuration("timestep", dt, 1.) for dt in (150., 300., 450.)]
    assert len({c["n_steps"] * c["dt_seconds"] for c in cases}) == 1
    ultra = configuration("ultra", 300., 1.)
    assert (ultra["truncation"], ultra["n_layers"], ultra["dt_seconds"], ultra["n_steps"]) == ("T170", 40, 150., 2000)
    assert ultra["hyperdiffusion_tau_seconds"] == "auto"
    with pytest.raises(ValueError):
        configuration("timestep", 600., 1.)


def test_paired_diagnostics_uses_physical_time_and_rejects_partial_comparison():
    first = [dict(step=step, **dict.fromkeys(FIELDS, value))
             for step, value in ((0, 0.), (2, 1.), (6, 3.))]
    second = [dict(step=step, **dict.fromkeys(FIELDS, 0.)) for step in (0, 1, 3)]
    result = paired_diagnostics(first, 150., second, 300.)
    assert result["fields"][FIELDS[0]]["time_weighted_rms_difference"] == pytest.approx(np.sqrt(3.5))
    with pytest.raises(ValueError, match="full common trajectory"):
        paired_diagnostics(first, 150., second[:-1], 300.)


def test_campaign_comparison_rejects_changed_inputs_and_initial_state(tmp_path):
    for dt in (450, 300, 150):
        folder = tmp_path / f"timestep-{dt}"
        folder.mkdir()
        config = dict(grid=dict(dt_seconds=dt, n_steps=900 // dt), forcing={}, co2_forcing={},
                      input_sha256={}, identity=dict(source_sha256={"physics.py": "fixed"}))
        (folder / "configuration.json").write_text(json.dumps(config))
        (folder / "report.json").write_text(json.dumps(dict(status="finite_complete")))
        rows = [dict(step=step, **dict.fromkeys(FIELDS, 0.)) for step in (0, 900 // dt)]
        (folder / "diagnostics.json").write_text(json.dumps(rows))
        np.savez(folder / "initial.npz", temperature=np.array([200.]))
    assert len(analyse(tmp_path)["pairs"]) == 3
    target = tmp_path / "timestep-150"
    np.savez(target / "initial.npz", temperature=np.array([201.]))
    with pytest.raises(ValueError, match="identical initial"):
        analyse(tmp_path)
    np.savez(target / "initial.npz", temperature=np.array([200.]))
    config["input_sha256"] = {"terrain": "changed"}
    (target / "configuration.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="Changed input_sha256"):
        analyse(tmp_path)
