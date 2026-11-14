"""The comparison changes dt while keeping the same physical duration."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from run_reviewer_stability import configuration


def test_timestep_cases_share_physical_duration_and_preserve_ultra_defaults():
    cases = [configuration("timestep", dt, 1.) for dt in (150., 300., 450.)]
    assert len({c["n_steps"] * c["dt_seconds"] for c in cases}) == 1
    ultra = configuration("ultra", 300., 1.)
    assert (ultra["truncation"], ultra["n_layers"], ultra["dt_seconds"], ultra["n_steps"]) == ("T170", 40, 150., 2000)
    assert ultra["hyperdiffusion_tau_seconds"] == "auto"
    with pytest.raises(ValueError):
        configuration("timestep", 600., 1.)
