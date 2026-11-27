"""Repeat verification must catch physical and learned-result changes."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from verify_radiation_repeat import compare


def test_compare_ignores_measurement_time_but_rejects_physics_and_nan():
    compare({"training_seconds": 1., "dt_seconds": 30.},
            {"training_seconds": 2., "dt_seconds": 30.})
    for changed in (150., np.nan):
        with pytest.raises(AssertionError):
            compare({"dt_seconds": 30.}, {"dt_seconds": changed})
    with pytest.raises(AssertionError):
        compare({"rmse_k": .5}, {"rmse_k": .6})
