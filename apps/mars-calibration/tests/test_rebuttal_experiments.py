"""Independent checks for the averaging, terrain and Viking data boundaries."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from analyse_rebuttal_climate import interval_mean, terrain_residual, read_viking


def test_interval_mean_integrates_ramp_at_fractional_boundaries():
    times = np.array([0., 1., 3., 5.])
    values = np.stack([2*times+3, 4*times-1], axis=1)
    np.testing.assert_allclose(interval_mean(times, values, .5, 4.), [7.5, 8.])
    with pytest.raises(ValueError):
        interval_mean(times, values, -1., 3.)


def test_pressure_residual_removes_known_hydrostatic_terrain_signal():
    z = np.array([[0., 1000.], [2000., 3000.]])
    residual = terrain_residual(610*np.exp(-z/10000), z, np.ones_like(z))
    np.testing.assert_allclose(residual, 0., atol=1e-14)


def test_viking_fixed_width_units_and_quality_flags(tmp_path):
    path = tmp_path / "viking.dat"
    path.write_text(' VL1  1  97.039    0  7.624  7.534  0.255  0.0670 18.01  0  0  0 18  0 49  0.000000 .F \n'
                    ' VL1  1  97.196    1  7.629  7.534  0.169  0.0687  2.28 -9 -9 -9 -9 -9 -9  0.000030 BB \n')
    rows = read_viking(path)
    assert len(rows) == 1
    assert rows[0]["station"] == "VL1"
    assert rows[0]["pressure_pa"] == pytest.approx(762.9)
