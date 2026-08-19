"""Tests for prescribed seasonal Ames dust staging."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dinosaur")

from src.gcm3d.coordinates import coordinate_system  # noqa: E402
from src.gcm3d.dust import seasonal_dust_on_grid  # noqa: E402


AMES_DUST = Path(__file__).resolve().parents[3] / "AmesGCM/data/DustScenario_Background.nc"


def test_ames_seasonal_dust_is_finite_periodic_and_spatial():
    grid = coordinate_system(truncation="T21", n_layers=6).horizontal
    tau0, zmax0 = seasonal_dust_on_grid(grid, AMES_DUST, 0.0)
    tau360, zmax360 = seasonal_dust_on_grid(grid, AMES_DUST, 360.0)
    assert tau0.shape == grid.nodal_shape
    assert np.isfinite(tau0).all() and np.isfinite(zmax0).all()
    assert tau0.min() > 0.0 and np.ptp(tau0) > 0.0
    assert np.allclose(tau0, tau360)
    assert np.allclose(zmax0, zmax360)


def test_dust_changes_with_season():
    grid = coordinate_system(truncation="T21", n_layers=6).horizontal
    tau0, _ = seasonal_dust_on_grid(grid, AMES_DUST, 0.0)
    tau270, _ = seasonal_dust_on_grid(grid, AMES_DUST, 270.0)
    assert np.max(np.abs(tau270 - tau0)) > 0.01
