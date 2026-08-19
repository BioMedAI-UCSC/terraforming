"""Tests for cli.main — the `mars maps` 3-D gcm3d command.

Covers:
  - `mars maps --help` exits zero and documents the key options.
  - a small end-to-end `mars maps` run writes PNGs + NetCDF (skipped without the
    gcm3d extra or the MOLA raster; runs in an isolated filesystem so no repo I/O).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from cli.main import cli


def test_mars_maps_help_exits_zero():
    result = CliRunner().invoke(cli, ["mars", "maps", "--help"])
    assert result.exit_code == 0
    assert "--truncation" in result.output
    assert "--co2" in result.output
    assert "--no-physics" in result.output


@pytest.mark.slow
def test_mars_maps_writes_outputs():
    pytest.importorskip("dinosaur")
    from src.celestials.planets.mars import topography as topo

    if not topo._DEFAULT_MOLA.exists():
        pytest.skip("MOLA raster not staged")

    runner = CliRunner()
    with runner.isolated_filesystem() as tmp:
        result = runner.invoke(cli, [
            "mars", "maps",
            "--truncation", "T21", "--layers", "6",
            "--dt", "600", "--steps", "20", "--ls", "270",
            "--name", "unittest",
        ])
        assert result.exit_code == 0, result.output
        outdir = Path(tmp) / "outputs" / "gcm3d_maps"
        assert (outdir / "unittest_temperature.png").exists()
        assert (outdir / "unittest_co2_ice.png").exists()
        assert (outdir / "unittest_maps.nc").exists()


@pytest.mark.slow
def test_mars_maps_no_physics_has_no_co2_panel():
    pytest.importorskip("dinosaur")
    from src.celestials.planets.mars import topography as topo

    if not topo._DEFAULT_MOLA.exists():
        pytest.skip("MOLA raster not staged")

    runner = CliRunner()
    with runner.isolated_filesystem() as tmp:
        result = runner.invoke(cli, [
            "mars", "maps", "--no-physics",
            "--truncation", "T21", "--layers", "6",
            "--dt", "600", "--steps", "10", "--name", "dry",
        ])
        assert result.exit_code == 0, result.output
        outdir = Path(tmp) / "outputs" / "gcm3d_maps"
        assert (outdir / "dry_temperature.png").exists()
        # dry dynamics carries no CO2 frost field
        assert not (outdir / "dry_co2_ice.png").exists()
