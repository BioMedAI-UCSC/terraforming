"""Tests for cli/output.py.

Covers:
  - plot_diurnal: all data kept for runs <5 Mars years (clear diurnal sine waves)
  - plot_diurnal: x-axis tick interval snaps to 10–50 sol range for multi-sol runs
  - plot_diurnal: long runs (≥5 Mars years) aggregate to monthly seasonal cycles
  - plot_diurnal: output files created for all three channels (temp, pressure, ice)
  - _sol_tick_interval: returns None for ≤7 sols, snaps to nearest 10 in [10, 50]
  - _config_key: slug encoding for all four experiment types
  - _versioned_dir: base path, _v2, _v3 collision resolution
  - _out_dir: output_path passthrough, out_dir passthrough, auto date+config path
  - save_csv: correct columns and row count
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

from cli.models import (
    ExperimentConfig, ExpType, InterventionConfig,
    OutputConfig, PlanetConfig, SimConfig,
)


# ── Minimal Snapshot / RunResult stubs ────────────────────────────────────────

@dataclass
class _Snap:
    time:                torch.Tensor
    surface_temperature: torch.Tensor
    surface_pressure:    torch.Tensor
    ice_mass:            torch.Tensor
    solar_flux:          torch.Tensor
    orbital_angle:       torch.Tensor


def _snap(t, T, P, I=1e15, flux=500.0, angle=0.0) -> _Snap:
    return _Snap(
        time=torch.tensor(t, dtype=torch.float64),
        surface_temperature=torch.tensor(T, dtype=torch.float64),
        surface_pressure=torch.tensor(P, dtype=torch.float64),
        ice_mass=torch.tensor(I, dtype=torch.float64),
        solar_flux=torch.tensor(flux, dtype=torch.float64),
        orbital_angle=torch.tensor(angle, dtype=torch.float64),
    )


@dataclass
class _Result:
    """Structurally matches cli.runner.RunResult — avoids importing the CUDA chain."""
    name:    str
    history: list
    lat:     float
    lon:     float


_SOL_SECONDS  = 88775.244
_MARS_YEAR_SOLS = 668


def _multi_step_history(n_sols: int, steps_per_sol: int) -> list:
    """Build a history with `steps_per_sol` snapshots per sol."""
    dt = _SOL_SECONDS / steps_per_sol
    history = []
    for sol in range(n_sols):
        for step in range(steps_per_sol):
            t = (sol * steps_per_sol + step + 1) * dt
            phase = 2 * np.pi * step / steps_per_sol
            T = 210.0 + 50.0 * np.sin(phase)
            history.append(_snap(t, T, 610.0 + sol * 0.01))
    return history


def _single_step_history(n_sols: int) -> list:
    """Build a history with exactly 1 snapshot per sol."""
    history = []
    for sol in range(n_sols):
        t = (sol + 1) * _SOL_SECONDS
        history.append(_snap(t, 210.0 + sol * 0.01, 610.0))
    return history


# ── _config_key ───────────────────────────────────────────────────────────────

class TestConfigKey:

    def test_sol_north_hemisphere(self):
        """50-sol run at 22°N → mars_50sols_22N."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.sol, sols=50.0),
            planet=PlanetConfig(latitude=22.0),
        )
        assert _config_key(cfg) == "mars_50sols_22N"

    def test_sol_south_hemisphere(self):
        """10-sol run at 40°S → mars_10sols_40S."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.sol, sols=10.0),
            planet=PlanetConfig(latitude=-40.0),
        )
        assert _config_key(cfg) == "mars_10sols_40S"

    def test_sol_equator(self):
        """Equatorial run uses 0N."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.sol, sols=1.0),
            planet=PlanetConfig(latitude=0.0),
        )
        assert _config_key(cfg) == "mars_1sols_0N"

    def test_year_experiment(self):
        """Year run encodes 1yr and the latitude."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.year),
            planet=PlanetConfig(latitude=22.0),
        )
        assert _config_key(cfg) == "mars_1yr_22N"

    def test_multi_experiment_encodes_sols(self):
        """Multi-coord run encodes sol count (no latitude — covers all latitudes)."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.multi, sols=3.0),
        )
        assert _config_key(cfg) == "mars_multi_3sols"

    def test_intervention_experiment(self):
        """Intervention encodes n_years and latitude."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.intervention),
            intervention=InterventionConfig(n_years=50),
            planet=PlanetConfig(latitude=22.0),
        )
        assert _config_key(cfg) == "mars_iv_50yr_22N"

    def test_sols_rounded_to_integer(self):
        """Fractional sols are rounded to the nearest integer in the slug."""
        from cli.output import _config_key
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.sol, sols=50.9),
            planet=PlanetConfig(latitude=22.0),
        )
        assert _config_key(cfg) == "mars_51sols_22N"


# ── _versioned_dir ─────────────────────────────────────────────────────────────

class TestVersionedDir:

    def test_returns_base_when_nothing_exists(self, tmp_path):
        """First run returns the unversioned base path."""
        from cli.output import _versioned_dir
        result = _versioned_dir(str(tmp_path), "mars_50sols_22N")
        assert result == str(tmp_path / "mars_50sols_22N")

    def test_appends_v2_when_base_exists(self, tmp_path):
        """Second run (base dir present) returns _v2."""
        from cli.output import _versioned_dir
        (tmp_path / "mars_50sols_22N").mkdir()
        result = _versioned_dir(str(tmp_path), "mars_50sols_22N")
        assert result == str(tmp_path / "mars_50sols_22N_v2")

    def test_appends_v3_when_base_and_v2_exist(self, tmp_path):
        """Third run returns _v3."""
        from cli.output import _versioned_dir
        (tmp_path / "mars_50sols_22N").mkdir()
        (tmp_path / "mars_50sols_22N_v2").mkdir()
        result = _versioned_dir(str(tmp_path), "mars_50sols_22N")
        assert result == str(tmp_path / "mars_50sols_22N_v3")

    def test_does_not_skip_versions(self, tmp_path):
        """Versioning is sequential — skipping v2 is impossible."""
        from cli.output import _versioned_dir
        (tmp_path / "mars_50sols_22N").mkdir()
        # v2 absent — must return v2, not v3
        result = _versioned_dir(str(tmp_path), "mars_50sols_22N")
        assert "_v3" not in result
        assert result.endswith("_v2")


# ── _out_dir ──────────────────────────────────────────────────────────────────

class TestOutDir:

    def test_respects_explicit_output_path(self):
        """output_path overrides all logic and is returned verbatim."""
        from cli.output import _out_dir
        cfg = SimConfig(output=OutputConfig(output_path="/custom/path"))
        assert _out_dir(cfg) == "/custom/path"

    def test_respects_explicit_out_dir(self):
        """out_dir (--name flag) nests under outputs/ without a date subdirectory."""
        from cli.output import _out_dir
        cfg = SimConfig(output=OutputConfig(out_dir="my_experiment"))
        assert _out_dir(cfg) == os.path.join("outputs", "my_experiment")

    def test_auto_dir_has_date_prefix(self, monkeypatch):
        """Auto path contains outputs/<dd-mmm-yy>/."""
        import cli.output as out_mod
        fixed_dt = datetime(2026, 5, 2, tzinfo=timezone.utc)
        monkeypatch.setattr(
            out_mod, "datetime",
            type("_DT", (), {"now": staticmethod(lambda tz=None: fixed_dt)}),
        )
        cfg = SimConfig()
        result = out_mod._out_dir(cfg)
        assert result.startswith(os.path.join("outputs", "02-May-26"))

    def test_auto_dir_encodes_config_key(self, monkeypatch):
        """Auto path includes the experiment config slug."""
        import cli.output as out_mod
        fixed_dt = datetime(2026, 5, 2, tzinfo=timezone.utc)
        monkeypatch.setattr(
            out_mod, "datetime",
            type("_DT", (), {"now": staticmethod(lambda tz=None: fixed_dt)}),
        )
        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.sol, sols=50.0),
            planet=PlanetConfig(latitude=22.0),
        )
        result = out_mod._out_dir(cfg)
        assert result == os.path.join("outputs", "02-May-26", "mars_50sols_22N")

    def test_auto_dir_versions_on_collision(self, tmp_path, monkeypatch):
        """Auto path appends _v2 when the date/config dir already exists."""
        import cli.output as out_mod
        fixed_dt = datetime(2026, 5, 2, tzinfo=timezone.utc)
        monkeypatch.setattr(
            out_mod, "datetime",
            type("_DT", (), {"now": staticmethod(lambda tz=None: fixed_dt)}),
        )
        # Pre-create outputs/<date>/<key> relative to tmp_path so the collision
        # is detected when the CWD is tmp_path.
        (tmp_path / "outputs" / "02-May-26" / "mars_50sols_22N").mkdir(parents=True)
        monkeypatch.chdir(tmp_path)

        cfg = SimConfig(
            experiment=ExperimentConfig(type=ExpType.sol, sols=50.0),
            planet=PlanetConfig(latitude=22.0),
        )
        result = out_mod._out_dir(cfg)
        assert result == os.path.join("outputs", "02-May-26", "mars_50sols_22N_v2")


# ── _sol_tick_interval ────────────────────────────────────────────────────────

class TestSolTickInterval:

    def test_returns_none_for_very_short_runs(self):
        """≤7 sols → None so matplotlib auto-scales the x axis."""
        from cli.output import _sol_tick_interval
        assert _sol_tick_interval(1) is None
        assert _sol_tick_interval(7) is None

    def test_snaps_to_10_for_short_multi_sol_runs(self):
        """8–14 sols → interval of 10 (minimum)."""
        from cli.output import _sol_tick_interval
        assert _sol_tick_interval(8)  == 10
        assert _sol_tick_interval(14) == 10

    def test_snaps_to_50_for_long_runs(self):
        """700+ sols (≈1 Mars year) → interval of 50 (maximum)."""
        from cli.output import _sol_tick_interval
        assert _sol_tick_interval(700)  == 50
        assert _sol_tick_interval(3000) == 50

    def test_result_is_always_multiple_of_10(self):
        """Interval is always a multiple of 10 in the range [10, 50]."""
        from cli.output import _sol_tick_interval
        for n in range(8, 700, 13):
            iv = _sol_tick_interval(n)
            assert iv is not None
            assert iv % 10 == 0
            assert 10 <= iv <= 50


# ── plot_diurnal ──────────────────────────────────────────────────────────────

class TestPlotDiurnal:

    def test_creates_three_output_files_given_short_run(self, tmp_path):
        """plot_diurnal saves temp, pressure, and ice PNG files for short runs."""
        from cli.output import plot_diurnal

        result  = _Result("sim", _multi_step_history(n_sols=5, steps_per_sol=24), 0.0, 0.0)
        outfile = str(tmp_path / "mars_sol.png")
        plot_diurnal(result, outfile, "Test")

        assert os.path.exists(str(tmp_path / "mars_sol_temp.png"))
        assert os.path.exists(str(tmp_path / "mars_sol_pressure.png"))
        assert os.path.exists(str(tmp_path / "mars_sol_ice.png"))

    def test_creates_three_output_files_given_long_run(self, tmp_path):
        """plot_diurnal saves three PNG files even for long (year-scale) runs."""
        from cli.output import plot_diurnal

        result  = _Result("sim", _single_step_history(n_sols=_MARS_YEAR_SOLS), 0.0, 0.0)
        outfile = str(tmp_path / "mars_year.png")
        plot_diurnal(result, outfile, "Test")

        assert os.path.exists(str(tmp_path / "mars_year_temp.png"))
        assert os.path.exists(str(tmp_path / "mars_year_pressure.png"))
        assert os.path.exists(str(tmp_path / "mars_year_ice.png"))

    def test_short_run_keeps_all_raw_data(self, tmp_path, monkeypatch):
        """Runs of ≤7 sols keep every timestep so the diurnal sine wave is visible."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        captured_x: list[int] = []
        real_plot = plt.plot

        def _capture_plot(x, y, **kw):
            captured_x.append(len(x))
            return real_plot(x, y, **kw)

        from cli import output as out_mod
        monkeypatch.setattr(out_mod.plt, "plot", _capture_plot)

        n_sols, sps = 3, 24
        result = _Result("sim", _multi_step_history(n_sols=n_sols, steps_per_sol=sps), 0.0, 0.0)
        out_mod.plot_diurnal(result, str(tmp_path / "mars.png"), "Test")

        assert len(captured_x) == 3
        for pts in captured_x:
            assert pts == n_sols * sps, f"Expected {n_sols * sps} raw points, got {pts}"

    def test_medium_run_keeps_all_raw_data(self, tmp_path, monkeypatch):
        """Runs >7 sols but <5 Mars years keep every raw timestep (no clipping)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        captured_x: list[int] = []
        real_plot = plt.plot

        def _capture_plot(x, y, **kw):
            captured_x.append(len(x))
            return real_plot(x, y, **kw)

        from cli import output as out_mod
        monkeypatch.setattr(out_mod.plt, "plot", _capture_plot)

        n_sols, sps = 20, 24
        result = _Result("sim", _multi_step_history(n_sols=n_sols, steps_per_sol=sps), 0.0, 0.0)
        out_mod.plot_diurnal(result, str(tmp_path / "mars_sol.png"), "Test")

        assert len(captured_x) == 3
        for pts in captured_x:
            assert pts == n_sols * sps, f"Expected {n_sols * sps} raw points, got {pts}"

    def test_temperature_diurnal_range_is_preserved(self, tmp_path, monkeypatch):
        """Regression: daily-mean aggregation flattened the ±50 K diurnal swing.

        With raw temperature data, the max−min spread across all plotted points
        must be close to the injected 100 K peak-to-peak amplitude.
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        captured_y: list = []
        real_plot = plt.plot

        def _capture_plot(x, y, **kw):
            captured_y.append(np.asarray(y).copy())
            return real_plot(x, y, **kw)

        from cli import output as out_mod
        monkeypatch.setattr(out_mod.plt, "plot", _capture_plot)

        result  = _Result("sim", _multi_step_history(n_sols=20, steps_per_sol=24), 0.0, 0.0)
        outfile = str(tmp_path / "mars_sol.png")
        out_mod.plot_diurnal(result, outfile, "Test")

        # First captured call is the temperature channel
        T_plotted = captured_y[0]
        diurnal_range = float(T_plotted.max() - T_plotted.min())
        assert diurnal_range > 80.0, (
            f"Diurnal range {diurnal_range:.1f} K is too small — "
            "averaging may have collapsed the day/night cycle."
        )

    def test_long_run_aggregates_to_monthly(self, tmp_path, monkeypatch):
        """Runs ≥5 Mars years aggregate to monthly bins (far fewer plot points)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        captured_x: list[int] = []
        real_plot = plt.plot

        def _capture_plot(x, y, **kw):
            captured_x.append(len(x))
            return real_plot(x, y, **kw)

        from cli import output as out_mod
        monkeypatch.setattr(out_mod.plt, "plot", _capture_plot)

        # 6 Mars years at 1 step/sol
        n_sols = 6 * _MARS_YEAR_SOLS
        result = _Result("sim", _single_step_history(n_sols=n_sols), 0.0, 0.0)
        out_mod.plot_diurnal(result, str(tmp_path / "mars.png"), "Test")

        assert len(captured_x) == 3
        # Monthly bins ≈ n_sols / 55.7
        expected_months = int(n_sols / out_mod._MARS_MONTH_SOLS)
        for pts in captured_x:
            assert abs(pts - expected_months) <= 2, (
                f"Expected ≈{expected_months} monthly points, got {pts}"
            )


# ── save_csv ──────────────────────────────────────────────────────────────────

class TestSaveCsv:

    def test_csv_has_correct_columns_and_row_count(self, tmp_path):
        """save_csv writes exactly one row per snapshot with the expected columns."""
        from cli.output import save_csv

        n = 10
        result  = _Result("sim", _single_step_history(n_sols=n), 0.0, 0.0)
        outfile = str(tmp_path / "data" / "out.csv")
        save_csv(result, outfile)

        assert os.path.exists(outfile)
        with open(outfile) as f:
            rows = list(csv.reader(f))

        assert rows[0] == ["time_hours", "temperature_k", "pressure_pa",
                           "ice_mass_kg", "solar_flux_wm2", "orbital_angle_rad"]
        assert len(rows) - 1 == n

    def test_csv_temperature_values_are_positive(self, tmp_path):
        """Physical invariant: all saved temperatures must be > 0 K."""
        from cli.output import save_csv

        result  = _Result("sim", _single_step_history(n_sols=5), 0.0, 0.0)
        outfile = str(tmp_path / "check.csv")
        save_csv(result, outfile)

        with open(outfile) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["temperature_k"]) > 0
