"""Tests for cli/output.py.

Covers:
  - plot_diurnal: short runs (≤50 sols) keep all raw data — clear diurnal sine waves
  - plot_diurnal: medium runs (<5 years) show a 10–50 sol window of diurnal data
  - plot_diurnal: long runs (≥5 Mars years) aggregate to monthly seasonal cycles
  - plot_diurnal: output files created for all three channels (temp, pressure, ice)
  - save_csv: correct columns and row count
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch


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
        """Runs of ≤50 sols keep every timestep (clear diurnal sine waves)."""
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

        n_sols  = 5
        sps     = 24
        result  = _Result("sim", _multi_step_history(n_sols=n_sols, steps_per_sol=sps), 0.0, 0.0)
        out_mod.plot_diurnal(result, str(tmp_path / "mars.png"), "Test")

        assert len(captured_x) == 3
        for pts in captured_x:
            assert pts == n_sols * sps, f"Expected {n_sols * sps} raw points, got {pts}"

    def test_medium_run_clips_to_10_to_50_sol_window(self, tmp_path, monkeypatch):
        """Runs longer than 50 sols but under 5 years clip to a 10–50 sol window."""
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

        n_sols = 300   # well above 50 sols, below 5 years
        sps    = 4
        result = _Result("sim", _multi_step_history(n_sols=n_sols, steps_per_sol=sps), 0.0, 0.0)
        out_mod.plot_diurnal(result, str(tmp_path / "mars.png"), "Test")

        assert len(captured_x) == 3
        window = out_mod._diurnal_window(n_sols)
        assert 10 <= window <= 50
        for pts in captured_x:
            # allow ±sps for boundary rounding
            assert abs(pts - window * sps) <= sps, (
                f"Expected ≈{window * sps} points for {window}-sol window, got {pts}"
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
