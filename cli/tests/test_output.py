"""Tests for cli/output.py.

Covers:
  - plot_diurnal: per-sol aggregation for multi-step-per-sol histories (anti-band fix)
  - plot_diurnal: raw data passthrough for single-step-per-sol histories
  - plot_diurnal: output files are created for all three channels (temp, pressure, ice)
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
    name:    str
    history: list
    lat:     float
    lon:     float


_SOL_SECONDS = 88775.244  # one Mars sol in seconds


def _multi_step_history(n_sols: int, steps_per_sol: int) -> list:
    """Build a history with `steps_per_sol` snapshots per sol."""
    dt = _SOL_SECONDS / steps_per_sol
    history = []
    for sol in range(n_sols):
        for step in range(steps_per_sol):
            t = (sol * steps_per_sol + step + 1) * dt
            # Diurnal oscillation: ±50 K around 210 K
            phase = 2 * np.pi * step / steps_per_sol
            T = 210.0 + 50.0 * np.sin(phase)
            history.append(_snap(t, T, 610.0 + sol * 0.01))
    return history


def _single_step_history(n_sols: int) -> list:
    """Build a history with exactly 1 snapshot per sol (large-dt case)."""
    history = []
    for sol in range(n_sols):
        t = (sol + 1) * _SOL_SECONDS
        history.append(_snap(t, 210.0 + sol * 0.01, 610.0))
    return history


# ── plot_diurnal ──────────────────────────────────────────────────────────────

class TestPlotDiurnal:

    def test_creates_three_output_files_given_multi_step_history(self, tmp_path):
        """plot_diurnal saves temp, pressure, and ice PNG files."""
        from cli.output import plot_diurnal

        result  = _Result("sim", _multi_step_history(n_sols=20, steps_per_sol=24), 0.0, 0.0)
        outfile = str(tmp_path / "mars_sol.png")

        plot_diurnal(result, outfile, "Test")

        assert os.path.exists(str(tmp_path / "mars_sol_temp.png"))
        assert os.path.exists(str(tmp_path / "mars_sol_pressure.png"))
        assert os.path.exists(str(tmp_path / "mars_sol_ice.png"))

    def test_creates_three_output_files_given_single_step_history(self, tmp_path):
        """plot_diurnal works cleanly when dt == 1 sol (no aggregation needed)."""
        from cli.output import plot_diurnal

        result  = _Result("sim", _single_step_history(n_sols=50), 0.0, 0.0)
        outfile = str(tmp_path / "mars_year.png")

        plot_diurnal(result, outfile, "Test")

        assert os.path.exists(str(tmp_path / "mars_year_temp.png"))
        assert os.path.exists(str(tmp_path / "mars_year_pressure.png"))
        assert os.path.exists(str(tmp_path / "mars_year_ice.png"))

    def test_aggregated_plot_has_one_point_per_sol(self, tmp_path, monkeypatch):
        """Regression: multi-step history must be collapsed to ~N_sols data points.

        Without the aggregation fix, plotting 20 sols × 24 steps renders a thick
        filled band because every diurnal oscillation is drawn as a separate line
        segment.  After the fix the plot has ≈one mean value per sol (within ±1
        due to floating-point sol-boundary alignment).
        """
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

        n_sols  = 20
        sps     = 24
        result  = _Result("sim", _multi_step_history(n_sols=n_sols, steps_per_sol=sps), 0.0, 0.0)
        outfile = str(tmp_path / "mars_sol.png")

        out_mod.plot_diurnal(result, outfile, "Test")

        # Three channels (temp, pressure, ice) — each must have far fewer points
        # than raw (20×24=480) and close to n_sols (±1 for boundary rounding).
        assert len(captured_x) == 3
        for pts in captured_x:
            assert abs(pts - n_sols) <= 1, (
                f"Expected ≈{n_sols} sol-averaged points, got {pts}. "
                "Diurnal aliasing (thick band) not fixed."
            )

    def test_short_run_plots_raw_data(self, tmp_path, monkeypatch):
        """Runs of ≤5 sols keep raw timesteps to show the diurnal cycle."""
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

        # 2 sols × 24 steps (≤5 sols) → should NOT aggregate
        n_steps = 2 * 24
        result  = _Result("sim", _multi_step_history(n_sols=2, steps_per_sol=24), 0.0, 0.0)
        outfile = str(tmp_path / "mars_short.png")
        out_mod.plot_diurnal(result, outfile, "Test")

        # Raw 48 steps should be plotted unchanged
        for pts in captured_x:
            assert pts == n_steps, f"Short run should keep {n_steps} raw points, got {pts}"


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

        header = rows[0]
        assert header == ["time_hours", "temperature_k", "pressure_pa",
                          "ice_mass_kg", "solar_flux_wm2", "orbital_angle_rad"]
        assert len(rows) - 1 == n  # header + n data rows

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
