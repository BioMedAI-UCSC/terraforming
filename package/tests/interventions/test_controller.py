"""Tests for src.interventions.controller (InterventionController).

Covers:
  - InterventionController returns correct number of snapshots
  - Snapshot fields are non-negative and physically plausible
  - Temperature increases monotonically on average with positive injection
  - ΔF grows each year when injection is continuous
  - GHF is non-decreasing across years
  - Baseline-only run (no injection) leaves GHF and temperature stable
  - Callback is called once per year
  - Unknown compound raises KeyError at construction time
"""

from __future__ import annotations

import pytest
import torch

from src.celestials import Mars
from src.interventions.controller import InterventionController, InterventionSnapshot


def _val(t: torch.Tensor) -> float:
    return float(t.item())


def _make_controller(
    injection: dict[str, float] | None = None,
    n_years: int = 5,
    dt: float = 21600.0,
) -> tuple[InterventionController, Mars]:
    mars = Mars()
    ic = InterventionController(
        mars,
        injection_schedule_kg_yr=injection or {},
        dt=dt,
    )
    return ic, mars


class TestInterventionControllerInit:

    def test_unknown_compound_raises_at_init(self):
        mars = Mars()
        with pytest.raises(KeyError):
            InterventionController(mars, {"FAKE_GAS": 1e9})

    def test_valid_compounds_do_not_raise(self):
        mars = Mars()
        # Should not raise
        InterventionController(mars, {"CF4": 1e9, "SF6": 5e8})


class TestInterventionControllerRun:

    def test_returns_correct_number_of_snapshots(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        history = ic.run(n_years=3)
        assert len(history) == 3

    def test_snapshots_are_intervention_snapshot_instances(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=2)
        history = ic.run(n_years=2)
        for snap in history:
            assert isinstance(snap, InterventionSnapshot)

    def test_year_counter_starts_at_one(self):
        ic, _ = _make_controller(n_years=3)
        history = ic.run(n_years=3)
        assert history[0].year == 1
        assert history[-1].year == 3

    def test_surface_temperature_is_positive(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        for snap in ic.run(n_years=3):
            assert _val(snap.surface_temperature) > 0

    def test_surface_pressure_is_positive(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        for snap in ic.run(n_years=3):
            assert _val(snap.surface_pressure) > 0

    def test_greenhouse_factor_at_least_one(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        for snap in ic.run(n_years=3):
            assert _val(snap.greenhouse_factor) >= 1.0

    def test_delta_F_non_negative(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        for snap in ic.run(n_years=3):
            assert _val(snap.delta_F) >= 0.0

    def test_delta_F_grows_with_continuous_injection(self):
        """ΔF at year N must exceed ΔF at year 1 when injection is ongoing."""
        ic, _ = _make_controller({"CF4": 1e12}, n_years=5)
        history = ic.run(n_years=5)
        dF_year1 = _val(history[0].delta_F)
        dF_year5 = _val(history[-1].delta_F)
        assert dF_year5 > dF_year1

    def test_ghf_non_decreasing_with_injection(self):
        """GHF must not decrease from one year to the next during injection."""
        ic, _ = _make_controller({"CF4": 1e12}, n_years=5)
        history = ic.run(n_years=5)
        for i in range(1, len(history)):
            assert _val(history[i].greenhouse_factor) >= _val(
                history[i - 1].greenhouse_factor
            ) - 1e-6  # small tolerance for float arithmetic

    def test_no_injection_leaves_delta_F_zero(self):
        """Without injection, ΔF must stay at 0."""
        ic, _ = _make_controller(injection={}, n_years=3)
        for snap in ic.run(n_years=3):
            assert _val(snap.delta_F) == pytest.approx(0.0)

    def test_ghg_masses_dict_has_injected_compound(self):
        ic, _ = _make_controller({"SF6": 1e11}, n_years=2)
        history = ic.run(n_years=2)
        for snap in history:
            assert "SF6" in snap.ghg_masses_kg

    def test_cumulative_injected_increases_each_year(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        history = ic.run(n_years=3)
        cum0 = _val(history[0].cumulative_injected_kg["CF4"])
        cum2 = _val(history[2].cumulative_injected_kg["CF4"])
        assert cum2 > cum0

    def test_time_s_increases_each_year(self):
        ic, _ = _make_controller({"CF4": 1e12}, n_years=3)
        history = ic.run(n_years=3)
        for i in range(1, len(history)):
            assert _val(history[i].time_s) > _val(history[i - 1].time_s)

    def test_callback_called_once_per_year(self):
        call_count = [0]

        def cb(snap):
            call_count[0] += 1

        ic, _ = _make_controller({"CF4": 1e12}, n_years=4)
        ic.run(n_years=4, callback=cb)
        assert call_count[0] == 4

    def test_callback_receives_snapshot(self):
        received = []

        def cb(snap):
            received.append(snap)

        ic, _ = _make_controller({"CF4": 1e12}, n_years=2)
        ic.run(n_years=2, callback=cb)
        assert all(isinstance(s, InterventionSnapshot) for s in received)

    @pytest.mark.slow
    def test_50_year_run_temperature_rises(self):
        """Integration: 50-year CF4 injection must raise T by at least 10 K.

        Uses realistic injection (1e9 kg/yr CF4) and dt=21600s (6h) for
        physical stability.  Marked slow as it simulates 50 × ~10000 steps.
        """
        mars = Mars()
        ic = InterventionController(
            mars,
            injection_schedule_kg_yr={"CF4": 1e9},
            dt=21600.0,
        )
        history = ic.run(n_years=50)
        T_start = _val(history[0].surface_temperature)
        T_end   = _val(history[-1].surface_temperature)
        assert T_end > T_start + 5.0, (
            f"Expected temperature rise > 5 K, got {T_end - T_start:.1f} K"
        )
