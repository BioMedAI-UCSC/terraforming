"""Tests for src.interventions.state (GHGState).

Covers:
  - Initialisation: zero mass, device placement, compound list
  - inject: mass increases, cumulative_injected tracks total injected
  - decay: exponential first-order decay reduces mass correctly
  - inject unknown compound raises KeyError
  - get_all_masses_kg and get_cumulative_injected return shallow copies
"""

from __future__ import annotations

import math

import pytest
import torch

from src.interventions.state import GHGState


_CPU = torch.device("cpu")


def _val(t: torch.Tensor) -> float:
    return float(t.item())


class TestGHGStateInit:

    def test_initial_masses_are_zero(self):
        state = GHGState(["CF4", "SF6"], device=_CPU)
        assert _val(state.get_mass_kg("CF4")) == pytest.approx(0.0)
        assert _val(state.get_mass_kg("SF6")) == pytest.approx(0.0)

    def test_initial_cumulative_injected_are_zero(self):
        state = GHGState(["CF4"], device=_CPU)
        assert _val(state.get_cumulative_injected()["CF4"]) == pytest.approx(0.0)

    def test_compounds_property(self):
        state = GHGState(["CF4", "SF6", "NF3"], device=_CPU)
        assert set(state.compounds) == {"CF4", "SF6", "NF3"}

    def test_device_property(self):
        state = GHGState(["CF4"], device=_CPU)
        assert state.device == _CPU

    def test_unknown_compound_raises_on_init(self):
        with pytest.raises(KeyError):
            GHGState(["FAKE_GAS"], device=_CPU)

    def test_tensors_are_on_correct_device(self):
        state = GHGState(["CF4"], device=_CPU)
        assert state.get_mass_kg("CF4").device.type == "cpu"


class TestGHGStateInject:

    def test_inject_increases_mass(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        assert _val(state.get_mass_kg("CF4")) == pytest.approx(1e9, rel=1e-6)

    def test_inject_is_additive(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        state.inject({"CF4": 5e8})
        assert _val(state.get_mass_kg("CF4")) == pytest.approx(1.5e9, rel=1e-6)

    def test_inject_updates_cumulative(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        state.inject({"CF4": 2e9})
        assert _val(state.get_cumulative_injected()["CF4"]) == pytest.approx(3e9, rel=1e-6)

    def test_inject_only_named_compound(self):
        """Injecting CF4 must not change SF6 mass."""
        state = GHGState(["CF4", "SF6"], device=_CPU)
        state.inject({"CF4": 1e9})
        assert _val(state.get_mass_kg("SF6")) == pytest.approx(0.0)

    def test_inject_unknown_compound_raises(self):
        state = GHGState(["CF4"], device=_CPU)
        with pytest.raises(KeyError, match="not tracked"):
            state.inject({"SF6": 1e9})

    def test_inject_zero_kg_leaves_mass_unchanged(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        state.inject({"CF4": 0.0})
        assert _val(state.get_mass_kg("CF4")) == pytest.approx(1e9, rel=1e-6)


class TestGHGStateDecay:

    def test_decay_reduces_mass(self):
        state = GHGState(["NF3"], device=_CPU)
        state.inject({"NF3": 1e9})
        state.decay(dt_years=1.0)
        # NF3 lifetime = 500 yr; decay factor = exp(-1/500) ≈ 0.998002
        expected = 1e9 * math.exp(-1.0 / 500.0)
        assert _val(state.get_mass_kg("NF3")) == pytest.approx(expected, rel=1e-5)

    def test_decay_does_not_affect_cumulative_injected(self):
        """Decay reduces atmospheric mass but does NOT reduce cumulative_injected."""
        state = GHGState(["NF3"], device=_CPU)
        state.inject({"NF3": 1e9})
        state.decay(dt_years=100.0)
        assert _val(state.get_cumulative_injected()["NF3"]) == pytest.approx(1e9, rel=1e-6)

    def test_decay_long_lived_compound_is_slow(self):
        """CF4 (50 000 yr lifetime) should barely decay over 1 year."""
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e12})
        state.decay(dt_years=1.0)
        # After 1 yr: factor = exp(-1/50000) ≈ 0.99998
        retained_fraction = _val(state.get_mass_kg("CF4")) / 1e12
        assert retained_fraction == pytest.approx(math.exp(-1 / 50_000), rel=1e-5)

    def test_decay_never_goes_negative(self):
        state = GHGState(["NF3"], device=_CPU)
        state.inject({"NF3": 1e6})
        state.decay(dt_years=10_000.0)  # very long — nearly all gone
        assert _val(state.get_mass_kg("NF3")) >= 0.0

    def test_decay_zero_dt_leaves_mass_unchanged(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        state.decay(dt_years=0.0)
        assert _val(state.get_mass_kg("CF4")) == pytest.approx(1e9, rel=1e-6)


class TestGHGStateAccessors:

    def test_get_all_masses_kg_is_dict(self):
        state = GHGState(["CF4", "SF6"], device=_CPU)
        result = state.get_all_masses_kg()
        assert isinstance(result, dict)
        assert set(result) == {"CF4", "SF6"}

    def test_get_all_masses_kg_is_shallow_copy(self):
        """Modifying the returned dict must not mutate internal state."""
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        snapshot = state.get_all_masses_kg()
        snapshot["CF4"] = torch.tensor(0.0)
        # internal mass should be unchanged
        assert _val(state.get_mass_kg("CF4")) == pytest.approx(1e9, rel=1e-6)

    def test_get_cumulative_injected_is_shallow_copy(self):
        state = GHGState(["CF4"], device=_CPU)
        state.inject({"CF4": 1e9})
        snap = state.get_cumulative_injected()
        snap["CF4"] = torch.tensor(0.0)
        assert _val(state.get_cumulative_injected()["CF4"]) == pytest.approx(1e9, rel=1e-6)
