"""Tests for src/engine/batched_controller.py.

Covers:
  - BatchedMars: construction validation, [B]-shaped state stacking
  - BatchedTimeController: dt validation, run() output, remainder step
  - BatchedHistory: field shapes, B property, to_lists() equivalence
  - Batch self-consistency: identical configs → identical columns
  - Chunk-compiled CUDA path vs eager path (slow, CUDA-gated)
"""

from __future__ import annotations

import pytest
import torch

from src.celestials import Mars
from src.engine import Accuracy, BatchedTimeController
from src.engine.batched_controller import BatchedHistory, BatchedMars
from src.engine.time_controller import Snapshot

SOL = 88_775.244
HOUR = 3600.0


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mars_batch(lats=(-40.0, 0.0, 45.0)):
    """B CPU Mars instances at the given latitudes."""
    return [Mars(latitude=lat, device="cpu") for lat in lats]


def _run(mars_list, duration, accuracy=Accuracy.FAST, **kw) -> BatchedHistory:
    btc = BatchedTimeController(mars_list, dt=HOUR, accuracy=accuracy, **kw)
    return btc.run(duration)


# ── BatchedMars ───────────────────────────────────────────────────────────────

class TestBatchedMars:

    def test_empty_mars_list_raises(self):
        """An empty batch is a configuration error, not a silent no-op."""
        with pytest.raises(ValueError):
            BatchedMars([])

    def test_state_tensors_have_batch_shape(self):
        """All per-instance state must stack to [B] on construction."""
        bm = BatchedMars(_mars_batch())
        assert bm._B == 3
        for t in (bm._T, bm._P, bm._M, bm.elapsed_time,
                  bm.orbital_angle, bm.solar_flux, bm._lat):
            assert t.shape == (3,)

    def test_pack_state_shape_is_B_by_3(self):
        bm = BatchedMars(_mars_batch())
        assert bm.pack_state().shape == (3, 3)

    def test_unpack_state_clamps_negative_ice_to_zero(self):
        """Physical invariant: ice mass can never go negative."""
        bm = BatchedMars(_mars_batch())
        y = bm.pack_state()
        y[:, 2] = -1.0
        bm.unpack_state(y)
        assert torch.all(bm._M >= 0.0)


# ── BatchedTimeController.run ─────────────────────────────────────────────────

class TestBatchedTimeControllerRun:

    def test_invalid_dt_raises_on_zero(self):
        with pytest.raises(ValueError):
            BatchedTimeController(_mars_batch(), dt=0.0)

    def test_invalid_dt_raises_on_negative(self):
        with pytest.raises(ValueError):
            BatchedTimeController(_mars_batch(), dt=-3600.0)

    def test_invalid_chunk_raises(self):
        """chunk < 1 is a configuration error."""
        with pytest.raises(ValueError):
            BatchedTimeController(_mars_batch(), chunk=0)

    def test_run_returns_batched_history(self):
        hist = _run(_mars_batch(), duration=24 * HOUR)
        assert isinstance(hist, BatchedHistory)

    def test_history_shapes_given_one_day(self):
        """24 hourly steps × 3 sims → time [24], fields [24, 3]."""
        hist = _run(_mars_batch(), duration=24 * HOUR)
        assert hist.time.shape == (24,)
        for t in (hist.surface_temperature, hist.surface_pressure,
                  hist.ice_mass, hist.solar_flux, hist.orbital_angle):
            assert t.shape == (24, 3)

    def test_remainder_step_appends_one_extra_row(self):
        """Duration not divisible by dt → one fractional step is recorded."""
        hist = _run(_mars_batch(), duration=3 * HOUR + 1800.0)
        assert hist.time.shape == (4,)

    def test_temperature_stays_physical_over_five_sols(self):
        """Invariant: T > 0 K and P >= 0 Pa at every recorded step."""
        hist = _run(_mars_batch(), duration=5 * SOL)
        assert torch.all(hist.surface_temperature > 0.0)
        assert torch.all(hist.surface_pressure >= 0.0)

    def test_rk4_accuracy_mode_runs(self):
        """ACCURATE (RK4) path produces the same shapes as FAST."""
        hist = _run(_mars_batch(), duration=12 * HOUR,
                    accuracy=Accuracy.ACCURATE)
        assert hist.surface_temperature.shape == (12, 3)


# ── BatchedHistory ────────────────────────────────────────────────────────────

class TestBatchedHistory:

    def test_B_property_matches_batch_size(self):
        hist = _run(_mars_batch(), duration=6 * HOUR)
        assert hist.B == 3

    def test_to_lists_shapes(self):
        """Legacy format: outer list of B, inner list of n_steps Snapshots."""
        hist = _run(_mars_batch(), duration=6 * HOUR)
        lists = hist.to_lists()
        assert len(lists) == 3
        assert all(len(inner) == 6 for inner in lists)
        assert isinstance(lists[0][0], Snapshot)

    def test_to_lists_values_match_tensors(self):
        """to_lists() must reproduce the tensor data exactly, per (sim, step)."""
        hist = _run(_mars_batch(), duration=6 * HOUR)
        lists = hist.to_lists()
        for i in range(hist.B):
            for s in (0, 3, 5):
                assert float(lists[i][s].surface_temperature) == pytest.approx(
                    float(hist.surface_temperature[s, i]))
                assert float(lists[i][s].surface_pressure) == pytest.approx(
                    float(hist.surface_pressure[s, i]))
                assert float(lists[i][s].ice_mass) == pytest.approx(
                    float(hist.ice_mass[s, i]))


# ── Batch self-consistency ────────────────────────────────────────────────────

class TestBatchConsistency:

    def test_identical_configs_produce_identical_columns(self):
        """Two batch elements with identical inputs must evolve identically.

        Regression guard for the recording-by-reference design: if any
        physics update ever mutates a recorded tensor in place, columns
        of identical sims would silently diverge or corrupt.
        """
        hist = _run([Mars(latitude=10.0, device="cpu"),
                     Mars(latitude=10.0, device="cpu")],
                    duration=2 * SOL)
        assert torch.equal(hist.surface_temperature[:, 0],
                           hist.surface_temperature[:, 1])
        assert torch.equal(hist.ice_mass[:, 0], hist.ice_mass[:, 1])

    def test_different_latitudes_diverge(self):
        """Sanity: latitude actually matters — equator ≠ 60°N history."""
        hist = _run([Mars(latitude=0.0, device="cpu"),
                     Mars(latitude=60.0, device="cpu")],
                    duration=2 * SOL)
        assert not torch.allclose(hist.surface_temperature[:, 0],
                                  hist.surface_temperature[:, 1])


# ── Chunk-compiled CUDA path ──────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestChunkedPathCuda:

    def test_chunked_matches_eager_given_fast_mode(self):
        """compile=True (chunked) must agree with compile=False to ~1e-9.

        5 sols = 123 steps: exercises full 64-step chunks, the eager tail,
        and the fractional remainder step. First call pays one-time JIT.
        """
        def batch():
            return [Mars(latitude=lat, device="cuda:0")
                    for lat in (-30.0, 20.0)]

        ref = _run(batch(), duration=5 * SOL, compile=False)
        chk = _run(batch(), duration=5 * SOL, compile=True)
        torch.cuda.synchronize()

        for a, b in ((chk.surface_temperature, ref.surface_temperature),
                     (chk.surface_pressure, ref.surface_pressure),
                     (chk.ice_mass, ref.ice_mass)):
            rel = ((a - b).abs() / b.abs().clamp(min=1.0)).max().item()
            assert rel < 1e-9

    def test_custom_chunk_size_matches_eager(self):
        """A non-default chunk size must still produce correct results.

        Uses chunk=32 (< the 128-step run) so multiple compiled chunks plus
        an eager tail are exercised with a tuned chunk value.
        """
        def batch():
            return [Mars(latitude=lat, device="cuda:0")
                    for lat in (-30.0, 20.0)]

        ref = _run(batch(), duration=5 * SOL, compile=False)
        chk = _run(batch(), duration=5 * SOL, compile=True, chunk=32)
        torch.cuda.synchronize()

        rel = ((chk.surface_temperature - ref.surface_temperature).abs()
               / ref.surface_temperature.abs().clamp(min=1.0)).max().item()
        assert rel < 1e-9

    def test_chunked_matches_eager_given_accurate_mode(self):
        """RK4 (ACCURATE) chunked path must also agree with eager to ~1e-9.

        Same comparison as the FAST case but for the 4-sample RK4 recipe.
        The compiled RK4 graph is ~4x larger, so the one-time JIT here is
        the slowest part of the suite (hence @pytest.mark.slow).
        """
        def batch():
            return [Mars(latitude=lat, device="cuda:0")
                    for lat in (-30.0, 20.0)]

        ref = _run(batch(), duration=5 * SOL,
                   accuracy=Accuracy.ACCURATE, compile=False)
        chk = _run(batch(), duration=5 * SOL,
                   accuracy=Accuracy.ACCURATE, compile=True)
        torch.cuda.synchronize()

        for a, b in ((chk.surface_temperature, ref.surface_temperature),
                     (chk.surface_pressure, ref.surface_pressure),
                     (chk.ice_mass, ref.ice_mass)):
            rel = ((a - b).abs() / b.abs().clamp(min=1.0)).max().item()
            assert rel < 1e-9
