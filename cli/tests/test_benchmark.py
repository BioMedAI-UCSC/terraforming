"""Tests for the `tform benchmark` command and cli.runner.run_gpu_benchmark.

Covers:
  - run_gpu_benchmark: row structure, CPU-only path (use_gpu=False)
  - benchmark_cmd: happy path (--no-gpu), invalid --batch, empty --batch

All tests force CPU-only (``--no-gpu`` / ``use_gpu=False``) and use tiny
batches and durations so they run in well under a second and require no GPU.
"""

from __future__ import annotations

from click.testing import CliRunner

from cli.main import cli
from cli.models import Accuracy
from cli.runner import run_gpu_benchmark

# Short enough to be a handful of hourly steps.
_SHORT_DURATION = 3600.0 * 5   # 5 hours
_SHORT_WARMUP = 3600.0 * 2     # 2 hours


# ── run_gpu_benchmark ─────────────────────────────────────────────────────────

class TestRunGpuBenchmark:

    def test_returns_one_row_per_batch_size(self):
        rows = run_gpu_benchmark([1, 2], _SHORT_DURATION, _SHORT_WARMUP,
                                 accuracy=Accuracy.fast, use_gpu=False)
        assert [r["B"] for r in rows] == [1, 2]

    def test_cpu_only_row_has_expected_shape(self):
        """With use_gpu=False, CPU time is recorded and GPU fields are None."""
        rows = run_gpu_benchmark([2], _SHORT_DURATION, _SHORT_WARMUP,
                                 accuracy=Accuracy.fast, use_gpu=False)
        row = rows[0]
        assert row["cpu_s"] > 0.0
        assert row["gpu_s"] is None
        assert row["jit_s"] is None
        assert row["speedup"] is None
        assert row["status"] == "CPU only"


# ── tform benchmark ───────────────────────────────────────────────────────────

class TestBenchmarkCommand:

    def test_no_gpu_run_exits_zero(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["benchmark", "--batch", "2", "--years", "0.0005", "--no-gpu"])
        assert result.exit_code == 0
        assert "CPU (s)" in result.output   # the table header printed

    def test_invalid_batch_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--batch", "abc", "--no-gpu"])
        assert result.exit_code != 0

    def test_empty_batch_exits_nonzero(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--batch", "", "--no-gpu"])
        assert result.exit_code != 0
