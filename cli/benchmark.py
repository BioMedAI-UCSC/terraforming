"""Benchmark runner for Mars simulations."""

from __future__ import annotations

import csv
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import click

from cli import config_loader
from cli import output as out_mod
from cli import runner
from cli.models import Accuracy, ExpType, RunFlags


DEFAULT_SOL_COUNTS = [7, 30, 180, 600, 5000, 10000]
SECONDS_PER_SOL = 88775.244


def run_benchmarks(
    sol_counts: list[int] | None = None,
    accuracy: Accuracy = Accuracy.fast,
) -> Path:
    """Run all requested sol benchmarks and save one benchmark session."""

    counts = sol_counts or DEFAULT_SOL_COUNTS

    # Every execution receives its own timestamped session folder.
    session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    session_dir = Path("outputs") / "benchmarks" / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    click.echo()
    click.echo(f"  Benchmark session: {session_id}")
    click.echo(f"  Sol counts: {', '.join(str(count) for count in counts)}")
    click.echo(f"  Accuracy: {accuracy.value}")
    click.echo()

    summaries: list[dict[str, object]] = []

    for sols in counts:
        run_dir = session_dir / f"{sols}-sols"

        click.echo(f"  Running {sols:,} sols...")

        base_cfg = config_loader.load(
            planet="mars",
            preset="current-mars",
        )

        flags = RunFlags(
            exp_type=ExpType.sol,
            sols=float(sols),
            accuracy=accuracy,
            plot=True,
            output_dir=str(run_dir),
        )

        cfg = config_loader.merge_overrides(base_cfg, flags)

        started = time.perf_counter()

        try:
            results = runner.run_sol(cfg)
            runtime_seconds = time.perf_counter() - started

            # Existing output system saves the full CSV and all T/P/M plots.
            out_mod.dispatch(results, cfg)

            history = results[0].history
            final_state = history[-1]
            step_count = len(history)

            temperature_daily = _extract_daily_series(
                history,
                "surface_temperature",
            )
            pressure_daily = _extract_daily_series(
                history,
                "surface_pressure",
            )
            ice_daily = _extract_daily_series(
                history,
                "ice_mass",
            )

            temperature_stability = _compute_stability(temperature_daily)
            pressure_stability = _compute_stability(pressure_daily)
            ice_stability = _compute_stability(ice_daily)



            summary = {
        
                "sols": sols,
                "runtime_seconds": runtime_seconds,
                "step_count": step_count,
                "runtime_per_step_seconds": (
                    runtime_seconds / step_count if step_count else None
                ),
                "status": "success",

                "temperature_k": _history_value(
                    final_state,
                    "surface_temperature",
                ),
                "temperature_tail_mean_k": temperature_stability["mean"],
                "temperature_tail_std_k": temperature_stability["std"],
                "temperature_trend_k_per_sol": temperature_stability["trend_per_sol"],
                "temperature_daily_samples": len(temperature_daily),

                "pressure_pa": _history_value(
                    final_state,
                    "surface_pressure",
                ),
                "pressure_tail_mean_pa": pressure_stability["mean"],
                "pressure_tail_std_pa": pressure_stability["std"],
                "pressure_trend_pa_per_sol": pressure_stability["trend_per_sol"],
                "pressure_daily_samples": len(pressure_daily),

                "ice_mass_kg": _history_value(
                    final_state,
                    "ice_mass",
                ),
                "ice_tail_mean_kg": ice_stability["mean"],
                "ice_tail_std_kg": ice_stability["std"],
                "ice_trend_kg_per_sol": ice_stability["trend_per_sol"],
                "ice_daily_samples": len(ice_daily),

                "output_directory": str(run_dir),
                "error": None,
            }

            click.echo(
                f"  ✓ {sols:,} sols completed in "
                f"{runtime_seconds:.3f} seconds"
            )

        except Exception as exc:
            runtime_seconds = time.perf_counter() - started

            summary = {
                "sols": sols,
                "runtime_seconds": runtime_seconds,
                "status": "failed",

                "temperature_k": None,
                "temperature_tail_mean_k": None,
                "temperature_tail_std_k": None,
                "temperature_trend_k_per_sol": None,
                "temperature_daily_samples": None,

                "pressure_pa": None,
                "pressure_tail_mean_pa": None,
                "pressure_tail_std_pa": None,
                "pressure_trend_pa_per_sol": None,
                "pressure_daily_samples": None,

                "ice_mass_kg": None,
                "ice_tail_mean_kg": None,
                "ice_tail_std_kg": None,
                "ice_trend_kg_per_sol": None,
                "ice_daily_samples": None,

                "step_count": None,
                "runtime_per_step_seconds": None,

                "output_directory": str(run_dir),
                "error": str(exc),
            }

            click.echo(f"  ✗ {sols:,} sols failed: {exc}")

        summaries.append(summary)
        click.echo()

    _save_session_metadata(
        session_dir=session_dir,
        session_id=session_id,
        accuracy=accuracy,
        summaries=summaries,
    )
    _save_summary_csv(session_dir, summaries)

    from cli.benchmark_compare import compare_with_previous

    compare_with_previous(session_dir)

    click.echo(f"  Benchmark complete: {session_dir}")
    click.echo()

    return session_dir


def _history_value(state: object, *names: str) -> float | None:
    """Read a numeric value from a history object or dictionary."""

    for name in names:
        if isinstance(state, dict) and name in state:
            return _to_float(state[name])

        if hasattr(state, name):
            return _to_float(getattr(state, name))

    return None


def _to_float(value: object) -> float | None:
    """Convert regular values or scalar tensors into floats."""

    if value is None:
        return None

    if hasattr(value, "item"):
        value = value.item()

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_daily_series(
    history: list[object],
    field_name: str,
) -> list[float]:
    """Convert hourly snapshots into one mean value per complete sol."""

    buckets: dict[int, list[float]] = {}

    for snapshot in history:
        time_seconds = _history_value(snapshot, "time")
        value = _history_value(snapshot, field_name)

        if time_seconds is None or value is None:
            continue

        sol_index = int(time_seconds // SECONDS_PER_SOL)
        buckets.setdefault(sol_index, []).append(value)

    if not buckets:
        return []

    # A snapshot can occur exactly at the end of the simulation, creating a
    # final bucket containing only one or two points. Estimate the normal
    # number of samples per complete sol and discard substantially incomplete
    # buckets.
    bucket_sizes = [len(values) for values in buckets.values()]
    typical_size = statistics.median(bucket_sizes)
    minimum_complete_size = max(1, math.ceil(typical_size * 0.5))

    daily_means: list[float] = []

    for sol_index in sorted(buckets):
        values = buckets[sol_index]

        if len(values) < minimum_complete_size:
            continue

        daily_means.append(statistics.fmean(values))

    return daily_means


def _compute_stability(
    values: list[float],
    tail_fraction: float = 0.10,
    minimum_tail_points: int = 3,
) -> dict[str, float | int | None]:
    """Measure variability and linear trend over the final daily values."""

    if not values:
        return {
            "mean": None,
            "std": None,
            "trend_per_sol": None,
            "tail_points": 0,
        }

    tail_size = max(
        minimum_tail_points,
        math.ceil(len(values) * tail_fraction),
    )
    tail_size = min(tail_size, len(values))
    tail = values[-tail_size:]

    mean_value = statistics.fmean(tail)
    std_value = statistics.pstdev(tail) if len(tail) > 1 else 0.0

    if len(tail) < 2:
        trend_per_sol = None
    else:
        # Consecutive daily means are one sol apart.
        sol_positions = list(range(len(tail)))
        mean_position = statistics.fmean(sol_positions)

        numerator = sum(
            (position - mean_position) * (value - mean_value)
            for position, value in zip(sol_positions, tail)
        )
        denominator = sum(
            (position - mean_position) ** 2
            for position in sol_positions
        )

        trend_per_sol = numerator / denominator if denominator else 0.0

    return {
        "mean": mean_value,
        "std": std_value,
        "trend_per_sol": trend_per_sol,
        "tail_points": len(tail),
    }

def _save_summary_csv(
    session_dir: Path,
    summaries: list[dict[str, object]],
) -> None:
    """Save one row per benchmark duration."""

    columns = [
        "sols",
        "runtime_seconds",
        "step_count",
        "runtime_per_step_seconds",
        "status",

        "temperature_k",
        "temperature_tail_mean_k",
        "temperature_tail_std_k",
        "temperature_trend_k_per_sol",
        "temperature_daily_samples",

        "pressure_pa",
        "pressure_tail_mean_pa",
        "pressure_tail_std_pa",
        "pressure_trend_pa_per_sol",
        "pressure_daily_samples",

        "ice_mass_kg",
        "ice_tail_mean_kg",
        "ice_tail_std_kg",
        "ice_trend_kg_per_sol",
        "ice_daily_samples",

        "output_directory",
        "error",
    ]

    with (session_dir / "summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

        for summary in summaries:
            writer.writerow(
                {column: summary.get(column) for column in columns}
            )


def _find_previous_session(
    benchmarks_dir: Path,
    current_session_id: str,
) -> Path | None:
    """Find the newest completed benchmark session before the current one."""

    candidates = []

    for path in benchmarks_dir.iterdir():
        if not path.is_dir():
            continue

        if path.name == current_session_id:
            continue

        # Only compare against completed sessions.
        if not (path / "summary.csv").exists():
            continue

        candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.name)



def _save_session_metadata(
    session_dir: Path,
    session_id: str,
    accuracy: Accuracy,
    summaries: list[dict[str, object]],
) -> None:
    """Save reproducibility information for this benchmark session."""

    metadata = {
        "session_id": session_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preset": "current-mars",
        "accuracy": accuracy.value,
        "python_version": sys.version,
        "platform": platform.platform(),
        "results": summaries,
    }

    with (session_dir / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=2)