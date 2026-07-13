"""Benchmark runner for Mars simulations."""

from __future__ import annotations

import csv
import json
import platform
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


def run_benchmarks(
    sol_counts: list[int] | None = None,
    accuracy: Accuracy = Accuracy.fast,
) -> Path:
    """Run all requested sol benchmarks and save one benchmark session."""

    counts = sol_counts or DEFAULT_SOL_COUNTS

    # Every execution receives its own timestamped session folder
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

            # Existing output system saves the full CSV and all T/P/M plots
            out_mod.dispatch(results, cfg)

            history = results[0].history
            final_state = history[-1]

            summary = {
                "sols": sols,
                "runtime_seconds": runtime_seconds,
                "status": "success",
                "temperature_k": _history_value(
                    final_state,
                    "temperature_k",
                    "surface_temperature",
                    "temperature",
                ),
                "pressure_pa": _history_value(
                    final_state,
                    "pressure_pa",
                    "surface_pressure",
                    "pressure",
                ),
                "ice_mass_kg": _history_value(
                    final_state,
                    "ice_mass_kg",
                    "ice_mass",
                ),
                "output_directory": str(run_dir),
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
                "pressure_pa": None,
                "ice_mass_kg": None,
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


def _save_summary_csv(
    session_dir: Path,
    summaries: list[dict[str, object]],
) -> None:
    """Save one row per benchmark duration."""

    columns = [
        "sols",
        "runtime_seconds",
        "status",
        "temperature_k",
        "pressure_pa",
        "ice_mass_kg",
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