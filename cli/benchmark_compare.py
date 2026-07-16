"""Compare completed benchmark sessions."""

from __future__ import annotations

import csv
from pathlib import Path

import click


def compare_with_previous(current_session_dir: Path) -> Path | None:
    """Compare the current benchmark with the newest earlier completed session."""

    previous_session = _find_previous_session(current_session_dir)

    if previous_session is None:
        click.echo("  Previous session: none")
        return None

    current_rows = _load_summary(current_session_dir / "summary.csv")
    previous_rows = _load_summary(previous_session / "summary.csv")

    click.echo()
    click.echo(f"  Previous session: {previous_session.name}")
    click.echo("  Session comparison:")

    matching_sols = sorted(set(current_rows) & set(previous_rows))

    if not matching_sols:
        click.echo("    No matching sol counts were found.")
        return previous_session

    for sols in matching_sols:
        current = current_rows[sols]
        previous = previous_rows[sols]

        click.echo()
        click.echo(f"    {sols:,} sols")

        _print_runtime_change(current, previous)
        _print_metric_change(
            label="Temperature trend",
            current=current,
            previous=previous,
            field="temperature_trend_k_per_sol",
            unit="K/sol",
        )
        _print_metric_change(
            label="Pressure trend",
            current=current,
            previous=previous,
            field="pressure_trend_pa_per_sol",
            unit="Pa/sol",
        )
        _print_metric_change(
            label="Ice trend",
            current=current,
            previous=previous,
            field="ice_trend_kg_per_sol",
            unit="kg/sol",
        )

    click.echo()
    return previous_session


def _find_previous_session(current_session_dir: Path) -> Path | None:
    """Return the newest completed session before the current session."""

    benchmarks_dir = current_session_dir.parent
    candidates: list[Path] = []

    for path in benchmarks_dir.iterdir():
        if not path.is_dir():
            continue

        if path == current_session_dir:
            continue

        if not (path / "summary.csv").exists():
            continue

        candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.name)


def _load_summary(path: Path) -> dict[int, dict[str, str]]:
    """Load summary rows and index them by sol count."""

    rows: dict[int, dict[str, str]] = {}

    with path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("status") != "success":
                continue

            try:
                sols = int(float(row["sols"]))
            except (KeyError, TypeError, ValueError):
                continue

            rows[sols] = row

    return rows


def _read_float(row: dict[str, str], field: str) -> float | None:
    """Read one optional numeric CSV field."""

    raw = row.get(field)

    if raw in (None, ""):
        return None

    try:
        return float(raw)
    except ValueError:
        return None


def _print_runtime_change(
    current: dict[str, str],
    previous: dict[str, str],
) -> None:
    """Print runtime and percentage change."""

    current_runtime = _read_float(current, "runtime_seconds")
    previous_runtime = _read_float(previous, "runtime_seconds")

    if current_runtime is None or previous_runtime is None:
        click.echo("      Runtime: unavailable")
        return

    if previous_runtime == 0:
        click.echo(
            f"      Runtime: {previous_runtime:.3f}s → "
            f"{current_runtime:.3f}s"
        )
        return

    percent_change = (
        (current_runtime - previous_runtime) / previous_runtime
    ) * 100

    if percent_change < 0:
        description = f"{abs(percent_change):.1f}% faster"
    elif percent_change > 0:
        description = f"{percent_change:.1f}% slower"
    else:
        description = "no change"

    click.echo(
        f"      Runtime: {previous_runtime:.3f}s → "
        f"{current_runtime:.3f}s  ({description})"
    )


def _print_metric_change(
    label: str,
    current: dict[str, str],
    previous: dict[str, str],
    field: str,
    unit: str,
) -> None:
    """Print the old and new values for one benchmark metric."""

    current_value = _read_float(current, field)
    previous_value = _read_float(previous, field)

    if current_value is None or previous_value is None:
        click.echo(f"      {label}: unavailable")
        return

    difference = current_value - previous_value

    click.echo(
        f"      {label}: {previous_value:.4g} → "
        f"{current_value:.4g} {unit}  "
        f"(Δ {difference:+.4g})"
    )