"""Calibration & validation of the Mars simulator against observed climatology.

Compares the 0-D model's seasonal cycle to reference Mars climatology — primarily
the **Viking Lander 1 surface-pressure annual cycle** (the canonical CO₂-cycle
benchmark, latitude-matched to the model's 22°N default and reproduced by MCD
6.1) — reports the standard model-vs-observation metrics, and tunes a named,
auditable subset of the model's free physics parameters.

    from src.calibration import get_reference, simulate_seasonal_cycle, evaluate
    ref = get_reference("vl1")
    cycle = simulate_seasonal_cycle(ref)
    metrics = evaluate(cycle, ref, field="pressure")
"""

from __future__ import annotations

from src.calibration.harness import (
    CalibrationResult,
    SeasonalCycle,
    calibrate,
    evaluate,
    simulate_seasonal_cycle,
)
from src.calibration.metrics import CycleMetrics, compute_metrics
from src.calibration.reference import (
    REFERENCES,
    ReferenceClimatology,
    get_reference,
    viking_lander_1,
    viking_lander_2,
)

__all__ = [
    "ReferenceClimatology",
    "get_reference",
    "viking_lander_1",
    "viking_lander_2",
    "REFERENCES",
    "SeasonalCycle",
    "simulate_seasonal_cycle",
    "evaluate",
    "calibrate",
    "CalibrationResult",
    "CycleMetrics",
    "compute_metrics",
]
