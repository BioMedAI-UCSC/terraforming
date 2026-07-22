"""Metrics for comparing a modelled seasonal cycle to a reference climatology.

The quantities a climate modeller actually looks at when overlaying a model on a
reference curve: the mean level, the amplitude of the seasonal swing, the phase
(does the model peak at the right season?), and the overall RMS misfit.

All functions take equal-length arrays sampled on the same Ls grid.
"""

from __future__ import annotations

import dataclasses

import numpy as np


def annual_mean(values: np.ndarray) -> float:
    """Annual mean of a seasonally-sampled quantity."""
    return float(np.mean(values))


def peak_to_peak_amplitude(values: np.ndarray) -> float:
    """Peak-to-peak (max − min) seasonal amplitude."""
    return float(np.max(values) - np.min(values))


def rmse(model: np.ndarray, reference: np.ndarray) -> float:
    """Root-mean-square error between model and reference."""
    return float(np.sqrt(np.mean((np.asarray(model) - np.asarray(reference)) ** 2)))


def bias(model: np.ndarray, reference: np.ndarray) -> float:
    """Mean signed error (model − reference); the systematic offset."""
    return float(np.mean(np.asarray(model) - np.asarray(reference)))


def amplitude_ratio(model: np.ndarray, reference: np.ndarray) -> float:
    """Model seasonal amplitude ÷ reference amplitude (1.0 = perfect)."""
    ref_amp = peak_to_peak_amplitude(reference)
    if ref_amp == 0.0:
        raise ValueError("reference has zero amplitude; ratio undefined")
    return peak_to_peak_amplitude(model) / ref_amp


def phase_lag_deg(ls_deg: np.ndarray, model: np.ndarray, reference: np.ndarray) -> float:
    """Seasonal phase lag (degrees of Ls) that best aligns model to reference.

    Circular cross-correlation of the mean-removed cycles over integer index
    shifts; the returned lag is the shift (in Ls degrees, wrapped to (−180, 180])
    maximising the correlation. Positive = model peaks *later* in the season than
    the reference. Requires an evenly-spaced, full-circle Ls grid.
    """
    ls = np.asarray(ls_deg, dtype=float)
    n = len(ls)
    step = 360.0 / n
    m = np.asarray(model, dtype=float) - np.mean(model)
    r = np.asarray(reference, dtype=float) - np.mean(reference)
    corr = np.array([np.sum(np.roll(m, k) * r) for k in range(n)])
    k_best = int(np.argmax(corr))
    # corr peaks at k = -s (mod n) when model = reference rolled later by s bins;
    # recover s so a model peaking later than the reference gives a positive lag.
    s = (n - k_best) % n
    lag = s * step
    if lag > 180.0:
        lag -= 360.0
    return float(lag)


@dataclasses.dataclass(frozen=True)
class CycleMetrics:
    """Summary of how a modelled seasonal cycle compares to a reference."""

    model_mean: float
    reference_mean: float
    bias: float
    model_amplitude: float
    reference_amplitude: float
    amplitude_ratio: float
    phase_lag_deg: float
    rmse: float

    def as_row(self) -> dict[str, float]:
        return dataclasses.asdict(self)


def compute_metrics(
    ls_deg: np.ndarray, model: np.ndarray, reference: np.ndarray
) -> CycleMetrics:
    """Bundle all seasonal-cycle metrics for a model-vs-reference comparison."""
    return CycleMetrics(
        model_mean=annual_mean(model),
        reference_mean=annual_mean(reference),
        bias=bias(model, reference),
        model_amplitude=peak_to_peak_amplitude(model),
        reference_amplitude=peak_to_peak_amplitude(reference),
        amplitude_ratio=amplitude_ratio(model, reference),
        phase_lag_deg=phase_lag_deg(ls_deg, model, reference),
        rmse=rmse(model, reference),
    )
