"""Bounded calibration and paired-block uncertainty, independent of the GCM."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import minimize


def projected_gradient(x, gradient):
    """Gradient projected onto the feasible directions of the unit box."""
    x, gradient = np.asarray(x), np.asarray(gradient)
    return np.where(((x <= 0) & (gradient > 0)) | ((x >= 1) & (gradient < 0)), 0., gradient)


def bounded_fit(value_gradient, validation, start, *, maxiter=500, maxeval=1000,
                gtol=1e-6, ftol=1e-12, starts=1, seed=0, emit=lambda row, x, best: None):
    """Train only on the training oracle; select checkpoints using validation.

    The evaluation ceiling is global across starts. Every evaluated point and
    gradient is emitted, including the initial checkpoint and line-search trials.
    A successful relative-loss stop is recorded separately from stationarity.
    """
    start = np.asarray(start, dtype=float)
    if (start.shape != (3,) or not np.isfinite(start).all() or np.any((start < 0) | (start > 1))
            or min(maxiter, maxeval, starts) < 1 or not np.isfinite([gtol, ftol]).all()
            or min(gtol, ftol) <= 0):
        raise ValueError("finite three-control start, positive budgets and tolerances required")
    rng = np.random.default_rng(seed)
    initial_points = [start, *rng.uniform(.1, .9, size=(starts-1, 3))]
    best, best_val = start.copy(), float(validation(start))
    if not np.isfinite(best_val):
        raise FloatingPointError("nonfinite initial validation loss")
    trace, terminations = [], []

    class BudgetReached(Exception):
        pass

    for start_index, initial in enumerate(initial_points):
        if len(trace) >= maxeval:
            break

        def oracle(x):
            nonlocal best, best_val
            if len(trace) >= maxeval:
                raise BudgetReached
            value, gradient = value_gradient(x)
            value, gradient = float(value), np.asarray(gradient, dtype=float)
            val = float(validation(x))
            if not np.isfinite([value, val]).all() or not np.isfinite(gradient).all():
                raise FloatingPointError("nonfinite calibration loss or gradient")
            if val < best_val:
                best, best_val = x.copy(), val
            row = dict(evaluation=len(trace)+1, start=start_index,
                       train_loss=value, validation_loss=val,
                       gradient_norm=float(np.linalg.norm(gradient)),
                       projected_gradient_inf=float(np.max(np.abs(projected_gradient(x, gradient)))))
            for i in range(3):
                row[f"normalized_control_{i}"] = float(x[i])
                row[f"gradient_{i}"] = float(gradient[i])
            trace.append(row)
            emit(row, x.copy(), best.copy())
            return value, gradient

        try:
            result = minimize(oracle, initial, jac=True, method="L-BFGS-B", bounds=[(0.,1.)]*3,
                              options=dict(maxiter=maxiter, maxfun=maxeval-len(trace),
                                           gtol=gtol, ftol=ftol, maxls=30))
            norm = float(np.max(np.abs(projected_gradient(result.x, result.jac))))
            terminations.append(dict(start=start_index, success=bool(result.success),
                message=str(result.message), iterations=int(result.nit),
                projected_gradient_inf=norm, stationary=bool(norm <= gtol),
                final_normalized_controls=result.x.tolist()))
        except BudgetReached:
            terminations.append(dict(start=start_index, success=False, stationary=False,
                                     message="global evaluation budget exhausted"))
            break
    report = dict(optimizer="L-BFGS-B", evaluations=len(trace), starts_requested=starts,
                  starts_completed=len(terminations), max_evaluations=maxeval,
                  gtol=gtol, ftol=ftol, terminations=terminations,
                  stationary=any(t["stationary"] for t in terminations),
                  validation_selected_normalized_controls=best.tolist(),
                  validation_loss=best_val,
                  selection="validation checkpoint may differ from the final training optimum")
    return best, report, trace


def paired_block_interval(baseline_mse, fitted_mse, blocks, *, draws=10000, seed=0):
    """Resample whole temporal blocks jointly; never use grid cells as replicas.

    Exact enumeration is used for <=6 blocks. This conditional interval describes
    the supplied evaluation blocks, not optimizer/seed or climatological uncertainty.
    """
    base, fit, blocks = np.asarray(baseline_mse), np.asarray(fitted_mse), np.asarray(blocks)
    if (base.ndim != 1 or fit.shape != base.shape or blocks.shape != base.shape or not len(base)
            or not np.isfinite(base).all() or not np.isfinite(fit).all() or min(base.min(), fit.min()) < 0):
        raise ValueError("paired finite nonnegative per-window MSE and block IDs required")
    labels = np.unique(blocks)
    gain = float(np.sqrt(base.mean())-np.sqrt(fit.mean()))
    if len(labels) < 2:
        return dict(gain_k=gain, blocks=len(labels), interval_95_k=None,
                    limitation="at least two temporal blocks are needed")
    sums = np.array([[base[blocks==b].sum(), fit[blocks==b].sum(), np.sum(blocks==b)] for b in labels])
    if len(labels) <= 6:
        samples = np.asarray(list(itertools.product(range(len(labels)), repeat=len(labels))))
    else:
        samples = np.random.default_rng(seed).integers(0, len(labels), size=(draws, len(labels)))
    totals = sums[samples].sum(axis=1)
    gains = np.sqrt(totals[:,0]/totals[:,2])-np.sqrt(totals[:,1]/totals[:,2])
    return dict(gain_k=gain, blocks=len(labels), resamples=len(gains),
                interval_95_k=np.quantile(gains,[.025,.975]).tolist(),
                limitation="conditional paired-block percentile interval; few years, one fitted run, no population-skill claim")
