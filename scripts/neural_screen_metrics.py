"""Paired, equal-weight forecast ensembling; no fitting on validation targets."""
from pathlib import Path
import json

import numpy as np


def forecast_scores(prediction, target, mass, field_scale):
    """Match HybridExperiment's per-example mass-weighted standardized loss."""
    prediction, target, mass, scale = map(np.asarray, (prediction, target, mass, field_scale))
    if prediction.ndim != 6 or prediction.shape != target.shape or prediction.shape[2] != 3:
        raise ValueError("expected matching [case, time, 3, layer, lon, lat] forecasts")
    if mass.shape != prediction.shape[:2] + prediction.shape[3:] or scale.shape != prediction.shape[2:4]:
        raise ValueError("ensemble mass/scale shapes differ from forecasts")
    if not all(np.isfinite(x).all() for x in (prediction, target, mass, scale)) or np.any(mass <= 0) or np.any(scale <= 0):
        raise ValueError("nonfinite forecasts or nonpositive ensemble weights/scales")
    error = (prediction - target) / scale[None, None, :, :, None, None]
    case_losses = np.sum(error**2 * mass[:, :, None], axis=(1, 2, 3, 4, 5)) / (3 * np.sum(mass, axis=(1, 2, 3, 4)))
    numerator = np.sum((prediction - target)**2 * mass[:, :, None], axis=(1, 4, 5))
    denominator = np.sum(mass, axis=(1, 3, 4))[:, None, :]
    return dict(neural_loss=float(np.mean(case_losses)), case_losses=case_losses.tolist(),
                rmse_by_field_layer=np.sqrt(np.mean(numerator / denominator, axis=0)).tolist())


def ensemble_report(paths):
    """Reject mismatched steps, cases, physics, normalization or validation data."""
    if len(paths) != 4:
        raise ValueError("the screen ensemble requires all four methods")
    artifacts = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            artifacts.append({key: archive[key] for key in archive.files})
    first = artifacts[0]
    for item in artifacts[1:]:
        for key in ("step", "metadata", "target", "mass", "field_scale", "physical_losses"):
            if not np.array_equal(first[key], item[key]):
                raise ValueError(f"unpaired ensemble artifacts: {key}")
    # Validate each member as well as the mean, including shape and finiteness.
    members = [forecast_scores(item["prediction"], item["target"], item["mass"], item["field_scale"])
               for item in artifacts]
    latent = [np.asarray(item["latent_losses"]) for item in artifacts]
    if any(loss.shape != (len(first["prediction"]),) or not np.isfinite(loss).all() or np.any(loss < 0) for loss in latent):
        raise ValueError("invalid member internal-state scores")
    mean = np.mean([item["prediction"] for item in artifacts], axis=0)
    report = forecast_scores(mean, first["target"], first["mass"], first["field_scale"])
    physical = np.asarray(first["physical_losses"])
    if physical.shape != (len(mean),) or not np.isfinite(physical).all() or np.any(physical < 0):
        raise ValueError("invalid paired physical baseline")
    baseline = float(physical.mean())
    report.update(step=int(first["step"]), examples=len(mean), physical_loss=baseline,
                  improvement_percent=100 * (baseline - report["neural_loss"]) / baseline if baseline else None,
                  paired_improvements=(physical - report["case_losses"]).tolist(),
                  method="equal-weight forecast ensemble", weights=[.25] * 4,
                  member_losses=[member["neural_loss"] for member in members],
                  member_latent_losses=[float(loss.mean()) for loss in latent],
                  artifacts=[str(Path(path)) for path in paths],
                  comparison=json.loads(str(first["metadata"])),
                  scope="Validation screening; averaged outputs are not a single coupled GCM state.")
    return report
