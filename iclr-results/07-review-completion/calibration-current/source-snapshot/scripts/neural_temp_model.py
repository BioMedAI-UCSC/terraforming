"""Fixed feature schema and small temperature-only postprocessors (no GCM calls)."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

SCHEMA = [
    ("start_t", "K"), ("start_surface_t", "K"), ("start_next_t", "K"),
    ("start_u", "m/s"), ("start_v", "m/s"), ("start_pressure", "Pa"), ("start_dust", "1"),
    ("forecast_t", "K"), ("forecast_delta_t", "K"), ("forecast_surface_t", "K"),
    ("forecast_surface_air_delta", "K"), ("forecast_vertical_delta", "K"),
    ("forecast_u", "m/s"), ("forecast_v", "m/s"), ("forecast_pressure", "Pa"),
    ("insolation", "W/m2"), ("sin_lst", "1"), ("cos_lst", "1"),
    ("sin_ls", "1"), ("cos_ls", "1"), ("sin_lat", "1"),
    ("sin_lon", "1"), ("cos_lon", "1"), ("terrain", "m"),
]
SCHEMA = [list(x) for x in SCHEMA]
NO_GCM = [i for i, (name, _) in enumerate(SCHEMA) if not name.startswith("forecast_")]


def build_features(fields):
    missing = {n for n, _ in SCHEMA} - fields.keys()
    if missing:
        raise ValueError(f"missing required predictors: {sorted(missing)}")
    result = np.stack(np.broadcast_arrays(*[np.asarray(fields[n], dtype=float) for n, _ in SCHEMA]), axis=-1)
    if not np.isfinite(result).all():
        raise ValueError("nonfinite predictors")
    return result


def normalized_weights(weights):
    weights = np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights.sum(axis=-1) <= 0):
        raise ValueError("invalid quadrature weights")
    return weights / weights.sum(axis=-1, keepdims=True)


def fit_normalization(x, y, base, weights, split):
    if split != "train":
        raise ValueError("normalization may only fit training starts")
    w = normalized_weights(weights) / len(x)
    mean = np.sum(w[..., None] * x, axis=(0, 1))
    scale = np.maximum(np.sqrt(np.sum(w[..., None] * (x - mean)**2, axis=(0, 1))), 1e-6)
    rms = max(float(np.sqrt(np.sum(w * (y - base)**2))), 1e-6)
    return {"mean": mean.tolist(), "scale": scale.tolist(), "residual_rms": rms,
            "feature_schema": SCHEMA, "fitted_split": "train"}


def standardize(x, norm, columns=None):
    z = (x - np.asarray(norm["mean"])) / np.asarray(norm["scale"])
    return z if columns is None else z[..., columns]


class ResidualMLP(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.double()

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_controls(x, y, base, weights, norm, split):
    if split != "train":
        raise ValueError("controls may only fit training starts")
    z = standardize(x, norm).reshape(-1, x.shape[-1])
    a = np.column_stack([np.ones(len(z)), z])
    w = (normalized_weights(weights) / len(x)).ravel()
    residual = (y - base).ravel()
    penalty = np.eye(a.shape[1]) * .001
    penalty[0, 0] = 0
    beta = np.linalg.solve(a.T @ (w[:, None] * a) + penalty, a.T @ (w * residual))
    return {"constant": float(w @ residual), "ridge": beta.tolist(), "fitted_split": "train"}


def predict_neural(checkpoint, x):
    columns = checkpoint["columns"]
    model = ResidualMLP(len(columns))
    model.load_state_dict(checkpoint["parameters"])
    with torch.no_grad():
        z = torch.as_tensor(standardize(x, checkpoint["normalization"], columns))
        correction = model(z).numpy() * checkpoint["normalization"]["residual_rms"]
    return x[..., 0 if checkpoint["kind"] == "no_gcm" else 7] + correction
