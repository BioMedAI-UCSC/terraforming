"""Optional JAX neural physics components; install terraforming[gcm3d].

Importing the ordinary framework does not import this package or a neural model.
No additional neural-library dependency is required beyond the GCM's JAX stack.
"""
from .models import ColumnMLP, Normalization, fit_normalization
from .radiation import (
    ColumnInputs, NeuralRadiation, column_features, feature_schema, heating_rates,
    inputs_from_state, radiation_budget_residual, reference_fluxes,
)
from .checkpoints import load_checkpoint, save_checkpoint
from .training import adam_init, adam_update, directional_gradient_check, weighted_mse

__all__ = [
    "ColumnMLP", "Normalization", "fit_normalization", "ColumnInputs", "NeuralRadiation",
    "column_features", "feature_schema", "heating_rates", "inputs_from_state",
    "radiation_budget_residual", "reference_fluxes", "load_checkpoint", "save_checkpoint",
    "adam_init", "adam_update", "directional_gradient_check", "weighted_mse",
]
