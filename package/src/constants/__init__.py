"""Project-wide constants and TensorFlow dtype configuration.

This module is the single source of truth for:
    • The project PyTorch dtype (``TF_DTYPE = torch.float64``)
    • The ``_c()`` helper for creating typed scalar tensors
    • Universal physical constants (SI units, as ``torch.Tensor``)

Import example::

    from src.constants import TF_DTYPE, _c, STEFAN_BOLTZMANN, PI
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PyTorch dtype used throughout the project
# ---------------------------------------------------------------------------
TF_DTYPE = torch.float64


# ---------------------------------------------------------------------------
# Helper: convert a Python scalar to a torch.tensor with the project dtype
# ---------------------------------------------------------------------------

def _c(value: float) -> torch.Tensor:
    """Convenience: scalar → ``torch.tensor(value, dtype=TF_DTYPE)``."""
    return torch.tensor(value, dtype=TF_DTYPE)


# ---------------------------------------------------------------------------
# Universal physical constants  (all torch.Tensor, float64, SI units)
# ---------------------------------------------------------------------------
STEFAN_BOLTZMANN: torch.Tensor   = _c(5.670374419e-8)     # W m⁻² K⁻⁴
BOLTZMANN_K: torch.Tensor        = _c(1.380649e-23)       # J K⁻¹
G_NEWTON: torch.Tensor           = _c(6.67430e-11)        # m³ kg⁻¹ s⁻²
AU_METRES: torch.Tensor          = _c(1.49597870700e11)   # 1 AU in metres

# 1.3608 ± 0.0005  kW/m2, which is 81.65 kJ/m2 per minute
SOLAR_CONSTANT_1AU: torch.Tensor = _c(1361.0)             # W m⁻² at 1 AU

PI: torch.Tensor                 = _c(3.141592653589793)   # π
