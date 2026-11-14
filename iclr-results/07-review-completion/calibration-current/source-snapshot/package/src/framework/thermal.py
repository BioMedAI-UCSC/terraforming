from dataclasses import dataclass, field
import torch
from src.constants import _c

@dataclass
class Thermal:
    surface_temperature: torch.Tensor = field(    # K
        default_factory=lambda: _c(0.0))
    greenhouse_factor: torch.Tensor = field(      # dimensionless (≥ 1)
        default_factory=lambda: _c(1.0))
