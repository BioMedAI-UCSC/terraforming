from dataclasses import dataclass, field
import torch
from src.constants import _c

@dataclass
class Water:
    ice_mass: torch.Tensor = field(               # kg
        default_factory=lambda: _c(0.0))
    liquid_mass: torch.Tensor = field(            # kg
        default_factory=lambda: _c(0.0))
    vapour_mass: torch.Tensor = field(            # kg
        default_factory=lambda: _c(0.0))
