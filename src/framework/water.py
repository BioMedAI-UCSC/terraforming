from dataclasses import dataclass, field
import torch
from src.constants import _c

@dataclass
class Water:
    ice_mass: torch.Tensor = field(               # kg  total (north + south)
        default_factory=lambda: _c(0.0))
    ice_mass_north: torch.Tensor = field(         # kg  north polar CO₂ reservoir
        default_factory=lambda: _c(0.0))
    ice_mass_south: torch.Tensor = field(         # kg  south polar CO₂ reservoir
        default_factory=lambda: _c(0.0))
    liquid_mass: torch.Tensor = field(            # kg
        default_factory=lambda: _c(0.0))
    vapour_mass: torch.Tensor = field(            # kg
        default_factory=lambda: _c(0.0))
