from dataclasses import dataclass, field
import torch
from src.constants import _c

@dataclass
class Radiation:
    albedo: torch.Tensor = field(                 # 0-1
        default_factory=lambda: _c(0.25))
    solar_flux: torch.Tensor = field(             # W m⁻² (at current distance)
        default_factory=lambda: _c(0.0))
