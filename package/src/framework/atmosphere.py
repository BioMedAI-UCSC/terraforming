from dataclasses import dataclass, field
from typing import Dict
import torch
from src.constants import _c

@dataclass
class Atmosphere:
    surface_pressure: torch.Tensor = field(       # Pa
        default_factory=lambda: _c(0.0))
    atmospheric_mass: torch.Tensor = field(       # kg
        default_factory=lambda: _c(0.0))
    composition: Dict[str, torch.Tensor] = field( # species → partial pressure (Pa)
        default_factory=dict,
    )
