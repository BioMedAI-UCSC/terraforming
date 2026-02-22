from dataclasses import dataclass, field
import torch
from src.constants import _c

@dataclass
class Magnetic:
    magnetic_field_strength: torch.Tensor = field(  # Tesla at surface
        default_factory=lambda: _c(0.0))
