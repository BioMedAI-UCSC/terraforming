from dataclasses import dataclass
import torch

@dataclass
class IntrinsicParameters:
    mass: torch.Tensor                  # kg
    radius: torch.Tensor                # m
    rotation_period: torch.Tensor       # seconds (1 sidereal day)
    gravity: torch.Tensor               # m s⁻² (surface)
