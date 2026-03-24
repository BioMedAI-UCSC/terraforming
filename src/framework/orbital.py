from dataclasses import dataclass
import torch

from src.constants import _c

@dataclass
class OrbitalParameters:
    """Keplerian orbital elements (assumed constant over short runs)."""

    semi_major_axis: torch.Tensor      # metres
    eccentricity: torch.Tensor         # dimensionless, 0 < e < 1
    orbital_period: torch.Tensor       # seconds
    axial_tilt: torch.Tensor           # radians

    def distance_from_sun(self, theta: torch.Tensor) -> torch.Tensor:
        """Heliocentric distance at true anomaly *theta* (radians).

        Equation (Kepler's first law – conic section):
            r(θ) = a(1 − e²) / (1 + e cos θ)

        Reference: https://en.wikipedia.org/wiki/Kepler_orbit
        """
        return (
            self.semi_major_axis
            * (_c(1.0) - self.eccentricity ** 2)
            / (_c(1.0) + self.eccentricity * torch.cos(theta))
        )
