"""SI particle and release contracts; no guessed warming/lifetime coefficients."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class ParticleSpec:
    name: str
    density_kg_m3: float
    length_m: float
    width_m: float
    height_m: float
    shape: str = "rod"
    optics_key: str | None = None
    source: str = ""
    production_energy_j_kg: float | None = None
    accommodation: float = 1.0 + math.pi / 8.0
    max_mass_loading: float | None = None

    def __post_init__(self):
        for name in ("density_kg_m3", "length_m", "width_m", "height_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.max_mass_loading is not None and (not math.isfinite(self.max_mass_loading) or self.max_mass_loading<=0):
            raise ValueError('user-selected mass loading limit must be finite and positive')
        if self.shape not in ("rod", "sphere", "disk"):
            raise ValueError("shape must be rod, sphere or disk")
        if not self.name or not self.source:
            raise ValueError("particle name and source/assumption provenance are required")
        if not 1 <= self.accommodation <= 1 + math.pi/8:
            raise ValueError("Epstein accommodation must be between specular and diffuse limits")
        if self.production_energy_j_kg is not None and (
            not math.isfinite(self.production_energy_j_kg) or self.production_energy_j_kg <= 0
        ):
            raise ValueError("production energy must be positive or explicitly unknown")
        if self.shape == "rod" and self.aspect_ratio < 2:
            raise ValueError("rod drag approximation requires aspect ratio >= 2")

    @property
    def volume_m3(self):
        if self.shape == "sphere":
            return math.pi*self.width_m**3/6
        if self.shape == "disk":
            return math.pi*self.width_m**2*self.height_m/4
        return self.length_m*self.width_m*self.height_m

    @property
    def mass_kg(self):
        return self.density_kg_m3*self.volume_m3

    @property
    def volume_radius_m(self):
        return (3*self.volume_m3/(4*math.pi))**(1/3)

    @property
    def aspect_ratio(self):
        return self.length_m/math.sqrt(self.width_m*self.height_m)

    @property
    def mean_projected_area_m2(self):
        # Cauchy's formula: mean projected area of a randomly oriented convex
        # body equals one quarter of its surface area.
        if self.shape == "sphere":
            return math.pi*self.width_m**2/4
        if self.shape == "disk":
            r = self.width_m/2
            return (2*math.pi*r*r+2*math.pi*r*self.height_m)/4
        l,w,h = self.length_m,self.width_m,self.height_m
        return (l*w+l*h+w*h)/2

    def as_dict(self):
        return {**asdict(self),"mass_kg":self.mass_kg,"volume_m3":self.volume_m3}


def aluminium_rod():
    return ParticleSpec("aluminium_rod_9um",2700.,9e-6,.16e-6,.16e-6,optics_key="al",
                        source="Ansari et al. 2024, DOI 10.1126/sciadv.adn4650, supplement Table 1; Al density 2700 kg/m3")


def iron_rod():
    return ParticleSpec("iron_rod_9um",7874.,9e-6,.16e-6,.16e-6,optics_key="fe",
                        source="Ansari et al. 2024, DOI 10.1126/sciadv.adn4650, supplement Table 1; Fe density 7874 kg/m3")


@dataclass(frozen=True)
class Injection:
    """Piecewise constant source in elapsed SI seconds, not ambiguous years."""
    start_s: float
    stop_s: float
    rate: float
    unit: str = "kg/s"
    sigma_min: float = 0.0
    sigma_max: float = 1.0
    latitude_min_deg: float = -90.0
    latitude_max_deg: float = 90.0
    longitude_min_deg: float = 0.0
    longitude_max_deg: float = 360.0

    def __post_init__(self):
        if not all(math.isfinite(v) for k,v in asdict(self).items() if k != "unit"):
            raise ValueError("source values must be finite")
        if self.start_s < 0 or self.stop_s <= self.start_s or self.rate < 0:
            raise ValueError("source requires 0 <= start < stop and nonnegative rate")
        if self.unit not in ("kg/s", "L/s"):
            raise ValueError("source unit must explicitly be kg/s or solid L/s")
        if not 0 <= self.sigma_min < self.sigma_max <= 1:
            raise ValueError("release sigma bounds must satisfy 0 <= min < max <= 1")
        if not -90 <= self.latitude_min_deg < self.latitude_max_deg <= 90:
            raise ValueError("invalid latitude interval")
        if not (0 <= self.longitude_min_deg < 360 and 0 < self.longitude_max_deg <= 360):
            raise ValueError("longitude bounds must lie in 0..360; decreasing bounds wrap across zero")

    def mass_rate(self, particle):
        return self.rate if self.unit == "kg/s" else self.rate*1e-3*particle.density_kg_m3

    def mass_between(self, start_s, stop_s, particle):
        """Analytic overlap, including timesteps crossing a source boundary."""
        return max(0.,min(stop_s,self.stop_s)-max(start_s,self.start_s))*self.mass_rate(particle)
