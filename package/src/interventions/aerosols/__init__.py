"""Engineered aerosol material, transport and radiation interfaces.

The configuration types remain importable without the optional JAX GCM extra.
"""
from .specs import Injection, ParticleSpec, aluminium_rod, iron_rod

__all__ = ["Injection", "ParticleSpec", "aluminium_rod", "iron_rod"]
