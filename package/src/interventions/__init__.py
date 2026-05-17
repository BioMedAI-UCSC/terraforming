"""Mars terraforming intervention layer.

Provides super-greenhouse gas injection, atmospheric concentration tracking,
radiative forcing computation, and a climate simulation controller.

Quick start
-----------
>>> from src.celestials import Mars
>>> from src.interventions import InterventionController, list_compounds
>>> print(list_compounds())
>>> mars = Mars()
>>> ic = InterventionController(mars, {"CF4": 1e9, "SF6": 5e8}, dt=21600)
>>> history = ic.run(n_years=100)
"""

from src.interventions.compounds import (
    COMPOUNDS,
    CompoundProperties,
    get_compound,
    list_compounds,
)
from src.interventions.state import GHGState
from src.interventions.forcing import (
    compute_concentration_ppb,
    delta_F_total,
    update_greenhouse_factor,
)
from src.interventions.controller import InterventionController, InterventionSnapshot

__all__ = [
    "COMPOUNDS",
    "CompoundProperties",
    "get_compound",
    "list_compounds",
    "GHGState",
    "compute_concentration_ppb",
    "delta_F_total",
    "update_greenhouse_factor",
    "InterventionController",
    "InterventionSnapshot",
]
