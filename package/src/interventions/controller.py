"""InterventionController — annual GHG injection scheduler for Mars.

The controller drives terraforming experiments by injecting super-greenhouse
gases on a yearly schedule.  All atmospheric state lives in
``mars.atmosphere.composition`` — injected GHGs are just additional entries
in the same dict alongside CO₂, N₂, Ar.  There is no separate GHGState object.

Architecture
------------
    mars.atmosphere.composition = {
        "CO2": 580 Pa,   # pre-existing
        "N2":   15 Pa,   # pre-existing
        "CF4":   X Pa,   # added by inject()
        "SF6":   Y Pa,   # added by inject()
        ...
    }

``mars.inject(schedule)`` converts kg → Pa and adds directly to composition.
``mars.decay_ghg(dt_years)`` decays all COMPOUNDS-registered species in place.
``mars.thermal.greenhouse_factor`` and ``mars.delta_F`` are always current.

The controller adds only one piece of state not derivable from physics:
cumulative_injected_kg — how much total mass has ever been injected per species.
This is a reporting quantity, not physics state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import torch

from src.constants import TF_DTYPE
from src.engine.time_controller import Accuracy, TimeController
from src.interventions.compounds import get_compound, COMPOUNDS
from src.celestials.planets.mars import MARS_ORBITAL_PERIOD as _MARS_YEAR_T


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------

@dataclass
class InterventionSnapshot:
    """Climate state at the end of one Mars year of intervention.

    All tensor fields live on the planet's device.
    """

    year:                     int                         # 1-based year counter
    time_s:                   torch.Tensor                # elapsed seconds

    # Climate state (mirrors TimeController Snapshot for compatibility)
    surface_temperature:      torch.Tensor                # K  (annual mean)
    temp_min:                 torch.Tensor                # K  (annual minimum)
    temp_max:                 torch.Tensor                # K  (annual maximum)
    surface_pressure:         torch.Tensor                # Pa (annual mean)
    ice_mass:                 torch.Tensor                # kg (end-of-year)
    solar_flux:               torch.Tensor                # W m⁻² (annual mean)
    orbital_angle:            torch.Tensor                # rad (end-of-year)
    greenhouse_factor:        torch.Tensor                # GHF at end of year

    # Intervention-specific
    delta_F:                  torch.Tensor                # total GHG forcing (W/m²)
    ghg_partial_pressure_Pa:  dict[str, torch.Tensor]     # Pa per GHG compound
    cumulative_injected_kg:   dict[str, torch.Tensor]     # total injected to date

    @property
    def time(self) -> torch.Tensor:
        """Alias for time_s — keeps compatibility with Snapshot consumers."""
        return self.time_s


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class InterventionController:
    """Annual GHG injection scheduler + climate simulation driver.

    Injected gases become part of ``mars.atmosphere.composition`` — the same
    dict that holds CO₂, N₂, and Ar.  No separate atmospheric bookkeeper exists.

    Parameters
    ----------
    mars : Mars
        Configured Mars instance.
    injection_schedule_kg_yr : dict[str, float]
        ``{compound_name: kg_per_Mars_year}`` for each injected species.
        All compound names must be present in the COMPOUNDS registry.
    dt : float
        Sub-annual integration timestep in seconds.  Default 3600 s (1 hour).
    accuracy : Accuracy
        Integration accuracy passed to the underlying TimeController.
    compile : bool
        If True, wrap physics methods with torch.compile.

    Examples
    --------
    >>> from src.celestials import Mars
    >>> from src.interventions import InterventionController
    >>> mars = Mars()
    >>> ic = InterventionController(mars, {"CF4": 1e9, "SF6": 5e8}, dt=21600)
    >>> history = ic.run(n_years=50)
    >>> # mars.atmosphere.composition now contains CF4 and SF6 partial pressures
    >>> print(mars.atmosphere.composition.keys())
    >>> print(f"ΔF = {mars.delta_F.item():.3f} W/m²")
    """

    def __init__(
        self,
        mars,
        injection_schedule_kg_yr: dict[str, float],
        dt: float = 3600.0,
        accuracy: Accuracy = Accuracy.FAST,
        compile: bool = False,
    ) -> None:
        for name in injection_schedule_kg_yr:
            get_compound(name)

        self._mars     = mars
        self._device   = mars._device
        self._schedule = {k: float(v) for k, v in injection_schedule_kg_yr.items()}
        self._tc       = TimeController(mars, dt=dt, accuracy=accuracy, compile=compile)

        # Reporting-only: cumulative kg injected per compound (not physics state)
        self._cumulative_injected_kg: dict[str, torch.Tensor] = {
            k: torch.zeros((), dtype=TF_DTYPE, device=self._device)
            for k in self._schedule
        }

        # Prime baselines on Mars so delta_F is well-defined from run() start
        if self._schedule:
            mars._init_ghg()

        self._elapsed     = torch.zeros((), dtype=TF_DTYPE, device=self._device)
        self._year_s      = float(_MARS_YEAR_T.item())
        self.all_hourly: list = []   # full hourly trace accumulated across all years

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_years: int,
        callback: Optional[Callable[[InterventionSnapshot], None]] = None,
    ) -> list[InterventionSnapshot]:
        """Run the intervention simulation for *n_years* Mars years.

        Per year:
            1. mars.inject(schedule)        → partial pressures updated in composition
            2. TimeController.run(1 year)   → ODE integrates with updated GHF
            3. mars.decay_ghg(1.0)          → COMPOUNDS species decay in composition
            4. InterventionSnapshot         → read from mars state

        Returns
        -------
        list[InterventionSnapshot]
            One snapshot per year (length = n_years).
        """
        history: list[InterventionSnapshot] = []

        for year in range(1, n_years + 1):
            # Step 1 — inject; GHF synced automatically inside mars.inject()
            if self._schedule:
                self._mars.inject(self._schedule)
                for name, kg in self._schedule.items():
                    self._cumulative_injected_kg[name] = (
                        self._cumulative_injected_kg[name] + kg
                    )

            # Step 2 — simulate one Mars year
            year_history = self._tc.run(duration=self._year_s)
            self.all_hourly.extend(year_history)
            self._elapsed = self._elapsed + self._year_s

            # Step 3 — decay; GHF resynced automatically inside mars.decay_ghg()
            self._mars.decay_ghg(dt_years=1.0)

            # Step 4 — snapshot from Mars (single source of truth)
            snap = self._make_snapshot(year, year_history)
            history.append(snap)

            if callback is not None:
                callback(snap)

        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_snapshot(self, year: int, year_history: list) -> InterventionSnapshot:
        def _mean(attr: str) -> torch.Tensor:
            return torch.stack([getattr(s, attr) for s in year_history]).mean()

        temps = torch.stack([s.surface_temperature for s in year_history])

        ghg_pp = {
            name: P.clone()
            for name, P in self._mars.atmosphere.composition.items()
            if name in COMPOUNDS
        }
        return InterventionSnapshot(
            year                    = year,
            time_s                  = self._elapsed.clone(),
            surface_temperature     = temps.mean(),
            temp_min                = temps.min(),
            temp_max                = temps.max(),
            surface_pressure        = _mean("surface_pressure"),
            ice_mass                = year_history[-1].ice_mass.clone(),
            solar_flux              = _mean("solar_flux"),
            orbital_angle           = year_history[-1].orbital_angle.clone(),
            greenhouse_factor       = self._mars.thermal.greenhouse_factor.clone(),
            delta_F                 = self._mars.delta_F.clone(),
            ghg_partial_pressure_Pa = ghg_pp,
            cumulative_injected_kg  = {
                k: v.clone() for k, v in self._cumulative_injected_kg.items()
            },
        )
