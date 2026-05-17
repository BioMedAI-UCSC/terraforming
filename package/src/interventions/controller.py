"""InterventionController — drives GHG injection + Mars climate simulation.

This is the top-level orchestrator for terraforming intervention experiments.
It wraps TimeController, injects super-greenhouse gases on an annual schedule,
and returns a year-by-year record of the evolving climate state.

Architecture
------------
The InterventionController is a **separate layer** from the Mars physics:

    InterventionController
        ├── GHGState          (tracks atmospheric GHG masses)
        ├── forcing.py        (converts GHG mass → ΔF → updated GHF)
        └── TimeController    (integrates Mars physics for one year at a time)

Each Mars year the controller:
    1. Injects the annual GHG mass into GHGState
    2. Computes ΔF from current concentrations
    3. Writes updated greenhouse_factor to mars.thermal
    4. Runs TimeController for exactly one Mars year
    5. Decays GHG masses by one year
    6. Records an InterventionSnapshot

The physics model (mars.py, planet.py, time_controller.py) is never modified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch

from src.constants import TF_DTYPE
from src.engine.time_controller import Accuracy, TimeController
from src.interventions.compounds import get_compound
from src.interventions.forcing import (
    delta_F_total,
    update_greenhouse_factor,
    _MARS_EMISSIVITY,
)
from src.interventions.state import GHGState
from src.constants import STEFAN_BOLTZMANN


# Mars year in seconds (module-level Python float — device-agnostic)
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
    surface_pressure:         torch.Tensor                # Pa (annual mean)
    ice_mass:                 torch.Tensor                # kg (end-of-year)
    solar_flux:               torch.Tensor                # W m⁻² (annual mean)
    orbital_angle:            torch.Tensor                # rad (end-of-year)
    greenhouse_factor:        torch.Tensor                # updated GHF

    # Intervention-specific
    delta_F:                  torch.Tensor                # total GHG forcing (W/m²)
    ghg_masses_kg:            dict[str, torch.Tensor]     # atmospheric mass per compound
    cumulative_injected_kg:   dict[str, torch.Tensor]     # total injected to date

    # Convenience property: time in Mars years
    @property
    def time(self) -> torch.Tensor:
        """Alias for time_s (seconds) — keeps compatibility with Snapshot consumers."""
        return self.time_s


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class InterventionController:
    """Annual GHG injection scheduler + climate simulation driver.

    Parameters
    ----------
    mars : Mars
        Configured Mars instance.  The initial greenhouse_factor on this
        instance is used as the baseline (CO₂-only) GHF; the intervention
        layer adds ΔF on top.
    injection_schedule_kg_yr : dict[str, float]
        ``{compound_name: kg_per_Mars_year}`` for each injected species.
        All compound names must be present in the COMPOUNDS registry.
    dt : float
        Sub-annual integration timestep in seconds.  Default 3600 s (1 hour).
        Smaller values are more accurate; 3600–21600 s are physically stable
        for the fast-physics solver.
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
    >>> print(f"Year 50 temperature: {float(history[-1].surface_temperature.item()):.1f} K")
    """

    def __init__(
        self,
        mars,                                           # Mars instance
        injection_schedule_kg_yr: dict[str, float],
        dt: float = 3600.0,
        accuracy: Accuracy = Accuracy.FAST,
        compile: bool = False,
    ) -> None:
        # Validate compound names up front
        for name in injection_schedule_kg_yr:
            get_compound(name)

        self._mars     = mars
        self._device   = mars._device
        self._schedule = {k: float(v) for k, v in injection_schedule_kg_yr.items()}
        self._tc       = TimeController(mars, dt=dt, accuracy=accuracy, compile=compile)

        self._ghg = GHGState(
            compounds=list(self._schedule.keys()),
            device=self._device,
        )

        # Store the baseline GHF so it can be factored into the forcing formula
        self._baseline_ghf = mars.thermal.greenhouse_factor.clone()

        # Cache F_in_base = ε σ (T₀ / GHF₀)⁴ using the initial surface temperature.
        # This constant is passed to update_greenhouse_factor every year so that the
        # GHF formula's denominator never changes as T rises — otherwise the growing T
        # would inflate F_in_base and cause GHF to decrease despite more forcing.
        sb = STEFAN_BOLTZMANN.to(self._device)
        T0 = mars.thermal.surface_temperature.clone()
        self._baseline_olr: torch.Tensor = (
            _MARS_EMISSIVITY * sb * (T0 / self._baseline_ghf) ** 4.0
        )

        # Running elapsed time (seconds)
        self._elapsed = torch.zeros((), dtype=TF_DTYPE, device=self._device)

        # Mars year duration in seconds (as a Python float for run() duration arg)
        self._year_s = float(_MARS_YEAR_T.item())

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        n_years: int,
        callback: Optional[object] = None,
    ) -> list[InterventionSnapshot]:
        """Run the intervention simulation for *n_years* Mars years.

        Algorithm (per year)
        --------------------
        1. **Inject** annual GHG mass into GHGState (before decay so that
           the first year's injection is fully present when forcing is computed).
        2. **Compute ΔF** from current GHG concentrations.
        3. **Update** mars.thermal.greenhouse_factor via forcing.py.
        4. **Simulate** one Mars year using TimeController.run().
           The physics sees the updated GHF for this entire year.
        5. **Decay** GHG masses by one year (exponential decay).
        6. **Record** an InterventionSnapshot summarising the year.

        Parameters
        ----------
        n_years : int
            Number of Mars years to simulate.
        callback : callable, optional
            If provided, called as ``callback(snapshot)`` after each year.

        Returns
        -------
        list[InterventionSnapshot]
            One snapshot per year (length = n_years).
        """
        history: list[InterventionSnapshot] = []

        for year in range(1, n_years + 1):
            # Step 1 — inject
            self._ghg.inject(self._schedule)

            # Step 2 & 3 — compute ΔF and update GHF
            # Always pass the initial CO₂-only GHF so ΔF is applied additively
            # on top of the fixed baseline (not compounded on itself each year).
            total_atm_mass = self._mars.atmosphere.atmospheric_mass
            dF = delta_F_total(self._ghg, total_atm_mass)
            update_greenhouse_factor(self._mars, dF,
                                     baseline_ghf=self._baseline_ghf,
                                     baseline_olr=self._baseline_olr)

            # Step 4 — simulate one Mars year
            year_history = self._tc.run(duration=self._year_s)
            self._elapsed = self._elapsed + self._year_s

            # Step 5 — decay
            self._ghg.decay(dt_years=1.0)

            # Step 6 — record annual snapshot (mean over year's sub-steps)
            snap = self._make_snapshot(year, year_history, dF)
            history.append(snap)

            if callback is not None:
                callback(snap)

        return history

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_snapshot(
        self,
        year: int,
        year_history: list,
        dF: torch.Tensor,
    ) -> InterventionSnapshot:
        """Summarise one year of simulation into an InterventionSnapshot.

        Temperature, pressure, and solar flux are averaged over all timesteps
        within the year (annual mean).  Ice mass and orbital angle are the
        end-of-year values.
        """
        d = self._device

        def _mean(attr: str) -> torch.Tensor:
            vals = torch.stack([getattr(s, attr) for s in year_history])
            return vals.mean()

        return InterventionSnapshot(
            year                  = year,
            time_s                = self._elapsed.clone(),
            surface_temperature   = _mean("surface_temperature"),
            surface_pressure      = _mean("surface_pressure"),
            ice_mass              = year_history[-1].ice_mass.clone(),
            solar_flux            = _mean("solar_flux"),
            orbital_angle         = year_history[-1].orbital_angle.clone(),
            greenhouse_factor     = self._mars.thermal.greenhouse_factor.clone(),
            delta_F               = dF.clone(),
            ghg_masses_kg         = self._ghg.get_all_masses_kg(),
            cumulative_injected_kg= self._ghg.get_cumulative_injected(),
        )
