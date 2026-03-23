"""Engine – TimeController with a single ``evolve`` and ``Accuracy`` enum.

The ``TimeController`` is the simulation engine.  It owns the **integration
strategy** and drives a ``Planet`` through time:

    from src.engine.time_controller import TimeController, Accuracy

    controller = TimeController(planet, dt=3600.0, accuracy=Accuracy.FAST)
    history    = controller.run(duration=MARS_ORBITAL_PERIOD)

Separation of concerns:
    Planet         → physics model  (derivatives, equilibrium, state)
    TimeController → integration    (RK4, reduced-order, time loop)

The ``Accuracy`` enum selects the ODE strategy:
    ``ACCURATE`` → 4th-order Runge-Kutta using ``planet.compute_derivatives``
    ``FAST``     → reduced-order model using ``planet.compute_fast_physics``

All scalar values are ``torch.Tensor`` (float64).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from src.constants import TF_DTYPE, _c

from src.celestials import (
    Planet,
)


# ---------------------------------------------------------------------------
# Strategy enum
# ---------------------------------------------------------------------------
class Accuracy(enum.Enum):
    """Integration accuracy level.

    ``FAST``     – reduced-order model (analytic equilibrium + relaxation).
                   Fast, suitable for long-duration sweeps.
    ``ACCURATE`` – full coupled ODE with 4th-order Runge-Kutta.
                   Higher fidelity, slower per step.
    """

    FAST = "fast"
    ACCURATE = "accurate"


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------
@dataclass
class Snapshot:
    """Lightweight record of state at one point in time."""

    time: torch.Tensor                   # elapsed simulation seconds
    surface_temperature: torch.Tensor    # K
    surface_pressure: torch.Tensor       # Pa
    ice_mass: torch.Tensor               # kg
    solar_flux: torch.Tensor             # W m⁻²
    orbital_angle: torch.Tensor          # rad


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class TimeController:
    """Simulation engine — drives a ``Planet`` through time.

    The controller owns the integration strategy and the time loop.
    The planet provides the physics (derivatives / analytic updates).

    Parameters
    ----------
    planet : Planet
        Any concrete Planet subclass (e.g. Mars).
    dt : float | torch.Tensor
        Integration timestep in seconds.  Default 3600 (1 hour).
    accuracy : Accuracy
        Which integration strategy to use.  Default ``Accuracy.FAST``.
    """

    def __init__(
        self,
        planet: Planet,
        dt: float | torch.Tensor = 3600.0,
        accuracy: Accuracy = Accuracy.FAST,
    ) -> None:
        self.dt = torch.as_tensor(dt, dtype=TF_DTYPE)
        if self.dt <= _c(0.0):
            raise ValueError("dt must be > 0")
        self.planet = planet
        self.accuracy = accuracy

    # ------------------------------------------------------------------
    # Core: single evolve method
    # ------------------------------------------------------------------
    def evolve(self, dt: torch.Tensor) -> None:
        """Advance the planet by *dt* seconds using the selected strategy.

        Steps performed:
            1. Advance orbital position  → updates ``planet.state.solar_flux``
            2. Apply physics step        → strategy-dependent
        """
        dt = torch.as_tensor(dt, dtype=TF_DTYPE)

        # Step 1 — orbital mechanics (common to both strategies)
        self.planet.advance_orbit(dt)

        # Step 2 — physics (strategy-dependent)
        if self.accuracy is Accuracy.ACCURATE:
            self._evolve_rk4(dt)
        else:
            self.planet.compute_fast_physics(dt)

    # ------------------------------------------------------------------
    # RK4 integrator  (the engine owns the integration method)
    # ------------------------------------------------------------------
    def _evolve_rk4(self, dt: torch.Tensor) -> None:
        """4th-order Runge-Kutta integration of the planet's ODE system.

        Uses ``planet.compute_derivatives(y)`` for the RHS and
        ``planet.pack_state() / unpack_state(y)`` for state ↔ tensor.

        References
        ----------
        RK4 method : https://en.wikipedia.org/wiki/Runge–Kutta_methods
        """
        planet = self.planet

        y = planet.pack_state()

        k1 = planet.compute_derivatives(y)
        k2 = planet.compute_derivatives(y + _c(0.5) * dt * k1)
        k3 = planet.compute_derivatives(y + _c(0.5) * dt * k2)
        k4 = planet.compute_derivatives(y + dt * k3)

        y_new = y + dt / _c(6.0) * (k1 + _c(2.0) * k2 + _c(2.0) * k3 + k4)

        planet.unpack_state(y_new)

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def run(
        self,
        duration: float | torch.Tensor,
        callback: Optional[Callable[[Planet, torch.Tensor], None]] = None,
    ) -> List[Snapshot]:
        """Run the simulation for *duration* seconds, return a snapshot list.

        Parameters
        ----------
        duration : float | torch.Tensor
            Total simulation time in seconds.
        callback : callable, optional
            Called as ``callback(state, elapsed)`` after every step.

        Returns
        -------
        list[Snapshot]
            One snapshot per timestep.
        """
        duration = torch.as_tensor(duration, dtype=TF_DTYPE)
        if duration <= _c(0.0):
            raise ValueError("duration must be > 0")

        history: List[Snapshot] = []
        elapsed = _c(0.0)

        while elapsed < duration:
            step = torch.minimum(self.dt, duration - elapsed)
            self.evolve(step)
            elapsed = elapsed + step

            s = self.planet
            history.append(
                Snapshot(
                    time=s.elapsed_time.clone(),
                    surface_temperature=s.thermal.surface_temperature.clone(),
                    surface_pressure=s.atmosphere.surface_pressure.clone(),
                    ice_mass=s.water.ice_mass.clone(),
                    solar_flux=s.radiation.solar_flux.clone(),
                    orbital_angle=s.orbital_angle.clone(),
                )
            )

            if callback:
                callback(s, elapsed)

        return history
