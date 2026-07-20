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

GPU support
-----------
The controller inherits its compute device from the planet: all tensors
(``dt``, ``elapsed``, ``duration``) are created on ``planet._device`` so
the entire simulation stays on-device.

The main loop pre-computes ``n_steps = ⌊duration / dt⌋`` at construction so
the ``for _ in range(n_steps)`` body contains **no GPU→CPU synchronisation**
(no tensor comparisons used as Python booleans).  This lets the GPU queue
many operations ahead of the CPU and maximises pipeline utilisation.

Optional ``compile`` flag wraps ``planet.compute_derivatives`` and
``planet.compute_fast_physics`` with ``torch.compile`` for kernel fusion.
On GPU with ``mode="reduce-overhead"``, this uses CUDA graphs to eliminate
per-kernel launch overhead.  The first call incurs JIT compilation; all
subsequent calls use the cached compiled graph.

All scalar values are ``torch.Tensor`` (float64).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import Callable, List, Optional

import torch

from src.constants import TF_DTYPE

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
    compile : bool
        If ``True``, wrap ``planet.compute_derivatives`` and
        ``planet.compute_fast_physics`` with ``torch.compile`` for kernel
        fusion.  Recommended when running on CUDA.  Default ``False``.
    """

    def __init__(
        self,
        planet: Planet,
        dt: float | torch.Tensor = 3600.0,
        accuracy: Accuracy = Accuracy.FAST,
        compile: bool = False,
    ) -> None:
        self.planet   = planet
        self.accuracy = accuracy
        self.device   = getattr(planet, '_device', torch.device('cpu'))

        _dt_f = float(dt.item()) if isinstance(dt, torch.Tensor) else float(dt)
        if _dt_f <= 0.0:
            raise ValueError("dt must be > 0")
        self.dt = torch.tensor(_dt_f, dtype=TF_DTYPE, device=self.device)

        if compile:
            planet.compute_derivatives  = torch.compile(
                planet.compute_derivatives,  fullgraph=False)
            planet.compute_fast_physics = torch.compile(
                planet.compute_fast_physics, fullgraph=False)

    # ------------------------------------------------------------------
    # Core: single evolve method
    # ------------------------------------------------------------------
    def evolve(self, dt: torch.Tensor) -> None:
        """Advance the planet by *dt* seconds using the selected strategy.

        Steps performed:
            1. Advance orbital position  → updates ``planet.state.solar_flux``
            2. Apply physics step        → strategy-dependent
        """
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

        Python-float coefficients (0.5, 2.0, 6.0) broadcast to the device
        of ``y`` without creating CPU constant tensors.

        References
        ----------
        RK4 method : https://en.wikipedia.org/wiki/Runge–Kutta_methods
        """
        planet = self.planet

        y  = planet.pack_state()

        k1 = planet.compute_derivatives(y)
        k2 = planet.compute_derivatives(y + 0.5 * dt * k1)
        k3 = planet.compute_derivatives(y + 0.5 * dt * k2)
        k4 = planet.compute_derivatives(y + dt * k3)

        y_new = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        planet.unpack_state(y_new)

    # ------------------------------------------------------------------
    # Snapshot helper
    # ------------------------------------------------------------------
    def _snapshot(self) -> Snapshot:
        s = self.planet
        return Snapshot(
            time=s.elapsed_time,
            surface_temperature=s.thermal.surface_temperature,
            surface_pressure=s.atmosphere.surface_pressure,
            ice_mass=s.water.ice_mass,
            solar_flux=s.radiation.solar_flux,
            orbital_angle=s.orbital_angle,
        )


    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def run(
        self,
        duration: float | torch.Tensor,
        callback: Optional[Callable[[Planet, torch.Tensor], None]] = None,
    ) -> List[Snapshot]:
        """Run the simulation for *duration* seconds, return a snapshot list.

        The loop uses a pre-computed ``n_steps`` count so the inner body
        contains no GPU tensor comparisons (no implicit GPU→CPU syncs).
        This keeps the GPU pipeline full when running on CUDA.

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
        dur_f = float(duration.item()) if isinstance(duration, torch.Tensor) else float(duration)
        dt_f  = float(self.dt.item())
        if dur_f <= 0.0:
            raise ValueError("duration must be > 0")

        n_steps   = int(dur_f / dt_f)
        remainder = dur_f - n_steps * dt_f

        # elapsed is kept on-device for use in callbacks
        elapsed = torch.zeros((), dtype=TF_DTYPE, device=self.device)

        history: List[Snapshot] = []
        for _ in range(n_steps):
            self.evolve(self.dt)
            elapsed = elapsed + self.dt
            history.append(self._snapshot())
            if callback:
                callback(self.planet, elapsed)

        if remainder > 1e-9:
            step = torch.tensor(remainder, dtype=TF_DTYPE, device=self.device)
            self.evolve(step)
            elapsed = elapsed + step
            history.append(self._snapshot())
            if callback:
                callback(self.planet, elapsed)

        return history