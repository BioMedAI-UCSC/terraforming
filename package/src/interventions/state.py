"""GHG atmospheric state tracker.

Tracks the atmospheric mass (kg) of each injected compound over time.
Handles first-order exponential decay and annual injection events.

All mass tensors stay on the specified device (CPU or CUDA) so no
cross-device copies occur during the simulation loop.
"""

from __future__ import annotations

import math

import torch

from src.constants import TF_DTYPE
from src.interventions.compounds import get_compound


class GHGState:
    """Atmospheric mass bookkeeper for a set of super-greenhouse gases.

    Parameters
    ----------
    compounds : list[str]
        Names of compounds to track (must be in the COMPOUNDS registry).
    device : torch.device
        Device on which all state tensors live.

    Notes
    -----
    All masses are stored as on-device scalar tensors.  Operations use
    Python floats (lifetimes, injection amounts) which PyTorch broadcasts
    correctly to the device without allocating intermediate CPU tensors.
    """

    def __init__(
        self,
        compounds: list[str],
        device: torch.device,
    ) -> None:
        self._device = device
        # Validate all names against registry before allocating state
        for name in compounds:
            get_compound(name)  # raises KeyError if unknown

        self._compounds = list(compounds)
        self._mass: dict[str, torch.Tensor] = {
            name: torch.zeros((), dtype=TF_DTYPE, device=device)
            for name in compounds
        }
        self._cumulative_injected: dict[str, torch.Tensor] = {
            name: torch.zeros((), dtype=TF_DTYPE, device=device)
            for name in compounds
        }

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def decay(self, dt_years: float) -> None:
        """Apply first-order exponential decay for each compound.

        Parameters
        ----------
        dt_years : float
            Time interval in Mars years over which to apply decay.

        The mass evolves as M(t+dt) = M(t) × exp(−dt / τ) where τ is
        the compound's atmospheric lifetime in years.  The decay factor is
        computed as a Python float and multiplied in-place so the operation
        stays on-device with a single scalar multiply.
        """
        for name, mass in self._mass.items():
            tau = get_compound(name).atmospheric_lifetime_yr
            decay_factor = math.exp(-dt_years / tau)
            self._mass[name] = mass * decay_factor

    def inject(self, schedule: dict[str, float]) -> None:
        """Add mass to the atmosphere for each compound in *schedule*.

        Parameters
        ----------
        schedule : dict[str, float]
            Mapping of compound name → kg to add.  Compounds not listed
            receive no injection.  Unknown compound names raise ``KeyError``.
        """
        for name, kg in schedule.items():
            if name not in self._mass:
                raise KeyError(
                    f"Compound '{name}' not tracked by this GHGState. "
                    f"Tracked: {self._compounds}"
                )
            self._mass[name] = self._mass[name] + float(kg)
            self._cumulative_injected[name] = (
                self._cumulative_injected[name] + float(kg)
            )

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    def get_mass_kg(self, compound: str) -> torch.Tensor:
        """Return current atmospheric mass tensor for *compound* (kg)."""
        return self._mass[compound]

    def get_all_masses_kg(self) -> dict[str, torch.Tensor]:
        """Return a shallow copy of all current mass tensors."""
        return dict(self._mass)

    def get_cumulative_injected(self) -> dict[str, torch.Tensor]:
        """Return total mass injected (before decay) per compound (kg)."""
        return dict(self._cumulative_injected)

    @property
    def compounds(self) -> list[str]:
        """Names of tracked compounds."""
        return list(self._compounds)

    @property
    def device(self) -> torch.device:
        return self._device
