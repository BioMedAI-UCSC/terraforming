"""Batched simulation engine — runs B Mars simulations in parallel on GPU.

``BatchedMars`` stacks B independent Mars configurations into ``[B, ...]``
tensors so a single kernel invocation advances all B simulations at once.

``BatchedTimeController`` drives a ``BatchedMars`` through time with the
same RK4 / fast-physics strategies as the single-simulation engine, but
returns a ``list[list[Snapshot]]`` — one history list per batch element.

Typical use (from ``cli/runner.py``)::

    from src.engine import BatchedTimeController, Accuracy

    mars_list = [Mars(latitude=lat, device='cuda') for lat in lats]
    btc = BatchedTimeController(mars_list, dt=3600.0, accuracy=Accuracy.FAST)
    histories = btc.run(duration=10_000 * 88_775.244)  # list[list[Snapshot]]

GPU utilisation notes
---------------------
* All state is ``[B, ...]`` on one device — every physics kernel processes
  B simulations per call, dramatically improving GPU occupancy vs B separate
  scalar kernels.
* The main loop uses ``for _ in range(n_steps)`` (pre-computed) so there are
  **no GPU→CPU synchronisations** inside the hot loop.
* Pass ``compile=True`` to wrap ``compute_derivatives`` / ``compute_fast_physics``
  with ``torch.compile(mode='reduce-overhead')`` for additional kernel fusion
  via CUDA graphs.
* **float64 note**: consumer GPUs (RTX series) have ≈1/32 float64 throughput
  vs float32.  For maximum speed on consumer hardware, consider lowering the
  dtype to float32 (future work).  Data-centre GPUs (A100, H100) support
  full-speed float64 (≈1/2 throughput ratio).
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch

from src.constants import TF_DTYPE
from src.engine.time_controller import Accuracy, Snapshot

# Import Mars lazily inside methods to avoid circular import at module load
# (engine/__init__.py loads celestials which loads mars which loads engine).

_TWO_PI = 2.0 * math.pi


# ---------------------------------------------------------------------------
# BatchedMars
# ---------------------------------------------------------------------------
class BatchedMars:
    """B Mars simulations batched into ``[B]`` tensors for parallel execution.

    All per-instance mutable state (``_T``, ``_P``, ``_M``, ``elapsed_time``,
    ``orbital_angle``, ``solar_flux``) is a ``[B]`` tensor on the shared
    device.  Physics constants are scalar tensors shared across the batch.

    Parameters
    ----------
    mars_list : list[Mars]
        B pre-configured Mars instances.  All must be on the same device.
    """

    def __init__(self, mars_list) -> None:  # mars_list: list[Mars]
        B = len(mars_list)
        if B == 0:
            raise ValueError("mars_list must be non-empty")

        device = mars_list[0]._device
        for m in mars_list[1:]:
            if m._device != device:
                raise ValueError(
                    "All Mars instances in the batch must be on the same device"
                )

        self._B      = B
        self._device = device

        # ---- Per-instance mutable state as [B] tensors ----
        def _stack(attr_fn):
            return torch.stack([attr_fn(m) for m in mars_list]).to(device)

        self.elapsed_time  = _stack(lambda m: m.elapsed_time)
        self.orbital_angle = _stack(lambda m: m.orbital_angle)
        self.solar_flux    = _stack(lambda m: m.radiation.solar_flux)
        self._T            = _stack(lambda m: m.thermal.surface_temperature)
        self._P            = _stack(lambda m: m.atmosphere.surface_pressure)
        self._M            = _stack(lambda m: m.water.ice_mass)
        self._M_N          = _stack(lambda m: m.water.ice_mass_north)
        self._M_S          = _stack(lambda m: m.water.ice_mass_south)
        self._albedo       = _stack(lambda m: m.radiation.albedo)
        self._gh           = _stack(lambda m: m.thermal.greenhouse_factor)
        self._lat          = _stack(lambda m: m._init_latitude)

        # ---- Shared scalar constants (same across all Mars instances) ----
        # We take these from the first instance which has already moved them
        # to the correct device in its setup_properties().
        m0 = mars_list[0]
        self._SB          = m0._SB
        self._TI          = m0._TI
        self._EMISS       = m0._EMISS
        self._LS_PERI     = m0._LS_PERI
        self._CAP_FRAC    = m0._CAP_FRAC
        self._Q_out_pole  = m0._Q_out_pole
        self._LAT_HEAT    = m0._LAT_HEAT
        self._ESCAPE_RATE = m0._ESCAPE_RATE
        self._TIDE_PA     = m0._TIDE_PA
        self._TIDE_PHASE  = m0._TIDE_PHASE
        self._DIURNAL_AMP = m0._DIURNAL_AMP
        # Orbital / intrinsic scalars
        self._rot_period  = m0.intrinsic_params.rotation_period
        self._orb_period  = m0.orbital_params.orbital_period
        self._semi_major  = m0.orbital_params.semi_major_axis
        self._eccentricity= m0.orbital_params.eccentricity
        self._axial_tilt  = m0.orbital_params.axial_tilt
        self._radius      = m0.intrinsic_params.radius
        self._gravity     = m0.intrinsic_params.gravity

    # ------------------------------------------------------------------
    # State packing / unpacking
    # ------------------------------------------------------------------
    def pack_state(self) -> torch.Tensor:
        """Return batched state ``[B, 4]`` as ``[T, P, M_N, M_S]`` columns.

        The two caps are independent state variables (mirrors ``Planet``); the
        old total-plus-proportional-resplit drained the caps to zero on the RK4
        (ACCURATE) path.
        """
        return torch.stack([self._T, self._P, self._M_N, self._M_S], dim=1)

    def unpack_state(self, y: torch.Tensor) -> None:
        """Write ``[B, 4]`` state back into the planet attributes."""
        self._T   = y[:, 0].clamp(min=1.0)
        self._P   = y[:, 1].clamp(min=0.0)
        self._M_N = y[:, 2].clamp(min=0.0)
        self._M_S = y[:, 3].clamp(min=0.0)
        self._M   = self._M_N + self._M_S

    # ------------------------------------------------------------------
    # Orbital update
    # ------------------------------------------------------------------
    def advance_orbit(self, dt: torch.Tensor) -> None:
        """Advance all B simulations' orbital positions and update solar flux."""
        self.elapsed_time  = self.elapsed_time + dt
        self.orbital_angle = torch.remainder(
            self.orbital_angle + _TWO_PI * dt / self._orb_period,
            _TWO_PI,
        )
        # Kepler: r(θ) = a(1−e²)/(1+e cos θ)
        distance    = (self._semi_major * (1.0 - self._eccentricity ** 2)
                       / (1.0 + self._eccentricity * torch.cos(self.orbital_angle)))
        # Inverse-square solar flux (Python float constants — device-agnostic)
        self.solar_flux = 1361.0 * (1.49597870700e11 / distance) ** 2

    # ------------------------------------------------------------------
    # Batched physics: compute_derivatives  [B, 4] → [B, 4]
    # ------------------------------------------------------------------
    def compute_derivatives(self, y: torch.Tensor) -> torch.Tensor:
        """Batched ODE RHS.  ``y`` shape ``[B, 4]`` (T, P, M_N, M_S) → ``[B, 4]``."""
        T     = y[:, 0].clamp(min=1.0)   # [B]
        # P and M_ice are part of the state but not used in derivative calcs
        # that reference only T for radiation; kept for future coupling.

        omega = _TWO_PI / self._rot_period                        # scalar
        h     = omega * self.elapsed_time - math.pi               # [B]
        Ls    = self.orbital_angle + self._LS_PERI                # [B]
        delta = torch.asin(torch.sin(self._axial_tilt) * torch.sin(Ls))  # [B]

        # ----- dT/dt -----
        cos_zen = (
            torch.sin(self._lat) * torch.sin(delta)
            + torch.cos(self._lat) * torch.cos(delta) * torch.cos(h)
        ).clamp(min=0.0)                                          # [B]

        Q_in  = (1.0 - self._albedo) * self.solar_flux * cos_zen  # [B]
        T_eff = T / self._gh.clamp(min=1.0)                        # [B]
        Q_out = self._EMISS * self._SB * T_eff ** 4                # [B]
        dT_dt = (Q_in - Q_out) / self._TI                         # [B]

        # ----- dM_ice/dt -----
        A_cap = self._CAP_FRAC * 4.0 * math.pi * self._radius ** 2  # scalar

        cos_N  = torch.sin(delta).clamp(min=0.0)                    # [B]
        cos_S  = (-torch.sin(delta)).clamp(min=0.0)                 # [B]
        Q_N    = (1.0 - self._albedo) * self.solar_flux * cos_N     # [B]
        Q_S    = (1.0 - self._albedo) * self.solar_flux * cos_S     # [B]
        net_N  = (Q_N - self._Q_out_pole) * A_cap / self._LAT_HEAT  # [B]
        net_S  = (Q_S - self._Q_out_pole) * A_cap / self._LAT_HEAT  # [B]

        # Gate each cap on its own state component (y), so RK4 integrates the two
        # reservoirs independently through every sub-step.
        ice_N = y[:, 2].clamp(min=0.0)                               # [B]
        ice_S = y[:, 3].clamp(min=0.0)                               # [B]
        dM_N = torch.where((ice_N <= 0.0) & (-net_N < 0.0), 0.0, -net_N)  # [B]
        dM_S = torch.where((ice_S <= 0.0) & (-net_S < 0.0), 0.0, -net_S)  # [B]
        dMice_dt = dM_N + dM_S                                       # [B]

        # ----- dP/dt -----
        # Mass-budget pressure only; the thermal tide is a diagnostic overlay
        # (observed_surface_pressure), mirroring Mars.compute_derivatives.
        A_planet = 4.0 * math.pi * self._radius ** 2                 # scalar
        dP_dt = (
            -self._ESCAPE_RATE * self._gravity / A_planet
            - dMice_dt * self._gravity / A_planet
        )                                                             # [B]

        return torch.stack([dT_dt, dP_dt, dM_N, dM_S], dim=1)        # [B, 4]

    # ------------------------------------------------------------------
    # Batched fast physics  (in-place update)
    # ------------------------------------------------------------------
    def compute_fast_physics(self, dt: torch.Tensor) -> None:
        """Batched reduced-order physics update (in-place on ``[B]`` tensors)."""
        Ls    = self.orbital_angle + self._LS_PERI                  # [B]
        delta = torch.asin(torch.sin(self._axial_tilt) * torch.sin(Ls))  # [B]
        lat   = self._lat                                            # [B]

        # Daily-mean insolation factor
        cos_h0        = (-torch.tan(lat) * torch.tan(delta)).clamp(-1.0, 1.0)
        h0            = torch.acos(cos_h0)
        ins_factor    = (
            h0 * torch.sin(lat) * torch.sin(delta)
            + torch.cos(lat) * torch.cos(delta) * torch.sin(h0)
        ) / math.pi
        ins_factor    = ins_factor.clamp(min=0.0)

        absorbed      = (1.0 - self._albedo) * self.solar_flux * ins_factor
        T_eq_base     = (absorbed / (self._EMISS * self._SB)).clamp(min=0.0) ** 0.25
        T_eq          = T_eq_base * self._gh

        omega = _TWO_PI / self._rot_period
        swing = self._DIURNAL_AMP * torch.cos(lat)
        T_eq  = T_eq - swing * torch.cos(omega * self.elapsed_time)

        T_cur    = self._T.clamp(min=1.0)
        tau      = (self._TI / (4.0 * self._EMISS * self._SB * T_cur ** 3)).clamp(min=1.0)
        self._T  = (T_eq + (T_cur - T_eq) * torch.exp(-dt / tau)).clamp(min=1.0)

        # Ice budget
        A_cap = self._CAP_FRAC * 4.0 * math.pi * self._radius ** 2
        cos_N = torch.sin(delta).clamp(min=0.0)
        cos_S = (-torch.sin(delta)).clamp(min=0.0)
        Q_N   = (1.0 - self._albedo) * self.solar_flux * cos_N
        Q_S   = (1.0 - self._albedo) * self.solar_flux * cos_S
        net_N = (Q_N - self._Q_out_pole) * A_cap / self._LAT_HEAT
        net_S = (Q_S - self._Q_out_pole) * A_cap / self._LAT_HEAT

        dM_N      = torch.where((self._M_N <= 0.0) & (-net_N * dt < 0.0), 0.0, -net_N * dt)
        dM_S      = torch.where((self._M_S <= 0.0) & (-net_S * dt < 0.0), 0.0, -net_S * dt)
        dMice     = dM_N + dM_S
        self._M_N = (self._M_N + dM_N).clamp(min=0.0)
        self._M_S = (self._M_S + dM_S).clamp(min=0.0)
        self._M   = self._M_N + self._M_S

        # Pressure — mass-budget mean only; tide is a diagnostic overlay
        # (observed_surface_pressure), mirroring Mars.compute_fast_physics.
        A_planet = 4.0 * math.pi * self._radius ** 2
        self._P  = (
            self._P
            - self._ESCAPE_RATE * self._gravity / A_planet * dt
            - dMice * self._gravity / A_planet
        ).clamp(min=0.0)

    def observed_surface_pressure(self) -> torch.Tensor:
        """Batched local pressure: mass-budget mean + thermal-tide overlay
        (same closed form as Mars.observed_surface_pressure)."""
        omega = _TWO_PI / self._rot_period
        tide = self._TIDE_PA * (
            torch.cos(omega * self.elapsed_time + self._TIDE_PHASE)
            - torch.cos(self._TIDE_PHASE)
        )
        return self._P + tide


# ---------------------------------------------------------------------------
# BatchedTimeController
# ---------------------------------------------------------------------------
class BatchedTimeController:
    """Simulation engine for B parallel Mars simulations.

    Parameters
    ----------
    mars_list : list[Mars]
        B configured Mars instances (all on the same device).
    dt : float
        Integration timestep in seconds.
    accuracy : Accuracy
        ``FAST`` or ``ACCURATE`` integration strategy.
    compile : bool
        Wrap physics methods with ``torch.compile``.
        Recommended when ``device='cuda'``.  Default ``False``.
    """

    def __init__(
        self,
        mars_list,                      # list[Mars]
        dt: float = 3600.0,
        accuracy: Accuracy = Accuracy.FAST,
        compile: bool = False,
    ) -> None:
        self.bmars    = BatchedMars(mars_list)
        self.accuracy = accuracy
        self._device  = self.bmars._device

        _dt_f = float(dt)
        if _dt_f <= 0.0:
            raise ValueError("dt must be > 0")
        self.dt = torch.tensor(_dt_f, dtype=TF_DTYPE, device=self._device)

        if compile:
            self.bmars.compute_derivatives  = torch.compile(
                self.bmars.compute_derivatives,  fullgraph=False)
            self.bmars.compute_fast_physics = torch.compile(
                self.bmars.compute_fast_physics, fullgraph=False)

    # ------------------------------------------------------------------
    # Integration strategies
    # ------------------------------------------------------------------
    def _evolve(self, dt: torch.Tensor) -> None:
        self.bmars.advance_orbit(dt)
        if self.accuracy is Accuracy.ACCURATE:
            self._evolve_rk4(dt)
        else:
            self.bmars.compute_fast_physics(dt)

    def _evolve_rk4(self, dt: torch.Tensor) -> None:
        """Batched RK4: y is ``[B, 4]``."""
        bm = self.bmars
        y  = bm.pack_state()
        k1 = bm.compute_derivatives(y)
        k2 = bm.compute_derivatives(y + 0.5 * dt * k1)
        k3 = bm.compute_derivatives(y + 0.5 * dt * k2)
        k4 = bm.compute_derivatives(y + dt * k3)
        bm.unpack_state(y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4))

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------
    def run(self, duration: float) -> List[List[Snapshot]]:
        """Run all B simulations for *duration* seconds.

        Returns
        -------
        list[list[Snapshot]]
            Outer list: one per batch element (length B).
            Inner list: one Snapshot per timestep.
        """
        dur_f     = float(duration)
        dt_f      = float(self.dt.item())
        n_steps   = int(dur_f / dt_f)
        remainder = dur_f - n_steps * dt_f

        elapsed = torch.zeros((), dtype=TF_DTYPE, device=self._device)

        # Pre-allocate B empty history lists
        histories: List[List[Snapshot]] = [[] for _ in range(self.bmars._B)]

        for _ in range(n_steps):
            self._evolve(self.dt)
            elapsed = elapsed + self.dt
            self._record(histories, elapsed)

        if remainder > 1e-9:
            step = torch.tensor(remainder, dtype=TF_DTYPE, device=self._device)
            self._evolve(step)
            elapsed = elapsed + step
            self._record(histories, elapsed)

        return histories

    def _record(self, histories: List[List[Snapshot]], elapsed: torch.Tensor) -> None:
        """Append one Snapshot per batch element (0-dim CUDA tensor, no sync)."""
        bm = self.bmars
        B  = bm._B
        P_obs = bm.observed_surface_pressure()
        for i in range(B):
            histories[i].append(Snapshot(
                time=elapsed.clone(),
                surface_temperature=bm._T[i].clone(),
                surface_pressure=P_obs[i].clone(),
                ice_mass=bm._M[i].clone(),
                solar_flux=bm.solar_flux[i].clone(),
                orbital_angle=bm.orbital_angle[i].clone(),
            ))
