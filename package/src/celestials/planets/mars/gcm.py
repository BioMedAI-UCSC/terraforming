"""MarsGCM — the JCM (JAX 3-D GCM) backend behind the package Planet interface.

This class subclasses :class:`Mars` (the PyTorch 0-D box model) **not for its
physics** — JCM owns all of that — but as an *adapter*: by inheriting Mars we
get the scalar "diagnostic mirror" (``thermal`` / ``atmosphere`` / ``water`` /
``radiation``) and ``advance_orbit`` for free.  Each step advances the real
3-D JCM state, then reduces it back into those scalars, so ``TimeController``
and ``Snapshot`` keep working unchanged.

The seam:  ``TimeController.evolve`` → (``Accuracy.GCM``) → ``step_gcm`` →
``jcm.model.Model`` → reduce 3-D grid → write scalar mirror → ``Snapshot``.
"""

from __future__ import annotations

# ── Standard library ──────────────────────────────────────────────────────────
from typing import Any

# ── Third-party: PyTorch (the mirror / Snapshot side) ─────────────────────────
import torch

# ── Third-party: JAX (reducing the 3-D JCM state to scalars) ──────────────────
import jax.numpy as jnp

# ── Third-party: JCM (the JAX 3-D physics side) ───────────────────────────────
# Model      — the JCM simulator we construct and step.
# constants  — set_constants(...) pushes Mars' gravity/radius/Ω/... into JCM.
# default_forcing — the boundary forcing the model steps with.
from jcm import model as jcm_model
from jcm import constants as jcm_const
from jcm.forcing import default_forcing
from jcm.physics.held_suarez.utils import get_held_suarez_coords

# ── First-party: the package interface we are adapting to ─────────────────────
# TF_DTYPE — torch dtype the scalar mirror / Snapshot expect.
# Mars     — base class (adapter target); super().setup_properties() builds the mirror.
# MARS     — framework-neutral float constants shared with the torch model,
#            reused here so JCM and the box model agree on Mars' numbers.
from src.constants import TF_DTYPE
from src.celestials.planets.mars.planet import Mars
from src.celestials.planets.mars.constants import MARS
from src.celestials.planets.mars.gcm_physics import mars_physics
from src.celestials.planets.mars.terrain import build_mars_terrain, DEFAULT_MOLA_IMG


# Placeholder dynamical timestep (seconds).  This value is stable for the
# Held-Suarez grid with the Mars constants.  Tune it when the real coords and
# topography arrive.
_MARS_DYNAMICAL_DT_S = 180.0

# JCM measures run length in days; the engine measures dt in seconds.
_SECONDS_PER_DAY = 86400.0


# ─────────────────────────────────────────────────────────────────────────────
# Constants bridge — Mars floats → JCM PhysicalConstants
# ─────────────────────────────────────────────────────────────────────────────
def _apply_mars_constants() -> jcm_const.PhysicalConstants:
    """Push Mars physical constants into JCM.

    JCM starts from Earth defaults.  This function applies the Mars overrides.
    It reuses ``MARS.as_jcm_overrides()``, the single source of truth for the
    Mars → JCM constant map.  That method lives in ``constants.py``, which
    imports no torch and no JAX, so both backends agree on the numbers.

    Call this **before** you build the physics.  JCM physics terms read
    ``jcm.constants.physical_constants`` at construction, so the override must
    exist first.

    Note: ``set_constants`` writes a global singleton.  This is correct for one
    planet.  JCM "Patch 3" will thread the constants later.

    Returns
    -------
    jcm.constants.PhysicalConstants
        The active constants after the Mars override.
    """
    # dict[str, Any] (not dict[str, float]) so Pyright does not treat a possible
    # "constants" key as a float for set_constants' typed leading parameter.
    overrides: dict[str, Any] = MARS.as_jcm_overrides()
    return jcm_const.set_constants(**overrides)


# ─────────────────────────────────────────────────────────────────────────────
# Model builder — coords + terrain + physics
# ─────────────────────────────────────────────────────────────────────────────
def _build_mars_model(mola_img_path=None) -> jcm_model.Model:
    """Build the JCM model for Mars.

    This uses the Held-Suarez grid for now.  The terrain is all land.  If you
    pass a MOLA file, the model uses the real Mars topography.  If not, the
    terrain is flat (a placeholder).

    Call ``_apply_mars_constants()`` before this function.  The physics term
    reads the constants singleton when you build it.  The terrain also reads the
    Mars gravity when it computes the surface geopotential.

    Parameters
    ----------
    mola_img_path : Path or None
        The MOLA topography file.  None gives flat terrain.
    """
    coords = get_held_suarez_coords()
    terrain = build_mars_terrain(coords, mola_img_path=mola_img_path)
    return jcm_model.Model(
        coords=coords,
        terrain=terrain,
        time_step=_MARS_DYNAMICAL_DT_S,
        physics=mars_physics(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# MarsGCM — JCM-backed planet
# ─────────────────────────────────────────────────────────────────────────────
class MarsGCM(Mars):
    """Mars driven by the JCM 3-D GCM, presented through the Planet interface.

    We do **not** define ``__init__`` — ``Mars.__init__`` already stores the
    scalar config + device and then calls ``setup_properties()``.  We override
    that hook: ``super().setup_properties()`` builds the scalar mirror, then we
    build the JCM model/state on top of it.
    """

    # MOLA topography file.  If it exists, the model uses the real Mars terrain.
    # If not, the model falls back to flat terrain.  Set to None to force flat.
    MOLA_IMG_PATH = DEFAULT_MOLA_IMG

    # ── Construction hook (called by Mars.__init__, not by us) ───────────────
    def setup_properties(self) -> None:
        # 1. Build the scalar diagnostic mirror (thermal/atmosphere/water/...).
        #    self._device and the scalar config already exist at this point.
        super().setup_properties()

        # 2. Set the Mars constants FIRST.  The physics term reads them when we
        #    build the model, so the order matters.
        self._const_jcm = _apply_mars_constants()

        # 3. Build the JCM model and set its initial state.  Use the MOLA
        #    topography if the file is present; otherwise fall back to flat
        #    terrain.  bootstrap_state populates the dycore state and the
        #    physics carry without any integration.  The model owns that state;
        #    step_gcm advances it with resume(), which threads the carry for us.
        mola = self.MOLA_IMG_PATH
        mola = mola if (mola is not None and mola.exists()) else None
        self._model = _build_mars_model(mola_img_path=mola)
        self._model.bootstrap_state()
        self._forcing = default_forcing(self._model.coords.horizontal)

    # ── The GCM step (called by TimeController under Accuracy.GCM) ────────────
    def step_gcm(self, dt: torch.Tensor) -> None:
        """Advance the 3-D JCM state by ``dt`` and reduce it into the mirror.

        The engine gives ``dt`` in seconds.  JCM measures a run in days.  So we
        convert.  ``resume`` advances the model by that time and threads the
        physics carry.  Then we reduce the 3-D fields to the scalar mirror.

        Note: ``dt`` must be at least one dynamical step (see
        ``_MARS_DYNAMICAL_DT_S``).  A shorter ``dt`` gives no integration.
        """
        dt_days = float(dt.item()) / _SECONDS_PER_DAY
        preds = self._model.resume(
            forcing=self._forcing,
            save_interval=dt_days,
            total_time=dt_days,
        )
        self._write_mirror(preds)

    def _write_mirror(self, preds) -> None:
        """Reduce the 3-D prediction to the scalar mirror.

        The mirror is what ``Snapshot`` reads.  So this keeps the 3-D model
        behind the 0-D interface.  We take an area-weighted global mean with the
        Gaussian quadrature weights.

        Fields (shape after ``resume``):
          - ``temperature``: (time, level, ix, il).  The surface is the last
            level.
          - ``normalized_surface_pressure``: (time, ix, il).  Multiply by the
            reference pressure ``p0`` to get pascals.

        The CO₂-ice mass stays unchanged.  The placeholder physics has no CO₂
        tracer.  A later step (the co2_cycle term) fills it.
        """
        weights = jnp.asarray(self._model.coords.horizontal.quadrature_weights)
        weight_sum = jnp.sum(weights)

        def _global_mean(field_2d):
            return float(jnp.sum(field_2d * weights) / weight_sum)

        dynamics = preds.dynamics
        surface_temperature = _global_mean(dynamics.temperature[-1, -1])
        normalized_pressure = _global_mean(dynamics.normalized_surface_pressure[-1])
        surface_pressure = normalized_pressure * float(self._const_jcm.p0)

        device = self._device
        self.thermal.surface_temperature = torch.as_tensor(
            surface_temperature, dtype=TF_DTYPE, device=device)
        self.atmosphere.surface_pressure = torch.as_tensor(
            surface_pressure, dtype=TF_DTYPE, device=device)

    # ── RK4 / fast-physics paths never drive a 3-D spectral state ────────────
    def compute_derivatives(self, y: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("MarsGCM steps via step_gcm, not RK4")

    def compute_fast_physics(self, dt: torch.Tensor) -> None:
        raise NotImplementedError("MarsGCM steps via step_gcm, not fast physics")
