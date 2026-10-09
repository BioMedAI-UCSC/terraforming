# Philosophy — gcm3d

## Problem Statement

The deprecated torch 0-D model integrates a global-mean energy/mass budget. It cannot answer
anything spatial: *where* does the winter CO₂ cap form, *how* does Tharsis shape
the surface-pressure field, *what* circulation do the equator–pole and
day–night temperature gradients drive. Answering those requires a real 3-D
general-circulation model (GCM) — a spectral dynamical core plus column physics.

Rather than write a dycore from scratch, `gcm3d` builds on **dinosaur**, the
differentiable JAX spectral primitive-equation core underneath NeuralGCM. This
provides a tested spectral core and JAX differentiation. Conservation and gradient
agreement still need measurement for each coupled configuration; using Dinosaur
does not certify the Mars physics or guarantee a particular trajectory throughput.

## Design Intent

1. **Planet-agnostic core, planet-specific instance.** The core consumes a
   `BodyConstants`; it hard-codes nothing about Mars. Mars supplies its instance
   (`MARS_BODY_3D`) next to the existing `MARS_*` constants, so there is a single
   source of truth. A new `BodyConstants` configures the dry core for another
   body; radiation, condensable reservoirs and boundary data need separate work.

2. **The dynamics⊕physics seam is stable.** Dinosaur's equations are an
   `ImplicitExplicitODE`. All Mars physics is added to the **explicit** side as a
   temperature/momentum/mass tendency, leaving dinosaur's semi-implicit
   gravity-wave treatment untouched. This is the same seam NeuralGCM uses to add
   *learned* residual tendencies — so a conventional package today can be swapped
   for a learned one later without touching callers (`column_primitive_equations`).

3. **Torch and JAX never cross-import.** The two frameworks meet only in the
   *experiment/test* layer, via DLPack or numerical parity checks — never by one
   module importing the other. `BodyConstants` and every `*Forcing` dataclass are
   pure Python precisely so the torch side can build them without importing JAX.

4. **Differentiable by construction.** Every rollout is a `jax.lax.scan`, so
   `jax.grad` flows from a diagnostic of the final state back to the initial state
   or forcing, and `jax.vmap` batches. Smooth (tanh) gates replace hard branches
   wherever a physical threshold (cap exhaustion, frost point) would otherwise
   zero the gradient. Some helpers retain piecewise projections; check gradient
   agreement with finite differences at the actual fitting horizon.

5. **Honesty about fidelity is a first-class requirement.** Output objects
   self-label their physics (`MarsMapFields.physics`) and flag when a run is a
   spin-up *transient* rather than a seasonally-equilibrated climatology
   (`is_transient`, < 668 sols). Longer duration still needs explicit seasonal
   convergence checks. See [validation and limitations](validation.md).

## What This Is Not

- **Not a validated Mars climatology (yet).** The dry-dynamics maps reproduce the
  *structure* the terrain imposes (surface pressure in hydrostatic balance with
  topography), not quantitative Ames/LMD magnitudes. The radiative path is a
  compact two-stream + optional Ames 12-band correlated-k scheme — good, but
  explicitly *not* full LMD/JCM line-by-line fidelity.
- **Separate from the deprecated torch engine.** The GCM is the recommended
  workflow for new work. Existing torch APIs remain for compatibility; deprecation
  does not imply all global-mean intervention capabilities have a continuous
  3-D replacement. The JAX seasonal 0-D prototype is not the spatial GCM.
- **Not a moist model.** No water cycle, clouds, or moist convection. Convection
  is dry adjustment only.

## Trade-offs Accepted

| Trade-off | Why |
|-----------|-----|
| Optional heavy JAX/`dinosaur` dependency | Keeps the torch core lightweight; gated behind one import guard. |
| Explicit grey, compact-band and correlated-k configurations | Keep ablations reproducible; resolved CO₂ radiation must be enabled before the bundled table is used. |
| Surface reservoirs live *outside* the dinosaur `State` (in `ColumnPhysicsState`) | Surface temperature/frost are not advected by wind, so they must not be dinosaur tracers (which get transported). |
| CO₂ exchange needs post-step corrections | Repair frost positivity and restore the global Gaussian-quadrature inventory after logarithmic-pressure integration, retaining configured escape. |
| Diurnal runs impose a strict CFL on the timestep | The moving day/night terminator is not band-limited; too coarse a step aliases and diverges to NaN — refused loudly rather than silently wrong. |
