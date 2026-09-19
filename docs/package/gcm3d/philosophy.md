# Philosophy — gcm3d

## Problem Statement

The torch 0-D model integrates a global-mean energy/mass budget. It cannot answer
anything spatial: *where* does the winter CO₂ cap form, *how* does Tharsis shape
the surface-pressure field, *what* circulation do the equator–pole and
day–night temperature gradients drive. Answering those requires a real 3-D
general-circulation model (GCM) — a spectral dynamical core plus column physics.

Rather than write a dycore from scratch, `gcm3d` builds on **dinosaur**, the
differentiable JAX spectral primitive-equation core underneath NeuralGCM. This
buys a validated, mass/energy/angular-momentum-conserving dynamical core *and*
end-to-end differentiability — the property the inverse-design / world-model work
needs at 10⁵–10⁶ trajectories, which is exactly JAX's strength.

## Design Intent

1. **Planet-agnostic core, planet-specific instance.** The core consumes a
   `BodyConstants`; it hard-codes nothing about Mars. Mars supplies its instance
   (`MARS_BODY_3D`) next to the existing `MARS_*` constants, so there is a single
   source of truth. Other bodies (Titan, Earth, exoplanets) come for free by
   defining a new `BodyConstants`.

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
   zero the gradient.

5. **Honesty about fidelity is a first-class requirement.** Output objects
   self-label their physics (`MarsMapFields.physics`) and flag when a run is a
   spin-up *transient* rather than a seasonally-equilibrated climatology
   (`is_transient`, < 668 sols). See `docs/ideas/gcm3d-physics-limitations.md`.

## What This Is Not

- **Not a validated Mars climatology (yet).** The dry-dynamics maps reproduce the
  *structure* the terrain imposes (surface pressure in hydrostatic balance with
  topography), not quantitative Ames/LMD magnitudes. The radiative path is a
  compact two-stream + optional Ames 12-band correlated-k scheme — good, but
  explicitly *not* full LMD/JCM line-by-line fidelity.
- **Not a torch replacement.** It does not touch `engine/`. The 0-D
  `terraforming_ode` is a *proof of concept* that the physics can live on the
  dinosaur substrate — an input to the go/no-go on porting the engine, not the port.
- **Not a moist model.** No water cycle, clouds, or moist convection. Convection
  is dry adjustment only.

## Trade-offs Accepted

| Trade-off | Why |
|-----------|-----|
| Optional heavy JAX/`dinosaur` dependency | Keeps the torch core lightweight; gated behind one import guard. |
| Grey / compact-band radiation as the default fallback | Differentiable and fast for ablations; the bundled Ames 12-band correlated-k table is the accurate default when staged. |
| Surface reservoirs live *outside* the dinosaur `State` (in `ColumnPhysicsState`) | Surface temperature/frost are not advected by wind, so they must not be dinosaur tracers (which get transported). |
| Explicit CO₂ mass exchange needs a post-step positivity projection | Multistage IMEX schemes do not preserve frost positivity; the projection restores exact per-cell CO₂ mass conservation. |
| Diurnal runs impose a strict CFL on the timestep | The moving day/night terminator is not band-limited; too coarse a step aliases and diverges to NaN — refused loudly rather than silently wrong. |
