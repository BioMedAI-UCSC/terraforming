# What is tform?

tform is a differentiable 3-D Mars climate framework. Its primary model combines
Dinosaur's JAX spectral primitive-equation dynamics with explicit column physics
and Mars boundary data. It produces spatial temperature, pressure, wind and frost
fields and supports gradients through coupled trajectories.

## What the GCM evolves

| State | Representation |
| --- | --- |
| Circulation | Spectral vorticity and divergence on a spherical grid |
| Atmospheric temperature | Vertical sigma-layer temperature variations |
| Surface pressure | Prognostic logarithmic pressure |
| Surface temperature | Non-advected column surface reservoir |
| CO₂ frost | Surface pressure-equivalent reservoir |
| Subsurface temperature | Optional multilayer regolith |

Radiation, sensible heat, momentum drag, PBL diffusion, dry convection and CO₂
phase change enter as explicit tendencies. The IMEX integrator retains Dinosaur's
implicit gravity-wave solve. A complete-step correction restores the global CO₂
inventory and frost positivity in the energy-limited configuration.

## Primary workflows

Begin with the [GCM quickstart](quickstart.md). Use `tform mars maps` for exported
maps or `tform serve` for browser exploration. For coupled fitting, follow
[calibration](../package/gcm3d/calibration.md). The
[neural interfaces](../package/gcm3d/neural-experiments.md) support replacement
radiation and bounded heating; [temperature-only fitting](../package/gcm3d/temperature-only.md)
instead operates on frozen forecasts.

These entry points have different defaults. Match forcing, boundary data,
precision, initial state, timestep and duration when comparing them.
Resolution presets are computational settings, not accuracy certificates.

## Scientific scope

The model uses dry physics, prescribed dust and simplified regolith/PBL closures.
There is no water/cloud cycle, interactive dust lifting, photochemistry or validated
composition-dependent high-pressure physics. Reference comparisons are diagnostic;
they do not establish independent observational validation. See
[validation and limitations](../package/gcm3d/validation.md).

!!! warning "Deprecated global-mean workflows"
    The torch ODE model, `tform mars run` and its global-mean presets are deprecated.
    Use GCM maps and coupled experiment workflows for new work. Existing intervention
    campaigns still use the deprecated trajectory with independent GCM snapshots;
    they are not continuous 3-D intervention integrations.
