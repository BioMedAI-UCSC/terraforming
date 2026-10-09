# Mars GCM climate model

The primary Mars model resolves atmospheric dynamics and column physics on a
sphere with terrain-following sigma layers. It evolves circulation and spatial
temperature/pressure fields alongside surface temperature, CO₂ frost and optional
multilayer soil. The torch global-mean ODE model is deprecated.

## Dynamics and state

Dinosaur advances hydrostatic primitive equations with spectral vorticity,
divergence, temperature variation and logarithmic surface pressure. Mars supplies
its body constants and MOLA lower boundary. The grid is spherical harmonics
horizontally and sigma coordinates vertically; surface reservoirs are not
advected as atmospheric tracers.

## Energy exchange

Solar geometry follows a Keplerian eccentric orbit and either moving diurnal
illumination or a daily mean. The radiation scheme returns directional interface
fluxes whose convergence drives atmospheric heating and surface energy input.
Resolved CO₂ radiation uses bundled Ames correlated-k tables or an explicitly
selected compact-band alternative; simple CLI configurations use a grey balance.
Dust optical depth is prescribed and can vary seasonally.

Surface sensible heat is transferred to the lowest layer with its physical
pressure-dependent heat capacity. Surface loss and atmospheric gain cancel in
the exchange budget. Optional conduction transfers heat through the regolith;
PBL diffusion and smooth dry convective exchange redistribute column energy.

## CO₂ reservoirs

Condensation removes atmospheric mass and releases surface latent heat;
sublimation reverses the transfer. Frost is stored in pressure-equivalent units,
and its local saturation temperature depends on pressure. Energy-limited phase
exchange is tied to the surface-energy residual.

Advancing logarithmic pressure introduces finite-step inventory error. The
complete-step wrapper repairs negative frost and rescales atmospheric pressure
so global Gaussian-quadrature atmosphere-plus-frost matches the pre-step inventory
minus prescribed escape. This is a global budget correction, not proof of exact
local transport or full-system energy conservation.

## Integration and learning

The GCM uses IMEX-RK-SIL3: implicit gravity-wave dynamics and explicit physics.
It is separate from the deprecated model's RK4/fast modes. JAX scan rollouts
support gradients, batching and optional rematerialization; tests compare selected
gradients with finite differences. Physical calibration controls gas/dust thermal
opacity and surface exchange. Learned radiation replaces fluxes, while bounded
neural temperature heating adds a source. Temperature-only postprocessing instead
fits frozen forecast outputs.

## What results mean

Short maps are transient diagnostics. A duration above 668 sols does not establish
equilibrium; phase-matched atmospheric, surface and deep-soil repeatability must
be checked. Existing reference comparisons differ in sampling and wind height
and are not independent observational validation. Water/cloud physics,
interactive dust and composition-dependent high-pressure validation remain absent.

## Implementation and next steps

- [GCM architecture](../../architecture/planet.md) and [Mars configuration](../../architecture/mars.md).
- [Physics algorithms](../../package/gcm3d/implementation.md) and [API](../../package/gcm3d/api.md).
- [First simulation](../../getting-started/quickstart.md).
- [Calibration](../../package/gcm3d/calibration.md) and [neural experiments](../../package/gcm3d/neural-experiments.md).
- [Validation and limitations](../../package/gcm3d/validation.md).
- [Deprecated global-mean derivation](global-mean-climate-model.md).
