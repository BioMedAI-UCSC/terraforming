# Mars GCM configuration

Mars supplies body constants, forcing and boundary data to the generic JAX GCM.
The primary spatial workflow lives in `src.celestials.planets.mars.maps`; it does
not advance the deprecated torch `Mars.compute_derivatives` state.

## From planet to spatial state

1. Build spectral/sigma coordinates and body-based units from `MARS_BODY_3D`.
2. Load checksum-verified MOLA terrain and transform it to the modal lower boundary.
3. Initialize a terrain-balanced atmosphere or load a compatible full-state restart.
4. Attach radiation, surface properties, dust and CO₂ forcing.
5. Integrate coupled dynamics and surface reservoirs, optionally with diffusion.
6. Export dimensional fields, sampling metadata, diagnostics and restart state.

The initial hydrostatic pressure uses Mars gravity, gas constant and temperature:
`p_s(z) = p0 * exp(-g*z/(R*T))`. Winds and circulation subsequently evolve under
the primitive equations; terrain is not just a plotting overlay.

## Physics configuration

| Process | Selection and behavior |
| --- | --- |
| Solar geometry | Keplerian season/orbit; diurnal or daily-mean illumination; solar slant path |
| Radiation | Grey, explicit compact multiband, or bundled Ames correlated-k CO₂ |
| Surface exchange | Conservative sensible heat and momentum drag; optional stability dependence |
| Regolith | Optional multilayer conduction and TES thermal-inertia/albedo fields |
| PBL/convection | Conservative diffusion and smooth dry convective relaxation |
| CO₂ | Pressure-dependent frost point, surface latent heat and atmosphere/frost exchange |
| Dust | Prescribed opacity, optionally separate visible/IR seasonal climatology |
| Diffusion | Optional spectral damping selected by physical e-folding timescale |

The factory's generic defaults are deliberately smaller than the browser baseline.
`run_maps()` with no forcing is dry dynamics. `tform mars maps` enables grey
daily-mean radiation and CO₂ by default. The server enables correlated-k radiation
and advanced surface physics, longwave scales 0.25/0.25, surface exchange 1.0,
and a 0.1-sol diffusion timescale, using staged TES/Ames data when available.
See the [configuration quickstart](../package/gcm3d/quickstart.md).

## Outputs and comparison

Forced temperature maps are surface skin temperature; dry maps use lowest-layer
air temperature. Winds represent the lowest sigma layer rather than fixed 10 m.
Frost is pressure-equivalent (Pa); divide by gravity for kg/m². Full-state
comparison exports also provide atmospheric profiles and forcing metadata.
Match season, averaging, height and dust before interpreting reference differences.

## Calibration, neural components and limits

Physical calibration fits separate CO₂/dust longwave and surface-exchange
multipliers through coupled trajectories. Neural radiation replaces directional
fluxes with constrained boundary conditions; neural heating is bounded but can
add net energy. Cached temperature-only fitting is a separate downstream workflow.

The model has no water/cloud cycle, interactive dust lifting or photochemistry.
Composition-dependent high-pressure terraforming physics is not validated.
Recorded multi-year atmospheric/surface repeatability passed selected thresholds,
but deep-soil equilibrium did not. See [validation](../package/gcm3d/validation.md).

## Deprecated model and intervention semantics

The [torch global-mean derivation](global-mean-mars.md) is retained for existing
users and is deprecated. Current multi-year browser intervention trajectories
still use it, with independent GCM diagnostic spin-ups at selected years.
They are not continuous 3-D intervention forecasts. Use coupled GCM experiment
workflows for new spatial-model research and disclose this limitation for old runs.
