# Mars maps on the dinosaur dycore (dry dynamics, over MOLA terrain)

**Goal.** Produce lat/lon maps of each variable — surface pressure, near-surface
temperature, and winds — that are *comparable in form* to the NASA Ames MGCM and
LMD PCM, which both run over real Mars topography. This is the 3-D counterpart to
the 0-D seasonal (`Ls`) outputs.

**Status: dry-dynamics maps GO; physics coupling (radiation + CO₂) landed.** The
`gcm3d` primitive-equations core runs Mars over the real MOLA terrain and emits
dimensional lat/lon fields + plots + NetCDF. The two 0-D physics pieces
(radiative energy balance and the CO₂ condensation cycle) are now available as
explicit forcings *on the 3-D dycore* — see "Physics coupling" below — so the
fluid develops its own temperature structure and a seasonal polar CO₂ cap instead
of holding the isothermal rest state.

## What was built

| Module / symbol (`src.gcm3d`) | Role |
|---|---|
| `topography.load_mola_meg` | Read the staged MOLA MEGDR raster (`data/mola/meg004/megt90n000cb.img`, 720×1440 int16 MSB, metres). |
| `topography.regrid_to_nodal` | Bilinear (periodic-lon) regrid of MOLA onto the dynamics nodal grid. |
| `topography.mola_modal_orography` | Nondimensionalize + `truncated_modal_orography` → the modal lower boundary. |
| `maps.hydrostatic_surface_pressure_pa` | Mars barometry `p_s = p0·exp(−g·z/(R·T))` (Mars g, CO₂ R, ~10 km scale height). |
| `maps.initial_rest_state` | Rest, isothermal state with surface pressure balanced to the terrain. |
| `maps.run_maps` | Build coords + MOLA orography, integrate the dry dycore, return `MarsMapFields`. |
| `maps.save_maps` / `plot_maps` / `save_netcdf` | Per-variable cylindrical PNGs (MOLA contours overlaid) + a NetCDF for quantitative comparison. |
| `physics.RadiativeForcing` / `mars_radiative_forcing` | Radiative + orbital constants (albedo, greenhouse, emissivity, thermal inertia, obliquity, orbit) not carried on `BodyConstants`. |
| `physics.forced_primitive_equations` | Dry dynamics **+** the per-column radiative energy balance, as an explicit `ImplicitExplicitODE`. |
| `physics.CO2Forcing` / `mars_co2_forcing` | Constants for the CO₂ condensation cycle (frost point, latent heat, supply/exchange rates, escape). |
| `physics.forced_co2_primitive_equations` / `initial_co2_state` | Dynamics + radiation + CO₂ cycle on a tuple-wrapped state `(dyn, co2_ice)`. |

Sample output: `outputs/gcm3d_maps/` (`mars_topography.png`,
`mars_surface_pressure.png`, `mars_temperature.png`, `mars_wind_speed.png`,
`mars_maps.nc`), at T42 / 25 levels.

## Why the output is trustworthy (and where it isn't)

**Geography is correct.** The regridded topography places Olympus Mons + the
Tharsis Montes (~225–250°E), Hellas (~70°E/−40°), Valles Marineris and the
lowland/highland dichotomy where they belong.

**The defining validation signature holds.** Surface pressure sits in Mars
hydrostatic balance with the terrain: ~110 Pa over the Tharsis highs to ~1290 Pa
in Hellas, with `corr(elevation, p_s) ≈ −0.98`. That low-over-Tharsis /
high-in-Hellas pattern is the first map every Mars GCM is checked against.

**Fidelity ceiling (the honest caveat).** With `forcing=None` this is the pure
**dry dynamical core** (near-surface T holds the ~200 K reference because nothing
drives it radiatively). With the physics coupling below it becomes a **grey
(single-band) radiative-dynamical model with a CO₂ cap cycle**: still no spectral
radiative transfer and no dust, so the field *structure* is comparable to Ames/LMD
and the temperature/cap *behaviour* is now qualitatively right, but the
*magnitudes* are not yet quantitatively validated. Remaining staged work: a
proper (multi-band) radiation scheme, dust, and quantitative comparison.

## Physics coupling: radiation + CO₂ (0-D → 3-D)

`src.gcm3d.physics` moves the two 0-D physics pieces onto the 3-D dycore as
**explicit forcing terms** added to dinosaur's `explicit_terms`. dinosaur's `State`
already carries `sim_time` (the base equations advance it), so the diurnal/seasonal
forcing needs no new state — set `sim_time=0.0` and it advances.

- **Radiative energy balance.** The *same* nonlinear surface balance as the 0-D
  kernel, `dT/dt = (Q_in − εσ(T/gh)⁴)/C`, with `Q_in = (1−A)·S·cos_zenith`
  evaluated *per grid column* (latitude, solar declination from `Ls`, and a
  rotating diurnal hour angle). Applied column-uniform (the dry core has no
  convection to redistribute a surface-only flux). This is what grows the hot
  dayside / cold nightside and the seasonal hemispheric asymmetry that drive the
  circulation. Verified: it grows a ~30–55 K spatial temperature contrast the dry
  core (≈0 K) cannot, stays differentiable end-to-end, and reproduces the 0-D
  balance per column to machine precision (through the spectral round-trip).

- **CO₂ condensation cycle (Leighton–Murray).** CO₂ condenses wherever the surface
  reaches the frost point (the winter pole), releasing latent heat and *removing
  atmospheric mass* (local `p_s` drops); the frost sublimes back when insolated.
  Surface frost is not advected by the wind, so it rides in a **tuple-wrapped state
  `(dyn, co2_ice)`** (a JAX pytree — the stepper is unchanged), with ice tracked in
  pressure-equivalent units so mass conservation is exact per cell,
  `d(p_s) = −d(ice) − escape`. Verified: total mass conserved to <0.1 % (escape
  off), frost forms at the winter pole (|lat|>60°), and the latent buffering lifts
  the winter-pole cold floor from ~66 K (radiation only) to ~130 K.

**Two numerical guards this surfaced** (both refuse silently-wrong output):

- *Diurnal-terminator CFL.* The day/night terminator sweeps 360°/sol; if a step
  lets it jump more than ~½ a longitude cell the sharp (non-band-limited) terminator
  aliases and, with the T⁴ feedback, diverges to NaN. `run_maps` requires
  `dt ≤ rotation_period/(2·n_lon)` for diurnal forcing (≈347 s at T42, 694 s at
  T21), or use `diurnal=False` (daily-mean insolation) to step coarser.
- *Condensation supply limit.* Condensation is gated by `tanh(p_s/supply_scale)`
  so it vanishes as the column thins — without it, runaway polar condensation
  collapses `p_s → 0` and `d(ln p_s)` blows up.

## MOLA topography — now required (was not, for 0-D)

For the 0-D global-mean ODE, MOLA was correctly *not* needed (no horizontal
grid). For maps it is essential: without real orography the surface-pressure
field is flat and there is nothing to compare. The `flat_orography` default in
`dynamics.py` remains for planet-agnostic / idealised runs; `run_maps` swaps in
the MOLA modal orography.

## Usage

```python
from src.gcm3d import run_maps, save_maps
from src.gcm3d import mars_radiative_forcing, mars_co2_forcing

# Dry dynamical core (no physics forcing):
fields = run_maps(truncation="T42", n_layers=25, dt_seconds=600.0, n_steps=300)

# + grey radiative energy balance (diurnal forcing needs dt <= rot/(2*n_lon)):
fields = run_maps(truncation="T42", n_layers=12, dt_seconds=300.0, n_steps=600,
                  forcing=mars_radiative_forcing())

# + CO₂ condensation cycle (adds a seasonal polar cap; fields.co2_ice_pa):
fields = run_maps(truncation="T42", n_layers=12, dt_seconds=300.0, n_steps=600,
                  forcing=mars_radiative_forcing(), co2_forcing=mars_co2_forcing())

save_maps(fields, "outputs/gcm3d_maps")   # PNGs + NetCDF (incl. co2_ice when present)
# fields.surface_pressure_pa / temperature_k / u_ms / v_ms / co2_ice_pa : (n_lat, n_lon)
```

## Next steps toward true Ames/LMD comparability

1. ~~Port a radiative-equilibrium temperature forcing so the temperature map
   reflects insolation, not the flat reference.~~ **Done** — grey per-column energy
   balance (`physics.forced_primitive_equations`).
2. ~~Add the CO₂ condensation cycle for the seasonal surface-pressure cycle.~~
   **Done** — Leighton–Murray cap cycle (`physics.forced_co2_primitive_equations`).
3. Drive a seasonal run and compare zonal-mean cross-sections + `Ls`-binned maps
   against Mars Climate Database / published Ames/LMD fields.
4. Upgrade the grey radiation to a multi-band scheme, add dust, and give the
   radiative forcing vertical structure (currently column-uniform for stability).
