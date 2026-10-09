# Implementation Details — gcm3d

## Module Map

| File | Purpose |
|------|---------|
| `body.py` | Pure-Python `BodyConstants` (radius, gravity, Ω, R, cp) + `EARTH`. |
| `_dinosaur.py` | Single guarded import of JAX + `dinosaur`; friendly error if the extra is absent. |
| `specs.py` | Body-anchored nondimensionalisation → dinosaur `SimUnits`. |
| `coordinates.py` | Spectral grid (`T21…T170`) × equidistant sigma levels. |
| `dynamics.py` | Dry `PrimitiveEquationsSigma`, semi-implicit stepper, scan integrator. |
| `physics.py` | All Mars column physics: orbit, insolation, two-stream radiation, surface energy, drag, PBL diffusion, dry convection, regolith conduction, CO₂ cycle. |
| `ames_radiation.py` | Bilinear interpolation of the bundled Ames 12-band correlated-k CO₂ tables + fixed dust optics. |
| `topography.py` | MOLA MEGDR raster → nodal → modal orography (SHA-256 verified). |
| `surface.py` | Interpolate an albedo/thermal-inertia (TES) dataset to the grid. |
| `dust.py` | Interpolate an Ames seasonal dust (τ, z_max) scenario to the grid. |
| `maps.py` | End-to-end 3-D Mars map run + NetCDF/PNG output + scale presets. |
| `terraforming_ode.py` | 0-D terraforming physics re-expressed as a dinosaur ODE (frozen-epoch and time-advancing seasonal variants). |
| `restart.py` | Versioned NPZ restart files; spin-up + time-averaging helper. |
| `benchmarks.py` | Planet-aware dry-dycore acceptance tests + conserved-quantity drift. |
| `mcd.py` | Mars Climate Database v6.1 web client + area-weighted comparison metrics. |

## Core Physics — derived

### 1. Keplerian orbit and insolation
The mean anomaly advances uniformly `M(t) = M₀ + 2π t/T_orb`; Kepler's equation
`M = E − e sin E` is solved by 6 Newton iterations (converges to ~machine ε for
e < 0.1). The true anomaly ν and heliocentric distance `r = a(1 − e cos E)` follow,
solar longitude `Ls = ν + Ls_perihelion` (≈251° for Mars), declination
`δ = arcsin(sin(tilt)·sin Ls)`, and flux `S = S₁AU·(AU/r)²`. `mean_anomaly_for_ls`
inverts the chain so a caller can set the epoch by season rather than by anomaly.

`cos_zenith_nodal` gives cos(solar zenith) on the `(n_lon, n_lat)` grid. Diurnal:
hour angle `h = lon + 2π t/T_rot`, `cz = sinφ sinδ + cosφ cosδ cos h`, clipped ≥ 0
— a terminator sweeping westward each sol. Non-diurnal: the daily-mean insolation
factor `⟨cz⟩ = (H₀ sinφ sinδ + cosφ cosδ sin H₀)/π`, `H₀ = arccos(−tanφ tanδ)`.

### 2. Two-stream radiation (`two_stream_radiative_fluxes`)
Two solar bands (Beer–Lambert transmission) and three thermal bands (hemispheric
two-stream). Optical depth per layer scales with local pressure ratio and a
temperature ratio raised to per-band exponents. Two closures:
- **Ames correlated-k (default, `ames_correlated_k_enabled`)**: gas SW/LW optical
  depths from `correlated_k_optical_depths` (bilinear in T and log-p over the
  bundled `ames_co2_12band.npz` table), split into split-Gaussian channels
  (`channel_weights`). Dust adds an energy-conserving hemispheric two-stream layer
  (reflectance/transmittance from single-scattering albedo ω and asymmetry g),
  thermal emission split by interpolated Planck band fractions
  (`planck_band_fractions`). Adding-doubling via two `jax.lax.scan`s
  (`add_layer`, `propagate_solar`) builds the multiple-scattering solar field;
  a thermal `propagate_thermal` scan builds up/down longwave.
- **Compact multiband fallback**: the same structure with the analytic band
  optical-depth model — used for ablations and installations without the asset.

Its principal contract is **exact discrete energy closure**:
`net_up = lw_up + sw_up − lw_down − sw`, layer convergence is the finite
difference of `net_up`, and the surface net closes the column budget independent
of layer count.

### 3. Surface energy balance (`surface_energy_tendencies`)
The nonlinear balance is solved once per column on a **prognostic** surface
temperature reservoir (thermal inertia `I`):
`dT_s/dt = (Q_in − Q_out − sensible)/I`. A bulk sensible flux
`H = ρ cp C_D |V| (T_s − T_air)` (or a constant coefficient when stability
exchange is off) transfers energy to the **lowest** sigma layer, divided by that
layer's areal heat capacity `cp·dp/g` with `dp = p_s·Δσ`. **Surface loss and
atmospheric gain therefore cancel exactly** and the budget is independent of the
number of layers — a deliberate conservation property.

### 4. Surface drag and boundary layer
- `surface_momentum_tendencies`: neutral log-law stress `τ = ρ C_D |V| V`,
  `C_D = (κ/ln(z/z₀))²` at the lowest-layer hydrostatic midpoint height; the
  layer acceleration `−τ/(dp/g)` always removes resolved kinetic energy.
- `pbl_vertical_diffusion_tendencies`: Richardson-number-dependent K-diffusion of
  momentum, heat and tracers between adjacent layers, solved implicitly
  (backward-Euler, batched O(L) Thomas solve in `_implicit_vertical_diffusion_tendency`).
  Dissipated kinetic energy is returned to the column as heat (resolved-budget conservation).
- `dry_convective_adjusted_temperature`: conservative pair-mixing that neutralises
  only adjacent statically-unstable layers — a differentiable, fixed-work (O(L),
  `2L−1`-step scan) approximation to block/PAVA isotonic regression, conserving
  σ-mass-weighted enthalpy per merge.

### 5. CO₂ condensation cycle (`co2_forcing_tendencies`)
The winter-pole physics the dry+radiation run is missing (without it the winter
pole cools to an unphysical ~66 K). Per cell: when the prognostic surface T falls
below the frost point, CO₂ condenses (releasing latent heat and **removing local
atmospheric mass** → p_s drops); deposited frost sublimes back when insolation
warms it, restoring mass. Frost `co2_ice` is stored in **pressure-equivalent**
units so per-cell mass conservation is exact: `d(p_s) = −d(ice) − escape`. The
frost point is either constant or the Clausius–Clapeyron curve
`T_sat = 3182.48/(23.3494 − ln p_hPa)` (`co2_frost_point_k`). `energy_limited`
mode ties the rate to the surface-energy residual (no tuned relaxation constant).
Because multistage IMEX does not preserve positivity, `positivity_preserving_co2_step`
projects frost ≥ 0 after each *complete* step, moving any deficit back to the column.

### 6. The dynamics⊕physics composition
`forced_co2_primitive_equations` builds a `parameterization(state)` that sums all
of the above into a `ColumnPhysicsTendencies`, then `column_primitive_equations`
adds them to dinosaur's dry `explicit_terms` (implicit gravity-wave side untouched,
`implicit_inverse` passes surface reservoirs through). Time enters via the dycore
state's own `sim_time` field (initialise to 0.0).

## Data Structures

- **`ColumnPhysicsState`** (NamedTuple pytree): `dynamics` (dinosaur `State`) +
  `surface_temperature`, `co2_ice`, `ground_temperature`. Chosen so surface
  reservoirs ride along without being advected as tracers.
- **Regolith**: 12 skin-depth-scaled soil layers (`_REGOLITH_LAYER_FRACTIONS`),
  the deepest retaining seasonal memory (beyond the annual skin depth); conservative
  finite-volume conduction.
- **Sigma vertical coordinate**: p/pₛ rescales with surface pressure automatically
  — essential for Mars, whose CO₂ cycle moves ~25 % of the atmospheric mass.

## Internal Invariants

- Sensible-flux surface loss ≡ lowest-layer atmospheric gain (layer-count-independent).
- Radiative fluxes satisfy exact discrete energy closure.
- CO₂ mass: `d(p_s) = −d(ice) − escape` per cell; positivity projection preserves total column CO₂.
- Rollouts are pure `jax.lax.scan` → differentiable (`jax.grad`) and batchable (`jax.vmap`); tests verify gradient flow.
- `dry_conserved_quantities` measures mass / total energy / axial angular-momentum drift for acceptance.

## Known Limitations

- **Fidelity**: dry-dynamics maps reproduce terrain-imposed *structure* (surface
  pressure), not Ames/LMD *magnitudes*; radiation is two-stream + 12-band
  correlated-k, not line-by-line. No moist processes. See
  [`gcm3d-physics-limitations.md`](../../ideas/gcm3d-physics-limitations.md).
- **Transient vs climatology**: runs < 668 sols are spin-up snapshots; outputs
  self-label this (`MarsMapFields.is_transient`, physics string suffix).
- **Diurnal CFL**: diurnal forcing needs ≥ 2 steps per longitude cell
  (`dt ≤ T_rot/(2 n_lon)`); the seasonal 0-D ODE needs `dt ≤ T_rot/8`. Both are
  enforced with a hard `ValueError` rather than emitting NaN/aliased output.
- **Regolith state and forcing are coupled explicitly**: pass the run's forcing
  to `initial_column_state(..., forcing=forcing)`. Custom layer fractions then
  determine both the ground-state size and conduction geometry; a mismatch is
  rejected with a descriptive error.

## Testing Notes

Tests mirror the source tree under `package/tests/gcm3d/`. Coverage includes:
- **Parity**: `terraforming_ode.tendency` matches the torch `compute_derivatives`
  to float64 machine precision; a dinosaur-stepped rollout tracks the torch RK4
  rollout to truncation order (`test_terraforming_ode*.py`).
- **Conservation**: mass/energy/angular-momentum drift on resting/JW/Held–Suarez
  benchmarks (`test_benchmarks.py`).
- **Physics invariants**: positive temperatures, exact energy closure, CO₂ mass
  budget, frost positivity (`test_physics.py`).
- **Assets**: MOLA checksum, Ames table interpolation (`test_topography.py`,
  `test_ames_radiation.py`).

## Resolved implementation checks

1. **Regolith layer count coupling — resolved.** State initialization accepts the
   forcing and derives the ground-layer count from its skin-depth tuple. Conduction
   also validates the state/configuration layer count before doing array arithmetic.
2. **Frost-point constant vs curve — documented.** The 149.0 K constant is an
   empirical fallback and intentionally differs from the pressure-dependent
   Clausius–Clapeyron result (~147.7 K at 610 Pa). The modes are not interchangeable.
3. **Escape equation — corrected.** `Mars.compute_derivatives` now documents its
   prescribed MAVEN non-thermal escape rate and cap/atmosphere mass exchange; it no
   longer claims to evaluate Jeans escape.
4. **Topography dead code — removed.** The unused half-pixel temporary was deleted.
5. **Atmospheric-mass capability — reconciled.** The original Dinosaur/JCM forcing
   interface could not alter atmospheric mass. This branch implements the anticipated
   deeper extension: CO₂ exchange supplies an explicit `log_surface_pressure`
   tendency, and the wrapped step applies a positivity-preserving atmosphere/frost
   projection. Atmospheric plus surface-reservoir CO₂ is tested for conservation
   when escape is disabled.
