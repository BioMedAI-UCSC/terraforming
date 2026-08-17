# gcm3d — physics status, limitations, and explicitly-declared absences

This document is the honest ledger for the `gcm3d` 3-D Mars model. It states what
is **verified**, what is an **approximation**, and what is **absent** — so no
output is mistaken for validated climatology. It is maintained against the physics
review checklist. Dates are absolute; "deferred" means not yet implemented, not
"impossible".

## Implementation roadmap (execute in order)

This is the working P0–P2 checklist. A box is checked only when the implementation,
an invariant/benchmark test, and documentation all exist. Work proceeds in numerical
order; later items must not be used to hide a failed earlier budget or dycore test.
The architecture follows JCM's differentiable conventional-physics model on Dinosaur
and preserves the same dynamics-plus-physics-tendencies seam needed by NeuralGCM
([JCM v1.1 paper][jcm], [NeuralGCM paper][neuralgcm], [Dinosaur][dinosaur]).

### P0 — trustworthy differentiable GCM foundation

- [x] **P0.1 — A replaceable column-physics contract.** Keep Dinosaur's prognostic
  state intact and compose explicit physics tendencies through
  `column_primitive_equations(base, parameterization)`. Both conventional packages
  and a future learned residual must return the same structured tendency tree.
  Acceptance: JAX can differentiate a multi-step rollout with respect to a physics
  parameter. This mirrors NeuralGCM's split
  \(\dot{x}=D(x)+P(x,F)\), advanced by an IMEX solver [neuralgcm].

- [x] **P0.2 — Closed surface/atmosphere energy exchange.** Carry prognostic
  surface temperature and apply shortwave/longwave flux once:

  \[
  C_s\,\dot T_s = Q_{SW}^{\downarrow}-Q_{LW}^{\uparrow}-H-L\dot m-G,
  \qquad
  \dot T_k = \frac{H_k}{c_p\,\Delta p_k/g}.
  \]

  The current milestone implements radiation plus a conservative lowest-layer bulk
  sensible flux; CO₂ latent heat is surface-local. Acceptance: surface energy change
  plus atmospheric energy change equals external net radiation, and the result is
  invariant to vertical layer count. The complete Mars surface balance and the same
  latent-energy sign convention are described by [Guo et al. (2009)][guo2009].

- [x] **P0.3 — Closed atmospheric CO₂/frost mass exchange.** Store surface frost in
  pressure-equivalent units and enforce
  \(\dot p_s+\dot p_{ice}=-g\dot M_{escape}/A\). Use the local pressure-dependent
  frost point
  \(T_f=3182.48/(23.3494-\ln p_{hPa})\) and latent heat
  \(L_{CO2}=5.71\times10^5\,\mathrm{J\,kg^{-1}}\), consistent with
  [Piqueux et al. (2016)][piqueux2016]. Acceptance: area-weighted atmosphere plus
  frost is conserved with escape disabled and frost remains non-negative.

- [x] **P0.4 — Kepler-consistent seasonal forcing.** Advance mean anomaly uniformly,
  solve \(M=E-e\sin E\), compute
  \(r=a(1-e\cos E)\), true anomaly, solar longitude, and inverse-square flux.
  Acceptance: perihelion/aphelion flux ratio and observed unequal Mars season lengths.

- [x] **P0.5 — Scientific-data integrity and honest outputs.** Verify the pinned
  MOLA SHA-256 at every load. Carry run duration, fidelity, and transient status
  through NetCDF/API/UI; short spin-ups must be visibly labelled diagnostic rather
  than climatological.

- [x] **P0.6 — Standard dry-dycore benchmark suite.** Add, in order:
  resting hydrostatic atmosphere, solid-body advection, balanced jet/mountain wave,
  and Held–Suarez statistical climate. Record mass, total-energy and axial-angular-
  momentum drift versus timestep and T21/T42/T85 resolution. Acceptance thresholds
  must be fixed before looking at Mars results. Held and Suarez define the canonical
  primitive-equation core intercomparison [Held & Suarez (1994)][held-suarez];
  converged reference solutions are discussed by [Jablonowski (2004)][jablonowski].

  - [x] **P0.6a — Resting isothermal atmosphere.** T21, 12 sigma layers,
    \(\Delta t=120\) s for 100 steps. Fixed thresholds cover atmospheric-mass and
    prognostic-state drift; the measured nonzero mass drift is retained as an
    explicit Dinosaur `log(p_s)` spectral baseline rather than hidden.
  - [x] **P0.6b — Solid-body tracer advection.** A Gaussian passive tracer is
    transported through one analytic \(2\pi\) revolution in the prescribed
    nondivergent flow \(u=U\cos\phi\). Fixed mass, normalized \(L_1\), and
    normalized \(L_2\) return-error thresholds exercise the same spectral scalar
    advection operator used by the primitive equations. This follows standard test
    case 1 of [Williamson et al. (1992)][williamson1992].
  - [x] **P0.6c — Jablonowski–Williamson balanced jet.** The canonical Earth
    analytic state is integrated for one day at T21/12 layers with Dinosaur's
    exponential spectral filter. Fixed thresholds cover atmospheric mass and all
    four prognostic fields. The initial condition and balance equations follow
    [Jablonowski & Williamson (2006)][jw2006]. The perturbed baroclinic evolution
    is retained as part of the resolution-convergence task P0.6e.
  - [x] **P0.6d — Held–Suarez statistical climate.** The CI benchmark spins up for
    60 days and averages 20 daily states at T21/8 layers. Acceptance pins the
    canonical warm-equator/cold-pole contrast, eastward-jet range, minimum
    temperature, and atmospheric-mass drift. It uses Dinosaur's implementation of
    the Newtonian relaxation and boundary-layer drag defined by
    [Held & Suarez (1994)][held-suarez]. A publication comparison must use the
    longer integration and convergence matrix in P0.6e.
  - [x] **P0.6e — Timestep/resolution conservation table.** Global dry mass,
    hydrostatic total energy
    \(\int (c_vT+\Phi+|\mathbf{u}|^2/2)\,dm\), and absolute axial angular
    momentum
    \(\int a\cos\phi(u+\Omega a\cos\phi)\,dm\) are pressure-mass weighted with
    Gaussian quadrature. The fixed 12,000 s resting-atmosphere matrix is:

    | Grid | dt (s) | relative mass drift | relative energy drift | relative AAM drift |
    |---|---:|---:|---:|---:|
    | T21 | 240 | 7.06e-15 | 7.24e-15 | 8.31e-15 |
    | T21 | 120 | 4.41e-15 | 4.57e-15 | 4.37e-15 |
    | T42 | 240 | 4.41e-16 | 3.81e-16 | 2.19e-16 |
    | T42 | 120 | 2.87e-15 | 3.05e-15 | 3.06e-15 |

    All cases are roundoff-limited in JAX float64; T21 improves with timestep
    refinement, while monotonic ordering is not meaningful once T42 reaches
    machine precision.

- [x] **P0.7 — Continuous integration, restart, and averaging.** Serialize the full
  `ColumnPhysicsState` (spectral dynamics, surface temperature, frost, tracers,
  simulation time), reproduce a continuous run bit-for-bit across a restart, and add
  spin-up plus averaging windows. Acceptance: `run(N)` agrees with
  `run(K) -> checkpoint -> run(N-K)` and all budget residuals remain bounded.
  The versioned NPZ+JSON format stores every dycore field, named tracer, simulation
  time, surface temperature, and frost reservoir without pickle. Tests require
  array-exact equality between a ten-step continuous rollout and a 4+checkpoint+6
  rollout. `integrate_with_averaging` separates spin-up from a sampled diagnostic
  averaging window and is verified against manual samples.

### P1 — deterministic Mars physics, JCM style

- [ ] **P1.1 — Multilayer regolith conduction.** Replace the effective areal heat
  capacity with \(\rho c\,\partial T/\partial t=\partial_z(k\partial_zT)\), using
  surface boundary fluxes and a deep zero-flux or prescribed-temperature boundary.
  Support MOLA/TES thermal-inertia maps, where \(I=\sqrt{k\rho c}\).
  Acceptance: analytic sinusoidal skin-depth test, energy closure, and timestep/grid
  convergence. Mars thermal inertia retrievals are reported by
  [Mellon et al. (2000)][mellon2000], and a Mars surface thermal formulation is given
  by [Kieffer (2013)][kieffer2013].
  - [x] Stage independent MGS/TES bolometric albedo and nightside thermal inertia
    from [USGS Astrogeology][tes-albedo]/[NASA PDS][tes-inertia] at one-degree
    resolution with `uv run python scripts/stage_tes_surface.py`. The pinned source
    SHA-256 values are `c91dfcaa1834b96db383beddec537d85e8e82b782682744ee723907ed9e58178`
    (albedo) and `faee32838da40045ee7ebd21bed156cc649620312c01165b556d77a2128258f4`
    (thermal inertia). The model uses an explicit, generic surface-data interface;
    MCD is never selected as a model boundary condition.
  - [x] Add four prognostic soil layers whose thicknesses scale with local diurnal
    skin depth \(\delta=I(\rho c)^{-1}\sqrt{P/\pi}\), conductivity
    \(k=I^2/(\rho c)\), conservative interface fluxes and a zero-flux deep boundary.
    Column surface-plus-soil energy closure and skin-depth scaling are tested.
  - [ ] Complete the prescribed-sinusoid amplitude/phase and timestep-convergence
    acceptance tests before checking P1.1 complete.

- [ ] **P1.2 — Surface drag and stable boundary-layer diffusion.** Implement
  Monin–Obukhov/Richardson-dependent momentum and sensible-heat exchange, plus
  conservative vertical diffusion of heat and tracers. Acceptance: column-integrated
  tracer conservation, kinetic-energy dissipation, neutral log-law limit, and stable
  nocturnal profiles. Mars GCM motivation and established parameterizations are
  documented by [Forget et al. (1999)][forget1999].
  - [x] Implement the neutral surface-layer limit
    \(C_D=[\kappa/\ln(z/z_0)]^2\),
    \(\boldsymbol{\tau}=\rho C_D|\mathbf{V}|\mathbf{V}\), and apply
    \(-\boldsymbol{\tau}/(\Delta p/g)\) to the lowest sigma layer. A direct test
    verifies that the tendency removes kinetic energy and acts only on that layer.
    For the T42/12-layer, 3.5-sol Ls=0 diagnostic, height-matching MCD to the
    exported \(\sigma=0.9583\) layer (~432 m) reduced wind-speed bias from the
    earlier unmatched +13.93 m/s comparison to +0.95 m/s and RMSE to 6.90 m/s;
    spatial correlation remains low (0.18), so stability functions and vertical
    diffusion are still required before P1.2 can be closed.
  - [x] Add bulk-Richardson stable suppression/unstable enhancement consistently
    to aerodynamic sensible heat and momentum exchange.
  - [x] Add pressure-mass-conserving adjacent-layer diffusion of temperature,
    tracers and horizontal momentum below a configurable PBL top. Local gradient
    Richardson stability functions are bounded for explicit stability, with a
    30-minute minimum mixing timescale. Tests verify stable versus unstable exchange
    ordering and column-mean heat/tracer conservation.
  - [ ] Add a stable nocturnal single-column benchmark before checking P1.2
    complete. The
    latest bounded-Richardson coupled transient transports too much upper momentum
    downward (23.68 m/s versus 9.48 m/s MCD at the matched level; RMSE 18.97 m/s),
    so it is not yet accepted.
  - [x] Replace the explicit momentum part with a pressure-mass-conserving
    backward-Euler column solve. Diagnosed kinetic-energy loss is returned as
    heat; tests verify momentum conservation and non-increasing kinetic energy.
    The stable nocturnal single-column/LES comparison remains required before
    P1.2 is complete.

- [ ] **P1.3 — Dry convective adjustment, then nonlocal daytime PBL.** First mix
  statically unstable columns to neutral dry potential temperature while conserving
  column enthalpy. Then add a selectable thermal-plume/mass-flux scheme. Acceptance:
  no superadiabatic resolved layers after adjustment, exact column enthalpy closure,
  and single-column comparison against the LES cases of
  [Colaïtis et al. (2013)][colaitis2013].
  - [x] Add a differentiable dry-column adjustment target: columns containing a
    superadiabatic pair mix to constant potential temperature, choosing that value
    to conserve pressure-mass-weighted enthalpy. Tests verify stability and closure
    to the Dinosaur nodal/modal transform tolerance.
  - [ ] Replace whole-column mixing with neutral-block adjustment and add the
    selectable Colaïtis-style daytime thermal-plume/mass-flux benchmark before
    checking P1.3 complete.

The first combined P1.1–P1.3 T42/12-layer diagnostic is intentionally recorded as
a mixed result rather than a validation claim: MCD temperature spatial correlation
improves from 0.901 to 0.920 as TES surface structure appears, but the 3.5-sol
transient RMSE worsens from 11.35 K to 13.30 K and warm bias grows to 10.64 K because
the soil has not been seasonally spun up. Restart format v2 now includes every
prognostic soil-layer temperature so future long spin-ups remain continuous.

- [ ] **P1.4 — CO₂ gas radiative transfer.** Replace the greenhouse-factor emission
  shortcut with a vertically resolved, differentiable two-stream or correlated-k
  shortwave/longwave package. Heating must be a flux divergence,
  \(\dot T_k=g(F_{k+1/2}-F_{k-1/2})/(c_p\Delta p_k)\), so the sum telescopes to the
  top/surface boundary fluxes. Acceptance: layer-summed closure and comparison with
  published Mars column/GCM benchmarks in [Forget et al. (1999)][forget1999].
  **Current fidelity blocker:** outgoing longwave remains the single-grey proxy
  \(F_{LW}=\epsilon\sigma(T_s/G)^4\). There are no resolved CO₂ 15-µm/NIR bands,
  correlated-k coefficients, two-stream atmospheric fluxes or layerwise radiative
  heating. Consequently the current temperature comparison is not JCM-fidelity
  radiation, regardless of the added surface/PBL physics.
  - [x] Add a differentiable, pressure-scaled two-stream interface with separate
    CO₂ near-IR and 15-µm thermal optical depths. Layer heating is diagnosed from
    interface-flux convergence and a direct test verifies that atmosphere plus
    surface equals the TOA budget to roundoff. This is a compact band model, not
    correlated-k, so P1.4 remains open pending published coefficient tables and
    the Forget et al. column comparison [Forget et al. (1999)][forget1999].

- [ ] **P1.5 — Energy-limited CO₂ phase change.** Replace the tunable relaxation rate
  with complementarity at the frost point: when frost is present, hold
  \(T_s=T_f(p_{CO2})\) and diagnose \(\dot m\) from the residual surface-energy flux,
  \(L\dot m=-Q_{residual}\). Couple atmospheric composition and pressure consistently.
  Acceptance: no frost-point overshoot, non-negative reservoirs, exact mass/latent-
  energy closure, and Viking seasonal-pressure comparison
  [Guo et al. (2009)][guo2009].
  - [x] Add an energy-residual phase-change tendency satisfying
    \(L\dot m=-Q_{residual}\), with atmospheric/frost mass transfer and a direct
    latent-energy closure test. Production maps apply a conservative post-step
    projection: negative stage-level frost is set to zero and the identical
    pressure-equivalent deficit is removed from the atmosphere. Tests verify
    non-negative frost and atmospheric-plus-cap mass conservation over a rollout.
    Viking seasonal-pressure validation remains before P1.5 can be closed.

- [ ] **P1.6 — Prescribed, radiatively active dust.** Start with observed seasonal
  column opacity and a prescribed vertical profile; transport can follow later.
  Implement solar scattering/absorption and infrared absorption/emission within the
  same flux-divergence radiation interface. Acceptance: zero-opacity limit recovers
  clear-sky radiation and heating responds correctly to single-scattering albedo.
  Dust is a primary control on Martian atmospheric temperature
  [Madeleine et al. (2011)][madeleine2011].
  - [x] Add prescribed nodal/scalar visible and infrared optical depth to the
    common two-stream flux solver, including single-scattering albedo. Tests show
    the zero-opacity limit exactly recovers clear sky and nonzero opacity changes
    layer heating. Seasonal opacity staging and an MCD dust-scenario comparison
    remain before P1.6 is complete.

- [ ] **P1.7 — One-Mars-year deterministic baseline.** Run at least one spin-up year
  plus one diagnostic year with checkpointing. Publish zonal means, surface-pressure
  cycle, TOA/surface budgets, and convergence tables. Compare against Viking pressure,
  TES/MCS temperature and the Mars Climate Database; the latter documents a mature
  Mars PCM and its represented processes [MCD v6.1][mcd].
  - [x] Add `scripts/benchmark_mcd.py`: fetch the MCD v6.1 climatology at 12 native
    fixed local times, form a diurnal mean, conservatively preserve units (including
    \(p_{\rm ice}=g_{\rm Mars}m_{\rm ice}\)), interpolate to the model grid, and report
    cosine-latitude-weighted bias, MAE, RMSE and spatial correlation. Winds are sampled
    at 10 m above the local surface rather than the no-slip surface. This is a diagnostic
    harness, not completion of P1.7: transient runs remain scientifically incomparable
    to the equilibrated MCD climatology [MCD web interface][mcd-web].
  - [x] Add `scripts/run_gcm3d_ablation.py`, a restartable 668-sol driver for
    drag, regolith, stability, PBL and convective-adjustment ablations. It writes
    versioned checkpoints, maps, NetCDF and a manifest; a T21 smoke run verifies
    restart and artifact production. The full T42 Mars-year matrix has not yet
    been executed and is therefore not presented as a completed climatology.

### P2 — learned residual physics and extended Mars cycles

- [ ] **P2.1 — Encoder/decoder and learned tendency module.** Add a column-local
  network that consumes resolved column state, surface state, forcing and static
  features, and returns the same `ColumnPhysicsTendencies` as deterministic physics.
  Apply hard constraints or conservative projections to mass and energy tendencies.

- [ ] **P2.2 — Online trajectory training.** Train through the coupled IMEX rollout,
  not only against instantaneous tendencies. Increase rollout length progressively,
  track gradient norms and climate drift, and retain deterministic-physics ablations.
  NeuralGCM reports curriculum growth from short to multi-day rollouts as important
  for stability [neuralgcm].

- [ ] **P2.3 — Dataset and evaluation contract.** Version preprocessing for MCD or
  another established Mars GCM plus spacecraft observations; separate train,
  validation, held-out Mars years/seasons and out-of-distribution pressure/dust cases.
  Define area/mass-weighted RMSE, spectra, extremes and conservation metrics before
  training.

- [ ] **P2.4 — Prognostic dust transport.** Add conservative mass/number tracers,
  sedimentation, lifting and deposition only after the prescribed-dust radiation
  baseline is validated. Compare opacity and temperature structure against the
  semi-interactive framework in [Madeleine et al. (2011)][madeleine2011].

- [ ] **P2.5 — Water cycle and clouds.** Add vapor/ice tracers, surface reservoirs,
  saturation adjustment, sedimentation and cloud radiative effects with closed water
  and energy budgets. This is explicitly after the dry/CO₂/dust baseline; the process
  scope of a mature Mars PCM is summarized in [MCD v6.1][mcd].

- [ ] **P2.6 — Terraforming-regime validity.** Replace fixed present-Mars gas
  properties with composition-dependent \(R,c_p,\kappa\), opacity and collision-
  induced absorption; test pressures and mixtures outside the present-Mars training
  distribution against independent radiative-convective calculations. Do not label
  extrapolated learned results predictive without those tests.

## Verified (with tests)

- **Keplerian orbit.** Mean anomaly advances uniformly; Kepler's equation is
  solved for the eccentric anomaly; true anomaly and heliocentric distance follow
  from it. Perihelion/aphelion flux ratio = ((1+e)/(1-e))² exactly; Mars season
  lengths reproduce 193/179/143/154 sols vs observed 194/178/143/154.
  (`tests/gcm3d/test_orbit.py`)
- **CO₂ frost point.** Pressure-dependent CO₂ saturation temperature
  `T = 3182.48/(23.3494 − ln p[hPa])`: 148 K at 6.1 hPa, 194 K at 1 atm, monotonic.
  (`tests/gcm3d/test_physics.py::TestCO2FrostPoint`)
- **MOLA provenance & integrity.** Pinned source URL + SHA-256 + size; staging
  script; env-configurable path; corrupt/missing raster fails loudly.
  (`scripts/stage_mola.py`, `tests/gcm3d/test_topography.py::TestProvenance`)
- **CO₂ mass budget** integrated with the grid's Gaussian quadrature weights
  (exact over the sphere); atmosphere↔frost exchange conserves total mass to
  <0.1 % with escape off.
- **End-to-end differentiability** of the forced dycore (jax.grad).
- **Closed surface/atmosphere energy exchange.** Shortwave and longwave fluxes act
  once on a prognostic surface temperature. Bulk sensible heat is transferred to
  the lowest atmospheric layer using its mass-dependent heat capacity `cp*dp/g`;
  tests verify surface loss + atmospheric gain equals the external radiative flux
  for multiple vertical layer counts.
- **Surface-local CO₂ latent heat.** Condensation/sublimation energy is applied to
  the prognostic surface reservoir rather than duplicated through the atmosphere.

## Approximations (implemented, but NOT physically complete)

- **Surface "heat capacity" is not a thermal model (P1, deferred).**
  `thermal_inertia` (60000) is used as an effective areal heat capacity, not a
  regolith conduction model. No subsurface layers, no diurnal skin depth, no
  terrain-dependent albedo/inertia.
- **Atmospheric mass coupling is partial (P1).** Evolving surface pressure enters
  the hydrostatic state and the CO₂ supply gate, but does **not** yet modulate heat
  capacity, radiative optical depth, or surface exchange. Injected compounds change
  only a scalar greenhouse factor (and pressure) at snapshots — not cp, R, opacity,
  or atmospheric mass in the 3-D physics.
- **Runs are short transients, not climatology (P0).** `run_maps` results are
  spin-up snapshots (e.g. `fast` = 700×450 s ≈ 3.6 sols). Fields flagged
  `is_transient=True` and the `physics` label says so. A seasonally-equilibrated
  interpretation needs ≥ 1 Mars year (668 sols) + spin-up + an averaging window;
  restart/checkpoint support is **not yet implemented**.
- **Scale presets are resolution tiers, not converged results.** `fast/balanced/
  high/ultra` set grid/steps only. No timestep- or resolution-convergence study has
  been run, so "high"/"ultra" denote grid size, **not** demonstrated convergence.

## Absent processes (explicitly declared)

None of the following are represented:

- Dust radiative heating and transport.
- CO₂ (and other gas) infrared absorption / real radiative transfer — the scheme
  is single-band grey.
- Convection / boundary-layer mixing.
- Stability-dependent surface exchange (neutral momentum drag is implemented).
- Vertical diffusion.
- Water cycle, clouds, ice.

## GCM-intervention semantics (explicitly declared)

A "gcm intervention" run is a **0-D global-mean terraforming trajectory with
independent 3-D diagnostic snapshots**, *not* a continuously-integrated 3-D
transient. Each snapshot re-spins-up from rest at that year's global-mean surface
pressure and greenhouse factor; winds, 3-D temperature, frost, and model time are
**not** carried forward between snapshots, and seasonal phase is set only by the
requested Ls. Continuous integration/restart is the preferred future design and is
deferred.

## Validation status (P2, deferred)

The current `tests/` are primarily **implementation tests** (shapes, finiteness,
invariants and parity), not scientific validation. Dry-dycore benchmarks and an
MCD map-comparison harness now exist, but the MCD result is a short transient versus
climatology diagnostic—not validation. Viking/TES/MCS comparisons, equilibrated
seasonal runs and quantitative acceptance thresholds remain outstanding. Until
those exist, gcm3d output is **not** validated against Mars observations.

## Primary references

[jcm]: https://doi.org/10.5194/gmd-19-6451-2026
[neuralgcm]: https://doi.org/10.1038/s41586-024-07744-y
[dinosaur]: https://github.com/neuralgcm/dinosaur
[held-suarez]: https://doi.org/10.1175/1520-0477(1994)075%3C1825:APFTIO%3E2.0.CO;2
[williamson1992]: https://doi.org/10.1175/1520-0477(1992)073%3C0821:STSNMS%3E2.0.CO;2
[jablonowski]: https://doi.org/10.1175/MWR2788.1
[jw2006]: https://doi.org/10.1256/qj.04.21
[forget1999]: https://doi.org/10.1029/1999JE001025
[guo2009]: https://doi.org/10.1029/2008JE003302
[piqueux2016]: https://doi.org/10.1002/2016JE005034
[kieffer2013]: https://doi.org/10.1029/2012JE004164
[mellon2000]: https://doi.org/10.1029/1999JE001088
[colaitis2013]: https://doi.org/10.1002/jgre.20104
[madeleine2011]: https://doi.org/10.1029/2011JE003855
[mcd]: https://www-mars.lmd.jussieu.fr/mars/info_web/ddd_MCD_v6.1.pdf
[mcd-web]: https://www-mars.lmd.jussieu.fr/mcd_python/
[tes-albedo]: https://astrogeology.usgs.gov/search/map/mars_mgs_tes_global_bolometric_albedo_map_7410m
[tes-inertia]: https://pds-geosciences.wustl.edu/missions/mgs/tes-timap.html
