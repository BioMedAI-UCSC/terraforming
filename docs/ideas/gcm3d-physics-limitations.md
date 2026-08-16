# gcm3d — physics status, limitations, and explicitly-declared absences

This document is the honest ledger for the `gcm3d` 3-D Mars model. It states what
is **verified**, what is an **approximation**, and what is **absent** — so no
output is mistaken for validated climatology. It is maintained against the physics
review checklist. Dates are absolute; "deferred" means not yet implemented, not
"impossible".

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

## Approximations (implemented, but NOT physically complete)

- **Radiation energy budget — NOT column-conserving (P0, deferred rework).**
  The grey surface energy balance `dT/dt=(Q_in−εσ(T/gh)⁴)/C` is currently applied
  *uniformly to every atmospheric layer*. This does **not** conserve column energy
  (it effectively adds the surface flux `n_layers` times) and there is **no
  distinct surface temperature**. It behaves as a grey Newtonian-style relaxation
  that produces plausible horizontal temperature *structure*, not a closed energy
  budget. The correct rework (distinct surface temperature + surface heat capacity,
  surface↔atmosphere sensible-heat exchange, mass-weighted atmospheric heating,
  and an energy-conservation diagnostic) is **not yet done**.
- **Latent heat — duplicated across layers (P0, deferred with the above).**
  CO₂ condensation latent heat is applied column-uniform for stability, so it is
  likewise duplicated per layer rather than released at the surface/lowest layer.
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
- Surface drag / turbulent momentum flux.
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

The current `tests/` are **implementation tests** (shapes, finiteness, invariants,
parity), not scientific validation. There is **no** observational benchmark suite
yet: no Viking surface-pressure seasonal curves, no MCD/TES map or cross-section
comparison, no dry-dycore benchmarks (solid-body rotation, Rossby–Haurwitz,
mountain wave, Held–Suarez), and no quantitative acceptance thresholds. Until those
exist, gcm3d output is **not** validated against Mars observations.
