# NASA AmesGCM / AmesCAP vs. this framework — differences, borrowings

Scan of [nasa/AmesGCM](https://github.com/nasa/AmesGCM) (the model) and its
companion [NASA-Planetary-Science/AmesCAP](https://github.com/NASA-Planetary-Science/AmesCAP)
(the Community Analysis Pipeline), compared against our terraforming codebase.
July 2026. Companion to [mars-model-landscape.md](mars-model-landscape.md).

**Bottom line up front:** AmesGCM is a different *kind* of artifact than ours —
a Fortran forward GCM for scientific fidelity, not a differentiable design
tool. We should copy almost nothing from the *dynamical/physics core* (it
would sink the paper), but there is real, directly-reusable value in (a) their
**analysis-pipeline conventions and derived-variable formulas** (CAP), (b) a
few **physics parameterizations** as reference implementations to distill into
differentiable form, and (c) their **validation datasets and diagnostics**.

---

## 1. What AmesGCM actually is

| | AmesGCM | This framework |
|---|---|---|
| Purpose | Forward simulation for Mars science | Inverse design of terraforming interventions |
| Core | GFDL FV3 finite-volume cubed-sphere dynamical core + Mars physics | 3–4 state ODE (energy/mass balance), Kepler orbit |
| Language | Fortran 76% / Shell 24% | Python + PyTorch |
| Dimensionality | 3-D, cubed-sphere, ~tens of vertical layers | 0-D (→ 1-D banded, feature B7) |
| Differentiable | No | Yes (the entire thesis) |
| Batched/GPU | No (MPI/CPU HPC) | Yes (GPU ensembles) |
| State size | ~10⁶ prognostic | 3–4 (→ ~20) |
| Cost / Mars year | CPU-hours–days | ≪1 s batched |
| Analysis | CAP (Python, netCDF, publication plots) | CLI + mkdocs, ad-hoc |
| Release maturity | v3.2.1, NASA OSA license, decades of dev | pre-release |

**Directory structure of AmesGCM** (for orientation):
- `atmos_cubed_sphere_mars/` — Mars-adapted FV3 dynamics
- `atmos_param_mars/` — physics parameterizations (the interesting part for us)
- `build_run/`, `patches/` — HPC build + third-party integration
- `diagnostics/`, `docs/`, `data/` — analysis, docs, inputs

**`atmos_param_mars/` physics modules** (our reference shopping list):

| Module | Process | Relevance to us |
|---|---|---|
| `Rad/` | Radiative transfer (correlated-k, two-stream) | **High** — gold-standard target for the ML2 neural-RT surrogate |
| `co2micromod_mgcm.F90` | CO₂ ice cloud microphysics + condensation | **High** — reference for B2 frost point & cap exchange |
| `mars_surface.F90`, `surface_flux.F90` | Ground temperature, surface energy balance | **Medium** — reference for thermal inertia / dT/dt term |
| `blkh2omod_mgcm.F90` | Water-ice cloud microphysics | Low (we don't model H₂O clouds) |
| `aerosol.F90`, `dust_update.F90`, `sedim_mod.F90`, `coagulation.F90` | Dust/aerosol lifecycle | **Medium** — reference for B9 dust events & C3 nanoparticles |
| `palmer_topo_drag.F90` | Gravity-wave / topographic drag | None (no resolved circulation) |
| `testconserv.F90` | Conservation checking | **High** — validates our F2 diagnostics approach |
| `null_physics_mod.F90` | Physics-off baseline | Pattern worth mirroring (dynamics-only test) |

---

## 2. Key differences (why we are not a small AmesGCM)

1. **Inverse vs forward.** AmesGCM answers "given parameters, what climate?"
   We answer "given a target climate, what intervention?" Nothing in AmesGCM
   computes a gradient; every terraforming study built on it (and on the LMD
   PCM) is forward-only trial-and-error. That gap is our contribution — do
   not erase it by importing their non-differentiable machinery.
2. **Fidelity vs tractability.** They resolve circulation, dust, clouds,
   3-D radiative transfer. We deliberately collapse to an energy/mass budget
   so the model is differentiable and runs 10⁶× cheaper. Our honest framing
   (see landscape doc) is the *reduced-order tier*, validated against — not
   competing with — AmesGCM/MCD.
3. **No intervention concept.** AmesGCM has no super-GHG injection, no
   albedo/mirror/shield machinery. Our `interventions/` layer has no analog
   there; this is genuinely novel surface area.
4. **Analysis maturity gap (the honest weakness).** CAP is a mature,
   documented, netCDF-based publication-plot pipeline. Ours is ad-hoc. This
   is the one place AmesGCM is unambiguously ahead and worth emulating.

---

## 3. What to directly copy / adapt

Ordered by value-per-effort. "Copy" respects their NASA Open Source Agreement
v1.3 — attribute, and reimplement formulas rather than lifting Fortran.

### C-A. CAP's derived-variable formulas and conventions  ⭐ highest value
CAP's `MarsVars` computes standard Mars diagnostics from raw state with
canonical formulas: potential temperature, geopotential height, column
integrations, mass stream function, wind components, etc. **Adopt the same
variable names, units, and formulas** for our output/plotting layer so our
results are directly comparable to published AmesGCM/MCD figures and legible
to Mars scientists (reviewers).
- **Where in our repo:** a new `analysis/` module + the objectives/plotting
  layer (features E1, F3). Reimplement in torch/numpy.
- **Payoff:** free reviewer credibility; our validation plots (F1) look like
  the ones the community already reads.

### C-B. `MarsCalendar` — Ls / sol / Mars-date conversions  ⭐ cheap, exact
CAP has a battle-tested solar-longitude ↔ sol ↔ Mars-year converter. We
already hand-roll Ls math (`mars.py:156-157`, `276`, `363-364`); adopting
their conventions (Ls=0 at northern spring equinox, MY numbering) removes a
whole class of off-by-a-season bugs and makes our x-axes match every Mars
paper.
- **Where:** small `framework/mars_calendar.py`; reconcile with
  `orbital.py` and the `initial_ls_deg` handling.
- **Payoff:** correctness + comparability, ~half a day.

### C-C. Radiative-transfer reference data for the ML2 surrogate  ⭐ enables ML2
Their `Rad/` correlated-k / two-stream scheme is exactly the high-fidelity
physics we want to distill into a differentiable MLP (ML doc, ML2 — the direct
analog of JAX MD's "GNN trained on DFT"). Even if we don't run their Fortran,
the scheme documents the band structure, gas opacities, and validation points
to target; ideally, generate ΔF/heating-rate training grids from it (or from
the same underlying opacity data).
- **Where:** `experiments/data/rt_grid.py` (ML10 infra).
- **Payoff:** turns ML2 from "invent a surrogate" into "distill a NASA scheme."

### C-D. `testconserv.F90` conservation-check pattern
They ship an in-model conservation tester. Mirror the *idea* directly in our
F2 diagnostics: assert CO₂ mass (atmosphere + caps + regolith) and energy
budgets close every step. Their existence is evidence this is standard,
expected practice — cite it.
- **Where:** `engine/diagnostics.py` (feature F2).

### C-E. CO₂ condensation / cap physics as a reference implementation
`co2micromod_mgcm.F90` + surface modules encode the community-standard CO₂
frost point, latent-heat exchange, and cap energy balance. Use as the
*reference* to (a) sanity-check our simplified cap terms (`mars.py:292-318`),
(b) parameterize B2 (pressure-dependent frost point) and B3 (regolith) with
defensible constants, and (c) cite for parameter provenance.
- **Where:** informs B2/B3 physics; provenance in `/math-review` docs.

### C-F. Namelist-driven configuration ergonomics
AmesGCM/CAP are entirely driven by text namelists + templates ("no coding"
plotting). Our F3 YAML-config system should follow the same philosophy:
every run/figure reproducible from a declarative file. Borrow the *ergonomic
pattern*, not the format.

### C-G. Validation datasets pathway (via CAP `MarsPull`)
CAP's `MarsPull` retrieves MGCM output; the MCMC also distributes reference
climatologies. This is a concrete route to the teacher/validation data our
ML3/ML4/F1 features need (alongside MCD and Viking PDS data).
- **Where:** `experiments/data/` fixtures.

---

## 4. What NOT to copy (and why)

- **The FV3 dynamical core / any 3-D dynamics.** Non-differentiable, Fortran,
  HPC-scale; importing it destroys the entire premise. Our "dynamics" is a
  banded EBM (B7) at most.
- **Full dust/aerosol microphysics chain.** Overkill; we want a *differentiable
  reduced* dust forcing (B9) and nanoparticle model (C3), informed by their
  modules, not the coagulation/sedimentation stack.
- **MPI/HPC build system.** Our scale story is GPU batching, not cluster jobs.
- **netCDF as the internal format.** Fine as an *export* target for
  interoperability with CAP tooling; wrong as our in-memory representation
  (we live in torch tensors). Consider a `to_netcdf()` exporter so CAP/MarsPlot
  can ingest our output — cheap interoperability win, not a core dependency.

---

## 5. Improvements AmesGCM highlights in our codebase

Scanning theirs exposed gaps in ours worth logging:

1. **Analysis/plotting is our weakest layer** vs CAP's maturity — prioritize
   F3 + a small `analysis/` module; it's the difference between "script" and
   "framework" in reviewers' eyes.
2. **No standard Mars diagnostics** (potential temp, column integrals, proper
   Ls handling) — C-A/C-B fix this cheaply.
3. **Conservation checking is aspirational (F2), theirs is shipped** — do F2
   early; it also guards the gradient path.
4. **Parameter provenance is thin.** Their physics constants trace to
   literature; our `mars.py` constants are partly hand-tuned. The ML4
   calibration feature + citing their/MCD values fixes the credibility gap.
5. **Interoperability = free credibility.** A `to_netcdf()` exporter that lets
   CAP's `MarsPlot` render our trajectories would let us show our reduced
   model beside AmesGCM output in the *same* plotting framework — a strong,
   cheap validation figure.

---

## 6. One-paragraph framing for the paper

> "We validate against and borrow diagnostic conventions from the NASA Ames
> MGCM [cite] and its Community Analysis Pipeline, but occupy a
> complementary niche: where the Ames GCM resolves 3-D circulation, dust, and
> cloud microphysics for forward scientific study at ~CPU-days per Mars year,
> our reduced-order model trades that fidelity for end-to-end
> differentiability and GPU-batched execution, enabling gradient-based
> *inverse design* of terraforming interventions — a capability no existing
> Mars climate model provides."
