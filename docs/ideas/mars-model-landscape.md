# Mars Climate Model Landscape — where this framework sits

Background for the ICLR 2027 related-work and positioning sections. Companion
to [iclr2027-direction.md](iclr2027-direction.md). Gathered July 2026.

---

## 1. The established hierarchy of Mars atmosphere models

### Tier 1 — Full 3-D GCMs (the reference standard)

**NASA Ames Mars GCM.** Lineage back to Leovy & Mintz (1969) and the
Pollack–Haberle model of the 1980s–90s. The public **Legacy** version couples
a modified ARIES/GEOS dynamical core to Mars physics packages: surface/ground
temperature model, PBL scheme, CO2 and H2O sublimation/condensation, water-ice
cloud microphysics, semi-interactive dust, two-stream correlated-k radiative
transfer. Open source: [nasa/legacy-mars-global-climate-model](https://github.com/nasa/legacy-mars-global-climate-model),
[NASA software catalog](https://software.nasa.gov/software/ARC-18710-1),
[tutorial NASA/TP-20210023086](https://ntrs.nasa.gov/api/citations/20210023086/downloads/NASA%20Ames%20Mars%20Global%20Climate%20Model%20(GCM)%20Tutorial%20Legacy%20Version%20Tutorial%20Binder.pdf).
The modern version (v3+, [Mars Climate Modeling Center](https://www.hou.usra.edu/meetings/tenthmars2024/pdf/3417.pdf))
swaps in the NOAA/GFDL FV3 cubed-sphere dynamical core — six tiles, avoiding
the polar singularity that matters for the CO2 cap cycle. Fortran, CPU, MPI.

**MarsWRF / PlanetWRF.** Mars instantiation of PlanetWRF
([Richardson et al. 2007, JGR](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2006JE002825)),
derived from the terrestrial WRF mesoscale model. Finite-difference primitive
equations on an Arakawa-C grid; default global grid 36 lat × 64 lon,
40 modified-sigma layers from 0–80 km; signature feature is **grid nesting** —
global-to-regional-to-local in one model. Fortran, CPU.

**LMD Mars GCM → Mars PCM.** Forget et al. (1999); developed at LMD/CNRS with
Open University, Oxford, IAA
([Mars PCM site](http://www-planets.lmd.jussieu.fr/),
[Forget 2022 overview](https://insu.hal.science/insu-03690056)). Typical grid
5.625° lon × 3.75° lat, 32 layers to ~2×10⁻³ Pa; full dust cycle, water cycle,
photochemistry, radiative transfer. Its output, validated against spacecraft
observations, is packaged as the **[Mars Climate Database (MCD v6.1)](https://meetingorganizer.copernicus.org/EPSC2022/EPSC2022-786.html)** —
the community's lookup standard, and our natural validation/teacher dataset
(workplan T11/T12).

State dimensionality of this tier: O(10⁵–10⁶) prognostic variables
(grid × layers × species). Cost: CPU-hours to days per simulated Mars year.
None are differentiable; none run batched on GPU.

### Tier 2 — 1-D radiative-convective models

Column models with detailed vertical radiative transfer but no dynamics.
The key one for us: **[Marinova, McKay & Hashimoto (2005), JGR 110, E03002](https://doi.org/10.1029/2004JE002306)** —
radiative-convective modeling of super-greenhouse-gas warming of Mars; the
source of the RF efficiencies in
[compounds.py](../../package/src/interventions/compounds.py) and the linear
optically-thin ΔF approximation in
[forcing.py](../../package/src/interventions/forcing.py).

### Tier 3 — Reduced-order / energy-balance / analytic models

**This is our model class.** Precedents in the terraforming literature:

- McKay, Toon & Kasting (1991, *Nature* 352) — "Making Mars habitable" —
  feasibility arguments on simplified climate models.
- Zubrin & McKay (1993) — analytic model of the CO2 polar-cap/atmosphere
  feedback (runaway regimes, hysteresis) — conceptually the closest ancestor
  of our T/P/M_ice system.
- Wordsworth, Kerber & Cockell (2019, *Nature Astronomy*) — silica aerogel
  regional warming, simplified thermal modeling.

Our model: 3 prognostic variables (T_surf, P_surf, M_ice split N/S) + Kepler
orbit + diurnal/seasonal insolation + Marinova-style GHG forcing. State is
~10⁵–10⁶× smaller than a GCM; a Mars year at dt=1 h is ~16k cheap tensor ops.

### Adjacent: ML work on the Martian atmosphere (the ICLR-relevant neighbors)

- **[Towards a Foundation Model for the Martian Atmosphere](https://arxiv.org/abs/2605.28851)**
  (Roy et al., May 2026) — a *roadmap/position* paper: train foundation models
  on Mars reanalysis + retrievals because GCMs are too expensive at mesoscale.
  No implementation yet, no differentiability. Useful signal: the ML-for-Mars
  niche is opening, and nobody occupies "differentiable Mars simulator."
- **[Ansari, Kite et al. (2024), Sci. Adv.](https://arxiv.org/abs/2409.03925)** —
  nanorod aerosol warming (>5000× more effective than gases), evaluated with
  two climate models (1-D + 3-D GCM). State-of-the-art terraforming
  *intervention analysis* — but forward-simulation only: candidate
  interventions are hand-designed, then checked. No inverse design.
- GCM emulation / interpretable ML on lander data (e.g.
  [MSL relative-humidity ML, PSJ 2024](https://iopscience.iop.org/article/10.3847/PSJ/ad25fd)) —
  learning *from* Mars models/data, not optimizing *through* them.

---

## 2. Comparison table (paper Table 1 candidate)

| | Ames MGCM (Legacy/v3) | MarsWRF | LMD Mars PCM / MCD | Marinova 2005 (1-D RC) | Zubrin–McKay 1993 | **This work** |
|---|---|---|---|---|---|---|
| Dimensionality | 3-D | 3-D (nestable) | 3-D | 1-D column | 0-D analytic | 0-D/2-box ODE |
| Prognostic state | ~10⁶ | ~10⁵–10⁶ | ~10⁶ | ~10²–10³ | ~2–3 | **3** |
| Dust / clouds / circulation | ✓ | ✓ | ✓ | partial (RT only) | ✗ | ✗ |
| CO2 cap cycle | ✓ (microphysics) | ✓ | ✓ | ✗ | ✓ (feedback) | ✓ (energy-budget) |
| GHG intervention forcing | ✗ | ✗ (added ad hoc in studies) | ✗ | ✓ | ✓ | ✓ (7-compound registry) |
| Language / hardware | Fortran / CPU-MPI | Fortran / CPU | Fortran / CPU | Fortran | closed-form | **PyTorch / GPU** |
| Cost per Mars year | hours–days | hours | hours | minutes | instant | **≪1 s (batched)** |
| Batched ensembles | ✗ | ✗ | ✗ | ✗ | n/a | ✓ (BatchedMars) |
| **Differentiable end-to-end** | ✗ | ✗ | ✗ | ✗ | ✗ | **✓ (goal, workplan Phase 1)** |
| Open source | ✓ | on request | ✓ (MCD access) | ✗ | n/a | ✓ (with paper) |

## 3. Positioning implications for the paper

**Claim (defensible):** the first *differentiable, GPU-batched* planetary
climate model for intervention *design*. Every existing Mars model — including
the ones used in the strongest terraforming study to date (Ansari/Kite 2024) —
is forward-only: you propose an intervention, run the model, inspect. Nothing
in the hierarchy supports gradient-based inverse design, RL-in-the-loop, or
10³–10⁴-member GPU ensembles. That capability gap, not fidelity, is the
contribution.

**Do not claim (indefensible):** fidelity anywhere near Tier 1. No dust cycle
(the dominant interannual variability driver), no circulation/heat transport,
no cloud microphysics, no regolith CO2 adsorption reservoir (a known major
term in warming scenarios — worth an explicit limitations paragraph), no
photochemistry. Frame as: the differentiable analog of the Zubrin–McKay /
energy-balance tier, with Marinova-calibrated forcing — the tier the
terraforming literature already uses for scoping studies.

**Validation strategy (feeds workplan T12):** reproduce (a) the observed
Viking annual pressure cycle (~610↔900 Pa CO2 seasonal swing) and diurnal
temperature range, and (b) MCD annual T/P climatology at the VL1 site.
Report discrepancies vs the LMD PCM/MCD as the honest fidelity statement.

**Teacher data (feeds workplan T11):** MCD v6.1 climatologies as the target
for the learned closure — "reduced-order model + NN closure trained through
the integrator approaches MCD fidelity at 10⁶× lower cost" is a much stronger
result than closing the FAST-vs-RK4 gap alone.

**Framing hook vs the foundation-model trend:** Roy et al. 2026 argue for
big data-driven emulators of Mars GCMs; we argue the complementary point —
for *design* problems you want a small mechanistic model you can differentiate
through, not a large emulator you can only sample. One paragraph of related
work should draw exactly this contrast.

**Reviewer risk:** planetary scientists will ask "why not differentiate
through a GCM emulator?" Prepared answer: emulators interpolate the training
distribution — terraforming trajectories (P → 10⁴–10⁵ Pa) are far outside any
Mars-observation training set; a mechanistic reduced model extrapolates by
construction, and the NN closure (T11) is bounded to a correction role.
