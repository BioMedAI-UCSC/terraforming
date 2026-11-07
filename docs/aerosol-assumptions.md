# Aerosol integration: evidence and assumption ledger

Scope: the aerosol transport, optical/radiation coupling, isolated lifetime
screen, and paper-experiment planner on `integration/aerosol-gpu-experiments`.
This is not a validation of the whole GCM or a complete Mars climatology audit.
Metadata describes a model choice; a declaration in `physics_contract.json`
does not establish its scientific validity. New run configurations archive
that contract's SHA256 and its model-assumption descriptions.

## Unsupported restrictions removed

| Previous behavior | Evidence or reasoning | Change |
| --- | --- | --- |
| Stop below 100 K or above 350 K | No source supported 100 K. The staged Ames gas-opacity temperature axis actually spans 50–350 K. | Derive coupled-run coverage from the asset; compute aerosol optical coefficients on that same interval. It is data coverage, not proof of accuracy at 50 K. |
| Stop outside 1–2000 Pa | No validation source supported those numbers. Ames pressure nodes span 1e-6–1e4 mbar. | Derive pressure coverage from the asset and convert to Pa, about 1e-4–1e6 Pa. |
| Clamp gas-opacity, Planck-fraction and particle-optics lookups to an endpoint | No evidence justifies constant extrapolation of these temperature/pressure tables. | Out-of-table queries return nonfinite results, causing run failure rather than invented endpoint physics. Planck-fraction coverage remains the generated table's actual 50–400 K interval. |
| Default maximum particle mass fraction 0.001 | No measured loading threshold or requested error bound justified that value. | No default cap. An optional caller-selected criterion remains explicitly labeled. Omitting particle backreaction is still an unvalidated dilute approximation. |
| Weight solar particle optics at 6000 K | The number had no cited basis. | Use the IAU 2015 B3 nominal solar effective temperature, 5772 K, and label the within-band blackbody approximation. This is not a measured solar spectral irradiance product. |
| Cap direct-beam slant path at 20 vertical paths (minimum cosine 0.05) | No source justified this cap; it changes illuminated near-horizon columns, not only darkness. | Use the plane-parallel secant in the shared gas and aerosol paths. Set a harmless finite path in exact darkness, where incident flux is zero. |
| Approximate disk drag using a volume-equivalent sphere | No validated disk mobility law was supplied. | Reject disk drag instead of substituting a sphere. Actual spherical collector drag remains supported. |
| Restrict isolated collision kernels to 100–350 K and 1–2000 Pa | These kernels do not use the gas-opacity table, so its coverage cannot justify their domain. | Require positive finite temperature, pressure and gravity. Constitutive-law accuracy outside tested conditions remains unvalidated. |
| Disallow timesteps above 300 s | No universal stability theorem supports a fixed ceiling independent of flow/grid. | Require finite positive dt. Stability and accuracy require refinement; allowing a value does not certify it. |
| Silently select ten sols of spinup | Integration duration does not establish equilibration. | Require `--spinup-sols` explicitly. |
| Interpret the source's unqualified lifetime year as an Earth year by default | The source convention is ambiguous and changes reference inventory. | Require `--lifetime-year earth` or `mars`. This affects comparison curves, not simulated removal. |
| Forbid numerical error criteria above 0.01 | No physical law gives a universal 1% ceiling. | Remove that upper ceiling on transport tolerance; unresolved-mass tolerance is a physical fraction bounded by 1. Both remain numerical accuracy policies. |
| Restrict optional aggregate morphology dimension to at least 1.5 | No cited measurement supported that cutoff. | Admit the idealized chain-to-compact range 1–3; do not call any dimension a measured Mars aggregate morphology. |

The Ames coefficient provenance is recorded in
`scripts/stage_ames_co2_radiation.py`, the NPZ source/hash fields, and independent
anchor/hash tests in `test_ames_radiation.py`. The new bounds are read from the
asset, not copied into a second set of constants.

Solar reference: [IAU 2015 Resolution B3](https://www.iau.org/common/Uploaded%20files/IAUGA2015-Resolution-B3-recommended-nominal-conversion.pdf).

## Retained physics, approximations and experiment choices

| Item | Basis and why retained | Effect and remaining uncertainty |
| --- | --- | --- |
| Aluminium/iron rod dimensions, density and single-particle spectra | Existing Ansari et al. source provenance, supplementary table transcription and checksums; SI conversion and independent opacity cross-checks are tested. | Source simulations of randomly oriented monomers, not measured Mars warming. Other shapes/materials require their own spectra. |
| Optical constants independent of temperature; constant spectral tail beyond 55 µm | The supplied source spectra have no material-temperature dimension; the existing source provenance explicitly declares the constant-tail convention. | Especially consequential for cold longwave radiation. Neither the source convention nor a hash validates cold material optical constants. No measured alternative is supplied. |
| Ames band structure and dry CO2-dominated opacity mixture | Use the actual staged NASA Ames table layout and driest available mixture rather than inventing new absorption coefficients. | Omits evolving gas composition and validated atmospheric CO2 condensation/cloud feedbacks. Table coverage is necessary for interpolation but insufficient for climate validity. |
| Within-band solar blackbody weighting | Planck spectrum at the sourced nominal solar effective temperature provides a reproducible spectral approximation. Ames band fluxes still supply the gas solver's band energies. | Real solar spectral structure is omitted; replace with a measured spectrum for a source-matched radiative comparison. |
| Plane-parallel column geometry and two-stream angular closure | Deliberate reduced radiative-transfer equations; energy/boundary tests and angular-reference comparisons exist. | Spherical twilight geometry and full particle scattering phase functions are not supplied. Conservation tests do not establish angular accuracy or climate validity. |
| Continuum cylinder drag and Epstein free-molecular drag | Existing Tirado et al. (1984), DOI 10.1063/1.447827, and Epstein (1924), DOI 10.1103/PhysRev.23.710, limiting laws. | Continuum circular-cylinder resistance is applied to the source square-section rods. This shape approximation is not chamber validated. |
| Harmonic blending of continuum and free-molecular resistance | A positive interpolation recovering both limiting resistances; there is no supplied measured rod transition law. | An engineering closure, not measured transition-regime mobility. Sensitivity to the blend remains required. Removing drag altogether would also remove physical settling. |
| Diffuse-reflection accommodation, 1+π/8 | A theoretical Epstein limiting reflection model, bounded by the specular and diffuse limits. | The actual rod-surface accommodation is unknown. The stored parameter identifies this idealization, not an inferred measurement. |
| CO2 Sutherland viscosity | Existing COMSOL gas-property formula/constants and explicit engineering-approximation classification. | No empirical validity range for the cold rod experiment was established by this audit. Do not infer one from the opacity table. |
| Random orientation and monomer optics | Matches the orientation-averaged source spectra. | No modeled alignment, aggregate or cloud-mediated optical effects; active aggregation remains rejected without optical data. |
| Flat terrain, emissivity 1, uniform configured albedo | Deliberate homogeneous screening benchmark; emissivity 1 also matches the current active coupling's supported thermal boundary. | Can affect circulation and warming. They are not claims about real Mars. A realistic-surface comparison requires observed fields and verification of the coupling with those boundaries. |
| 210 K, 610 Pa, Ls=0 initial state; albedo 0.25 | Explicit experiment initial/boundary conditions archived in the plan, not an observed 3D atmosphere. | The one-hour spinup used previously is not evidence of equilibration. Longer seasonal convergence and held-out climatology remain necessary. |
| Global, sigma-layer-weighted injection | The injection's longitude, latitude and sigma bounds are explicit experiment inputs; they define where the hypothetical source is applied. | It is not a validated industrial release plume or deployment strategy. Source energy/momentum are assumed equilibrated with ambient gas and omitted. |
| No particle displacement or momentum backreaction | Dilute-particle reduction of the coupled equations; current runs have very small diagnosed particle mass fractions. | There is no validated universal numerical cutoff. Removing the default cap does not make dense particle flows supported. |
| Optional sticking, charge, aggregate dimension and collection efficiency | Sensitivity scenarios; efficiencies are fractions, charge is caller selected, dimension describes idealized geometry. | No measured Mars values are inferred. Prescribed removal requires an explicit lifetime and source; unsupported charged collection and active aggregate optics stay rejected. |
| T21/L12, 60 s dt and prescribed wind/temperature damping | Numerical discretization and stabilization choices recorded in the configuration. The damping follows the existing Mars-map convention. | No resolution/timestep/damping convergence has been established for annual aerosol warming. The damping timescale is not a measured physical mixing constant. |
| Spectral positivity limiter | Tested mathematical construction preserving each layer's area-integrated particle mass and number, preventing representation-induced negative populations. | Redistributes particles horizontally; cumulative redistribution is reported. It is numerical mixing, not validated physical diffusion. |
| Gas projection and bounded transport inventory correction | Existing numerical safeguards. The default transport tolerance 1e-4 is an explicit numerical accuracy policy, not a physical domain limit. | Correction magnitudes and raw errors must be reviewed and refined. Global conservation after correction does not prove local transport accuracy or energy conservation. |
| Three annual means and warming thresholds | Output comparison definitions; source 4 K/35 K references are not model coefficients or success targets. | Three years does not prove equilibrium; a partial-year mean is not an annual climate result. |

Mathematical checks remain: finite arrays, positive physical dimensions and
carrier gas, nonnegative particle inventories, legal units and source intervals,
fraction bounds, verified assets and checkpoint identities. Resource/array-size
budgets are software limits and are not advertised as physical validity domains.

This audit does not establish a validated climate response. In particular, a
36-sol polar upper-air temperature near 100 K still needs physical and numerical
investigation even though it lies inside the source gas-table coverage. Existing
failed runs stay archived; changes to physics/assumptions require fresh source
and suite identities, not an unmarked production resume.
