 # Minimum Plan for a Differentiable Mars Climate Model Paper

  ## Summary

  Freeze the project as a deterministic, end-to-end differentiable Mars GCM. The paper’s core contribution will be a stable seasonal climate model whose physical parameters can be calibrated through
  gradients. Ames, MCD, and ARCO will be reproducible diagnostic references, without claiming close reproduction or observational validation.

  No neural residual, learned closure, water cycle, clouds, interactive dust, photochemistry, or terraforming experiment is required.

  ## Required Work

  ### 1. Freeze the minimum physical configuration

  Use one canonical configuration throughout:

  - T21 horizontal resolution, 12 sigma levels, 300 s timestep.
  - Resolved diurnal and eccentric-orbit forcing.
  - Ames-derived correlated-(k) CO₂ radiation.
  - Prescribed seasonal dust with a Conrath vertical profile and visible/infrared optics.
  - MOLA topography and TES albedo, emissivity, thermal inertia, and roughness.
  - Twelve-layer prognostic regolith.
  - Stability-dependent surface exchange, implicit PBL mixing, and conservative dry convection.
  - Energy-limited surface CO₂ condensation/sublimation with the global mass projection.
  - Fourth-order spectral diffusion with the established 0.1-sol shortest-wave damping time.

  Treat atmospheric CO₂ condensation, frost-dependent optics, water, clouds, interactive dust, and subgrid gravity-wave drag as documented limitations.

  ### 2. Establish an equilibrated stable baseline

  - Resume the corrected diurnal run for complete 668.619-sol periods.
  - After each period, run the existing seasonal convergence evaluator at (L_s=45,135,225,315^\circ).
  - Require year-over-year changes no larger than:
      - 5 Pa global mean surface pressure.
      - 5 Pa-equivalent global mean CO₂ frost.
      - 1 K global mean surface temperature.
      - 0.25 K global mean deep-soil temperature.

  - Accept the baseline only when every field passes at all four seasons.
  - Run one untouched evaluation year after convergence and export daily means plus fixed-local-time samples.
  - Report finiteness, temperature and wind extrema, CO₂ drift, atmospheric energy, axial angular momentum, throughput, and restart reproducibility.

  ### 3. Demonstrate useful differentiability

  Calibrate only interpretable physical parameters:

  - CO₂ longwave-opacity scale.
  - Dust longwave-opacity scale.
  - Surface/PBL drag multiplier.

  Use bounded transforms so optimization cannot produce negative opacity or drag.

  - Demonstrate short-horizon coupled calibration with 0.25 sol (74 physical steps), T21/L12 and a 300 s timestep; retain every physical operator. Verify forward-mode autodiff against centered finite differences at raw-parameter steps 1e-3 and 1e-4, requiring relative error below 1e-4 for every parameter at both step sizes.
  - Use the existing MY32 Ls45 window (target index 5) for the minimum single-window demonstration. Four-season calibration is follow-up work, not a claim of this experiment.
  - Freeze the exact 20/20 optimizer-call budgets before execution. MY33 validation is separate follow-up work; do not label training loss as validation loss.
  - Preserve MY34–35 for the final diagnostic evaluation.
  - Compare gradient-based L-BFGS with a bounded Powell run using identical initialization, bounds, data, objective, and exactly 20 objective calls per method. Count 12 finite-difference forward calls separately; a gradient oracle call is not equal in compute cost to a Powell call. If an optimizer terminates early, restart it at its best evaluated point until the budget is consumed, recording restart counts.
  - Require the gradient method to reduce the predeclared calibration loss and achieve short-horizon calibration loss at least as good as Powell within the same evaluation budget.
  - Keep the calibrated parameters fixed for all subsequent comparisons.

  The objective should combine normalized, Gaussian-area-weighted errors for atmospheric temperature and horizontal winds. Surface pressure can be reported but should not dominate calibration because
  the calibrated parameters do not directly control atmospheric mass.

  ### 4. Produce matched reference comparisons

  Create one common comparison schema containing time, (L_s), local time, latitude, longitude, pressure/sigma level, surface pressure, surface temperature, atmospheric temperature, eastward wind,
  northward wind, wind speed, and CO₂ frost where available.

  For every comparison, record source version, URL, hashes, dust scenario, averaging interval, interpolation, and unavailable variables.

  - ARCO-MACDA: compare daily means over four seasonal windows from MY34–35. Regrid 35 levels and (36\times72) fields to T21/L12. Report the physical and calibrated configurations separately.
  - NASA Ames: use the staged 133-sample FV3 release. Match season, five-sol averaging, vertical coordinate, local time, dust, and boundary fields as closely as the archive permits.
  - MCD v6.1: cache global queries at four seasons and local times 0, 6, 12, and 18 hours. Match altitude, dust scenario, season, and averaging.
  - Report area-weighted bias, MAE, RMSE, relative MAE, and spatial correlation. Add zonal-mean latitude-pressure temperature and wind comparisons.
  - Do not impose accuracy pass/fail thresholds. Label these as diagnostic model/reanalysis comparisons and identify every forcing mismatch.

  ### 5. Complete the paper and reproducibility package

  Rewrite the manuscript around three claims:

  1. The differentiable Mars GCM remains stable through an equilibrated seasonal cycle with controlled CO₂, energy, and angular-momentum budgets.
  2. Its gradients agree with finite differences and support efficient calibration of physical parameters.
  3. It produces reproducible, season- and local-time-matched diagnostics against Ames, MCD, and ARCO.

  Required main-paper evidence:

  - Model and differentiable-parameter architecture diagram.
  - Numerical verification and conservation table.
  - Seasonal-convergence and budget figure.
  - Gradient-check and L-BFGS-versus-Powell calibration figure.
  - ARCO, Ames, and MCD comparison maps and zonal profiles.
  - Physical-versus-calibrated metric table.
  - Runtime, memory, and reproducibility table.

  Remove all neural-residual promises, diagrams, experiment rows, and NeuralGCM-style learned-physics claims. Neural GCMs may remain related work.

  Build with the official venue template, satisfy its page limit and anonymization rules, and run the complete workflow from a clean environment before freezing the PDF.

  ## Test and Acceptance Plan

  - All focused and full non-slow test suites pass.
  - Slow dycore conservation and Held–Suarez tests pass independently.
  - Column radiation closes energy and has finite temperature/pressure gradients.
  - Surface exchange, regolith conduction, convection, and CO₂ phase exchange preserve their declared budgets.
  - Restarted and uninterrupted integrations agree within the existing numerical tolerance.
  - The canonical run passes every seasonal convergence threshold.
  - All exported states, diagnostics, reference inputs, and metrics are finite.
  - Autodiff and finite-difference gradients agree within (10^{-4}) relative error away from known physical switches.
  - Short-horizon gradient calibration satisfies the declared calibration-loss criterion against Powell with exact 20/20 counts. Held-out validation is not established by this gate.
  - Every paper number traces to a committed configuration, manifest, artifact, evaluator, and hash.

  ## Assumptions and Claim Boundaries

  - “Comparable” means matched diagnostic metrics and plots, not tight reproduction of Ames or MCD.
  - ARCO is a data-assimilating reanalysis; Ames and MCD are model references. None is described as observational truth.
  - A separate lander or spacecraft observation experiment is optional for this scoped paper.
  - The current manually selected 0.25 longwave scales are treated as initialization, not final evidence; final values must come from the declared calibration protocol.
  - Failure to equilibrate the deep soil blocks an equilibrium claim. Failure of physical calibration blocks the differentiability-utility claim but does not invalidate the stable simulator itself.
