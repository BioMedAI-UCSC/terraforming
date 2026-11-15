# Submission 54586 rebuttal campaign

This campaign uses `experimenal` source, pinned to a commit at submission.
It produces new evidence. It does not relabel the submitted paper's results.

Six independent A100 Jobs run these tasks:

| Task | Protocol | Outputs |
|---|---|---|
| projected | T21/L12, current full physics, two orbital years, 300 steps/sol | Per-step mass corrections and horizontal Courant numbers; roughly hourly surface fields; first-year budgets; second-year reference comparisons |
| unprojected | Identical initial state and protocol, numerical frost repair and global pressure correction disabled | Raw drift, failure if any, and pressure cycle. Physical settling/saturation remain enabled. |
| stability | Current source, common initial state, one year each at 450/300/150 s | Existing raw budgets, extrema, spectra, horizontal Courant and shared-checkpoint fields |
| spatial | 2,048 bounded albedo controls, 300 steps, twelve central finite-difference checks | Sensitivity map; gradient ascent, diagonal CMA-ES and uniform changes; 64 forward calls per arm; three CMA seeds |
| radiation | Seeds 0/1/2, 100 initial updates, 20 per continuation, 3,000 coupled steps | Fixed-update column results and longer T21 coupled checks |
| heating | Seeds 0/1/2, 50 updates, frozen native MACDA windows | Held-out heating pilot results, including weak or negative gains |

`run_rebuttal_experiments.py` implements the new physical and spatial protocols.
`analyse_rebuttal_climate.py` uses the saved hourly fields to integrate the same
daily/five-sol windows as each reference. MCD uses the original four local times
per longitude. It reports independent elevation-regressed log-pressure metrics.
Viking uses quality-screened PDS VL_AVEP daily means at station locations, with
hydrostatic elevation correction from the same MOLA raster. Compare stations to
stations; global means are saved separately.

All inputs and source bytes are hashed. The runtime archives the exact checkout.
Every attempt has a separate directory. Completed tasks are retained on retries.
Nautilus injects a six-hour deadline on standalone Pods and a one-day cleanup
TTL on Jobs. Managed Job Pods have no run deadline. Each science Job has the
manual `aegis.terraforming/retain-results` finalizer, which blocks automatic
deletion. Remove that finalizer only when the user requests cleanup after
reviewing the retained results. A completed Job can show `Terminating` after the
cluster's cleanup attempt; the finalizer retains its record until manual release.
No workload deadline, timeout wrapper or timed deletion is configured. A nonfinite
unprojected result is evidence of failure, not a reason to change the protocol.

## Limits for the paper

- The initial climate is cold-started. A second orbital year is not proof of
  equilibrium; deep-soil convergence must be assessed before climate claims.
- Temporal means use hourly trapezoidal quadrature and interpolated boundaries.
- MACDA is reanalysis; Ames and MCD are model references. Wind heights differ.
- The MCD cache's Ls, local time and scenario are checked against response headers.
  Original high-resolution-topography provenance remains qualified.
- Viking-year dust is not prescribed. This is a seasonal pressure-cycle test,
  not a contemporaneous weather validation.
- Equal forward calls are not equal compute. Gradient arms also pay backward
  calls; the traces record synchronized times. CMA-ES uses diagonal covariance.
- Radiation seeds change both generated data and initialization, as in the
  original script. These are not initialization-only uncertainty estimates.
- Coupled radiation retains the example's four-layer configuration; it is not
  a T21/L12 seasonal climate validation.

The launch record is `deployment.json`. Check pod logs plus each task's saved
log/status. Job completion is separate from scientific acceptance.
