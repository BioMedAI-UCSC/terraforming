# Minimal Mars GCM validation record

This record separates executable evidence from planned paper claims. It covers a
T21, 12-level, 300 s Dinosaur/JAX Mars configuration with Ames-derived
correlated-k CO2 radiation, prescribed constant dust opacity, TES surface
properties, regolith, surface exchange/PBL diffusion, dry convective adjustment,
and an energy-limited surface CO2 reservoir. It does not implement atmospheric
CO2 condensation or establish a complete Mars climate model.

## Results produced on 2026-09-12

- The complete focused GCM suite passed: 127 passed and 3 skipped, including the
  slow dynamical-core benchmarks. The full non-slow package suite passed: 292
  passed, 3 skipped, and 5 deselected.
- A coupled eight-step T21/4 rollout is differentiable with respect to surface
  albedo. The gradient of area-mean surface temperature was -1.03419446 K per
  unit albedo. Centered finite differences at 1e-3 and 1e-4 agreed to relative
  errors of 6.1e-10 and 3.6e-9. This proves a short smooth computational path;
  it does not prove useful long-horizon gradients or behavior at physics switches.
- A new Ls=0 T21/12 run completed 10 sols with finite state. Across checkpoint
  samples, surface temperature remained 145.57--236.41 K and mean surface
  pressure remained 669.65--669.71 Pa. The run advanced from Ls=0 to Ls=5.09.
  It is a transient stability result, not an equilibrated climate result.
- The one-sol Ls=0 snapshot compared with cached MCD v6.1 fixed-local-time maps
  has surface-temperature RMSE 14.99 K and spatial correlation 0.908; surface
  pressure RMSE 64.40 Pa and correlation 0.978; wind-speed RMSE 5.69 m/s and
  correlation 0.298. The model uses daily-mean sunlight while MCD samples were
  averaged over local time, and the model snapshot has not spun up. These values
  are diagnostic and must not appear as climate-validation results.
- The smallest NASA Ames FV3 Beta Output Release 1 file (342 KiB fixed fields)
  was compared directly. Dinosaur's MOLA elevation versus Ames `zsurf` has bias
  -6.83 m, MAE 215.32 m, RMSE 424.57 m, and spatial correlation 0.9892 after
  horizontal interpolation. This verifies boundary-field alignment only.

## Claims currently supported

The repository supports a modular differentiable Mars primitive-equation model,
short coupled finite rollouts, exact restart execution, standard dycore tests,
and reproducible model-reference diagnostics. It supports the narrow statement
that gradients propagate through a short coupled rollout and agree with finite
differences in a smooth case.

It does not yet support claims of an equilibrated present-Mars climatology,
atmospheric agreement with NASA Ames, observational validation, useful gradient
calibration, or a learned residual model. NASA's smallest atmospheric product is
the 2.54 GiB height-level file; it must be downloaded and subset before a matched
temperature/wind comparison. A production result also requires at least one
Mars-year spin-up followed by a separately sampled evaluation period.

## Reproducible artifacts

- `outputs/iclr_validation/gradient.json`
- `outputs/iclr_validation/ls0_t21_l12_dt300/`
- `outputs/iclr_validation/mcd_ls0/benchmark.json`
- `outputs/iclr_validation/ames_fixed_topography.json`
- `data/ames/fv3betaout1/03340.fixed.nc`
- `data/ames/fv3betaout1/fixed_comparison_protocol.json`
