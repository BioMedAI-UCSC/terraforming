# gcm3d data-staging & validation scripts

These scripts stage the external datasets the framework/Mars 3-D GCM needs and
evaluate its output. They are setup or experiment commands rather than runtime
package imports. MOLA uses a pinned checksum; generated references and experiment
bundles record their own provenance and hashes.

| Script | Purpose | Produces |
|--------|---------|----------|
| `stage_mola.py` | Download the MGS **MOLA MEGDR** 4-px/° global topography (`megt90n000cb.img`) from the NASA PDS Geosciences Node; verify SHA-256 + size. | `data/mola/meg004/megt90n000cb.img` (used as the dycore's lower boundary). |
| `stage_ames_co2_radiation.py` | Decode the bundled Ames 12-band Fortran CO₂ k-tables into a compact JAX `.npz` asset (nearly pure-CO₂ mixture, H₂O mass fraction 1e-7). | `package/src/framework/physics/ames_co2_12band.npz` (bundled; consumed by `ames_radiation.py`). |
| `stage_tes_surface.py` | Stage independent MGS/**TES** albedo + thermal inertia to a 1° NetCDF. | Surface boundary file for `run_maps(surface_properties_path=…)`. |
| `benchmark_mcd.py` | Fetch **Mars Climate Database v6.1** maps (global diurnal mean; the MCD web service exposes two free dimensions) and compare a `gcm3d` MOLA-map NetCDF against them. | Area-weighted bias / RMSE / correlation report (uses `mcd.py`). |
| `run_gcm3d_ablation.py` | Restartable Mars-year (668-sol) soil spin-up with incremental PBL/physics ablations — a climate experiment, not a short map. | Restart files (`restart.py`) + diagnostics. |
| `stage_ames_surface_reference.py` | Convert local Ames outputs into reference fields and separate visible/IR dust. | Ames reference NetCDF. |
| `stage_arco_macda.py` | Stream bounded archive subsets, handle gaps and regrid fields; requires the `arco` extra. | Data and provenance. |
| `benchmark_ames.py`, `evaluate_arco_seasonal.py` | Independent-model and reanalysis seasonal comparisons. | Weighted diagnostic reports. |
| `evaluate_seasonal_convergence.py` | Check matched-phase year-to-year repeatability, including deep soil. | Convergence report. |
| `run_mars_paper_experiments.py` | Frozen calibration and paired ablation protocols. | Reports, source/input hashes and samples. |
| `run_differentiable_experiments.py`, `prepare_differentiable_windows.py` | Aligned windows, gradient checks, fitting and paired ablations. | Acceptance, skill and provenance. |
| `neural_temp_data.py`, `run_neural_temp.py`, `train_neural_temp.py` | Frozen native temperature-only postprocessing. | Forecast cache, models and gated test reports. |

## Typical setup

```bash
rtk proxy .venv/bin/python scripts/stage_mola.py
rtk proxy .venv/bin/python scripts/stage_tes_surface.py
# ames_co2_12band.npz is committed; re-run stage_ames_co2_radiation.py only to regenerate it
```

Provenance and integrity constants (source URLs, SHA-256, size) live alongside the
loaders in `topography.py` and `ames_radiation.py`; see
[the gcm3d implementation notes](../package/gcm3d/implementation.md) and
[the physics-limitations ledger](../ideas/gcm3d-physics-limitations.md).

Run `--help` before staging: external archives, output paths and required local
inputs vary. A fresh checkout does not contain the existing `outputs/` bundles.
MCD may need live network access; `tform mars compare` instead reads cached inputs.
Model-reference diagnostics do not establish independent observational accuracy.
See [reference comparisons](../cli/reference-comparison.md),
[calibration](../package/gcm3d/calibration.md),
[temperature-only](../package/gcm3d/temperature-only.md) and
[validation status](../package/gcm3d/validation.md) for full workflows.
