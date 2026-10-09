# Physical calibration and ablations

`apps/mars-calibration` differentiates coupled GCM trajectories with three bounded
physical controls:

| Control | Forcing field | Meaning |
| --- | --- | --- |
| CO₂ longwave | `ames_co2_longwave_opacity_scale` | Gas thermal optical-depth multiplier |
| Dust longwave | `ames_dust_longwave_opacity_scale` | Prescribed dust thermal optical-depth multiplier |
| Surface exchange | `surface_exchange_multiplier` | Consistent heat/momentum bulk exchange multiplier |

These controls do not change composition or gas constants. Short-window loss
reduction is not evidence of equilibrated calibrated climate.

```bash
rtk proxy uv pip install -e ./apps/mars-calibration
rtk proxy .venv/bin/python scripts/run_mars_paper_experiments.py --help
rtk proxy .venv/bin/python scripts/run_differentiable_experiments.py --help
```

The frozen input bundle contains restart, targets, forcing and checksums.
`outputs/nautilus-calibration-inputs` in existing examples is a prepared local
artifact, not a dataset shipped with a fresh checkout. Prepare it using the
application README in the repository.

## Experiment protocols

The original driver compares L-BFGS-B/Powell with fixed oracle budgets and
separate finite-difference checks. The paper runner adds paired full-physics and
operator-removal cases. Its annual configuration spans 59,356,800 seconds,
197,856 steps at 300 s; snapshots are instantaneous and RMSE is a sample average.

The newer differentiable runner offers `gradients`, `calibration`, `neural`,
`prepare-synthetic`, and `ablations`. Physical fitting uses forward-mode AD;
neural fitting uses reverse-mode AD and bounded heating. It uses Adam and norm
clipping, so its update budgets differ from the original oracle budgets.

The [experiment runbook](../../differentiable-experiments.md) provides input
preparation, alignment, gradient checks and reporting commands. Use fresh output
directories. Inspect acceptance fields and source hashes; completion alone does
not establish scientific success. `--require-gpu` rejects CPU fallback.

## Checkpointed climate integration

With the named TES and Ames inputs staged, start with a one-sol pilot in a fresh
directory before running this substantial workload:

```bash
rtk proxy .venv/bin/python scripts/run_gcm3d_ablation.py \
  data/tes/mgs_tes_surface_1deg.nc \
  --output-dir outputs/climate-baseline --config convection \
  --truncation T21 --layers 12 --dt 300 --diurnal --initial-ls 0 \
  --sols 2006 --chunk-sols 5 --cooldown-seconds 0 \
  --co2-lw-scale 0.25 --dust-lw-scale 0.25 \
  --surface-exchange-multiplier 1.0 \
  --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc
```

`convection` is the cumulative full-physics configuration. Resume a matching run
with `--resume`. Explicit 0.25 scales match the corrected baseline; script defaults
are 1.0. Check phase-matched repeatability, including deep soil, rather than simply
counting elapsed years. See [validation](validation.md).
