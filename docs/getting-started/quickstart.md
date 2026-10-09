# Run your first Mars GCM

Complete [installation](installation.md), including the GCM extra and MOLA
staging. Run commands from the repository root using the installed environment.

## 1. Export a first map

```bash
rtk proxy .venv/bin/python -m cli.main mars maps \
  --scale fast --truncation T21 --layers 12 --dt 300 --steps 100 \
  --name first-gcm-map
```

This short T21/L12 run uses daily-mean grey radiation and energy-limited CO₂
exchange. Outputs appear under `outputs/gcm3d_maps/first-gcm-map/` as PNGs and
NetCDF. It verifies the workflow; it is not a climatology or the full browser
physical baseline. `fast` is a resolution preset, distinct from the deprecated
global-mean model's `Accuracy.FAST` integrator.

## 2. Understand the fields

| Output | Units | Interpretation |
| --- | --- | --- |
| Temperature | K | Surface skin in forced runs; lowest-layer air in dry runs |
| Pressure | Pa | Terrain-dependent surface pressure |
| `u`, `v`, wind speed | m/s | Lowest sigma-layer winds, not fixed 10 m height |
| CO₂ frost | Pa-equivalent | Surface reservoir; mass per area is this value divided by gravity |
| Elevation | m | Regridded MOLA lower boundary |

Read `temperature_kind`, `insolation_sampling`, wind-height and climate-status
metadata before comparing references. The final map is an instantaneous sample.

## 3. Explore in the browser

```bash
rtk proxy .venv/bin/python -m cli.main serve --no-browser
```

Open `http://127.0.0.1:8000` and select GCM. The server enables correlated-k CO₂
radiation, regolith, stability exchange, PBL diffusion and dry convection,
with spectral diffusion and available TES/seasonal Ames dust data. Its explicit
CO₂/dust longwave multipliers are 0.25 and surface exchange multiplier is 1.0.
This differs from the simple CLI map configuration.

## 4. Configure Python and restart

The [Python GCM quickstart](../package/gcm3d/quickstart.md) contains complete
forcing, map/export and restart examples. Use `run_maps` for reporting/deployment,
and parameterized step/rollout kernels for differentiated objectives.
Start with float64 and an explicit stable timestep.

## 5. Choose the next experiment

- [Physical calibration and paired ablations](../package/gcm3d/calibration.md).
- [Neural radiation and bounded heating](../package/gcm3d/neural-experiments.md).
- [Frozen temperature-only postprocessing](../package/gcm3d/temperature-only.md).
- [Reference comparisons](../cli/reference-comparison.md).
- [Validation and limitations](../package/gcm3d/validation.md).

A Mars year is about 668.62 sols. Equilibrium needs matched-phase convergence,
including deep soil, rather than elapsed duration alone.

!!! warning "Deprecated global-mean model"
    Use GCM workflows for new simulations. `tform mars run` and torch ODE
    examples are retained under [deprecated documentation](../deprecated-global-mean.md).
