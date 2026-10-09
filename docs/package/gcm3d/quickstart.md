# Installation and first 3-D map

Run from the repository root with Python 3.12+. RTK is optional for readers;
remove `rtk proxy` if it is unavailable.

```bash
rtk proxy uv sync --all-packages --extra gcm3d --dev
rtk proxy .venv/bin/python scripts/stage_mola.py
rtk proxy .venv/bin/python -c 'import jax; print(jax.devices())'
rtk proxy .venv/bin/python -m cli.main mars maps --scale fast --name first-map
```

MOLA staging verifies the pinned raster in `data/mola/meg004/`. The Ames CO₂
table is bundled. TES surface and prescribed dust files are additional inputs;
see [data staging](../../scripts/gcm3d-data.md). CPU JAX is sufficient for small
runs; GPU acceleration needs a compatible GPU-enabled JAX installation.

CLI maps write PNG and NetCDF into `outputs/gcm3d_maps/<name>/`. Defaults use
daily-mean grey radiation and CO₂ exchange, rather than the server's full
physical baseline. `--surface-properties` attaches staged TES data and its surface
physics. Use `--ls` for the starting season and `mars maps --help` for overrides.

```bash
rtk proxy .venv/bin/python -m cli.main mars maps --diurnal --dt 300 --steps 100 --name diurnal-smoke
rtk proxy .venv/bin/python -m cli.main mars maps --no-physics --steps 100 --name dry-smoke
```

## Python configuration

```python
import jax
jax.config.update("jax_enable_x64", True)

from dataclasses import replace
from src.celestials.planets.mars.gcm import radiative_forcing, co2_forcing
from src.celestials.planets.mars.maps import run_maps, save_maps

forcing = replace(
    radiative_forcing(co2_radiation_enabled=True, diurnal=True),
    regolith_enabled=True, stability_exchange_enabled=True,
    pbl_diffusion_enabled=True, convective_adjustment_enabled=True,
    ames_co2_longwave_opacity_scale=0.25,
    ames_dust_longwave_opacity_scale=0.25,
)
fields, final_state = run_maps(
    truncation="T21", n_layers=12, dt_seconds=300.0, n_steps=100,
    forcing=forcing, co2_forcing=co2_forcing(),
    hyperdiffusion_tau_seconds=0.1 * forcing.rotation_period_s,
    return_final_state=True,
)
save_maps(fields, "outputs/first-python-map")
```

This short run uses scalar surface properties and zero dust opacity. Matching
the browser baseline also requires its TES fields, prescribed dust, epoch,
initial state and duration. `forcing_with_ames_dust_climatology` attaches evolving,
separate visible/IR dust; the fixed-season helper uses a different IR approximation.

Forced `temperature_k` maps are surface temperature; dry maps use lowest-layer
air temperature. Winds are at the lowest sigma layer, not fixed 10 m height.
Read `temperature_kind` and sampling metadata before comparing datasets.

## Restart

```python
from src.framework.gcm.restart import save_restart, load_restart

save_restart(final_state, "outputs/first-python-map/restart.npz")
state = load_restart("outputs/first-python-map/restart.npz")
fields, next_state = run_maps(
    truncation="T21", n_layers=12, dt_seconds=300.0, n_steps=100,
    forcing=forcing, co2_forcing=co2_forcing(),
    hyperdiffusion_tau_seconds=0.1 * forcing.rotation_period_s,
    initial_state=state, return_final_state=True,
)
```

Keep grid, units, precision, forcing, diffusion and closure settings consistent.
A restart stores state, not the whole experiment configuration. See
[calibration and ablations](calibration.md) for checkpointed climate campaigns.

Diurnal forcing requires `dt <= rotation_period/(2*n_lon)`. Passing this guard
does not prove dynamical stability. Inspect finiteness, budgets and convergence.
A Mars year is about 668.62 sols; elapsed duration alone does not prove equilibrium.
