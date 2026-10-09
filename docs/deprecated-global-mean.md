# Deprecated global-mean model

!!! warning "Deprecated: use the 3-D GCM for new work"
    The torch global-mean model, its ODE integration modes and associated run
    workflows are deprecated in the documentation. Existing implementation remains
    available for compatibility; this documentation change does not remove APIs.

Start new work with the [GCM installation](getting-started/installation.md),
[quickstart](getting-started/quickstart.md) and [architecture](architecture/planet.md).

## Existing workflows

`tform mars run` and its `sol`, `year`, `multi`, `spots` and `intervention` modes
use the reduced/global-mean path. `Accuracy.FAST` and `Accuracy.ACCURATE` select
its analytic update or RK4 method; they are not GCM resolution or accuracy tiers.
The GCM uses a separate IMEX integration path.

The browser's multi-year intervention trajectory still depends on this model.
Its GCM maps are independent spin-ups at selected global-mean states, not a
continuous 3-D simulation of gas injection. Deprecation does not imply that a
replacement continuous 3-D intervention implementation already exists.

## Reference for existing users

- [Torch planet architecture](architecture/global-mean-planet.md).
- [Torch Mars derivation](architecture/global-mean-mars.md).
- [Global-mean engine API](api/engine.md).
- [Intervention API](api/interventions.md) and [architecture](architecture/interventions.md).
- [Global-mean presets](cli/presets.md) and [command reference](cli/commands.md).

For existing installations the CLI remains callable, for example:

```bash
rtk proxy .venv/bin/python -m cli.main mars run --preset current-mars --type year
```

Use this reference to interpret or maintain existing experiments; select the
GCM workflow for new simulations.
