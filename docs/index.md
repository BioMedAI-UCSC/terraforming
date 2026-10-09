# Differentiable Mars GCM

**tform** models Mars's spatial atmosphere and surface with a differentiable
three-dimensional general-circulation model. The primary workflow uses JAX and
Dinosaur to evolve circulation, radiation, surface temperature, regolith and CO₂
frost over MOLA terrain.

Start with [installation](getting-started/installation.md), then
[run your first GCM map](getting-started/quickstart.md). For how a simulation is
built, read the [architecture overview](architecture/planet.md) and
[Mars configuration](architecture/mars.md).

## What you can do

| Workflow | Guide |
| --- | --- |
| Generate pressure, temperature, wind and frost maps | [GCM quickstart](package/gcm3d/quickstart.md) |
| Explore fields and controls in a browser | [Visualizer](architecture/visualizer.md) |
| Differentiate and calibrate physical controls | [Calibration and ablations](package/gcm3d/calibration.md) |
| Train replacement radiation or bounded heating | [Neural experiments](package/gcm3d/neural-experiments.md) |
| Fit a temperature correction to frozen forecasts | [Temperature-only postprocessing](package/gcm3d/temperature-only.md) |
| Compare against Ames, MCD and ARCO-MACDA | [Reference diagnostics](cli/reference-comparison.md) |
| Assess numerical and scientific evidence | [Validation and limitations](package/gcm3d/validation.md) |

## Implementation

| Module | Role |
| --- | --- |
| `src.framework.gcm` | Body constants, spectral/sigma coordinates, dry dynamics, restart and parameterized integration |
| `src.framework.physics` | Radiation, surface exchange, regolith, PBL, convection and CO₂ exchange |
| `src.framework.neural` | Column models, radiation adapters, bounded heating and inference checkpoints |
| `src.celestials.planets.mars` | Mars constants, forcing factories, terrain/boundary data and map exports |
| `apps/mars-calibration` | Coupled physical calibration and reproducible experiment protocols |
| `cli` and `ui` | Map commands, comparison reports and browser visualization |

Software tests and reference diagnostics support specific implementation claims.
They do not establish observational accuracy, fully equilibrated climate or
predictive high-pressure terraforming behavior. Read the validation guide before
interpreting simulated maps as Mars climate predictions.

!!! warning "Deprecated global-mean model"
    The torch global-mean model and its `fast`/`accurate` ODE workflows are
    deprecated. Use the GCM for new work. Their documentation remains available
    under [Deprecated global-mean model](deprecated-global-mean.md) for existing users.
