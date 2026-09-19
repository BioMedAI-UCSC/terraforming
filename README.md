<div align="center">

<img src="docs/assets/logo.svg" alt="Terraforming — differentiable planetary climate framework" width="560">

# Terraforming

### A reusable differentiable framework for building planetary general circulation models

[![Tests](https://github.com/BioMedAI-UCSC/terraforming/actions/workflows/tests.yml/badge.svg)](https://github.com/BioMedAI-UCSC/terraforming/actions/workflows/tests.yml)
[![Deploy Docs](https://github.com/BioMedAI-UCSC/terraforming/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/BioMedAI-UCSC/terraforming/actions/workflows/docs-deploy.yml)
[![Documentation](https://img.shields.io/badge/docs-project_documentation-2f9e6f)](https://biomedai-ucsc.github.io/terraforming-docs/)
[![Python](https://img.shields.io/badge/python-3.12-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![JAX](https://img.shields.io/badge/JAX-differentiable_GCM-orange)](https://github.com/jax-ml/jax)

**[Documentation](https://biomedai-ucsc.github.io/terraforming-docs/)** ·
**[GCM architecture](docs/package/gcm3d/architecture.md)** ·
**[Physics status and limitations](docs/ideas/gcm3d-physics-limitations.md)** ·
**[AmesGCM comparison](docs/ideas/amesgcm-comparison.md)**

</div>

---

## What this project is

Terraforming is an experimental framework for constructing differentiable,
three-dimensional planetary climate models. It provides the shared dynamics,
physical operators, state coupling, integration, diagnostics, and data interfaces
needed to assemble a GCM without baking one planet's constants or datasets into
the numerical core.

The framework builds on [Dinosaur](https://github.com/neuralgcm/dinosaur), the
spectral primitive-equation dynamical core used by NeuralGCM. Its intended use is
to combine conventional differentiable physics with learned residual tendencies,
gradient-based parameter calibration, and eventually carefully bounded planetary
intervention studies.

**Mars is the first and currently the only supported planet.** It serves as the
reference implementation used to develop and validate the generic interfaces. The
current Mars model supplies:

- spectral primitive-equation dynamics on terrain-following sigma levels;
- MOLA topography and spatial surface properties;
- diurnal and eccentric-orbit solar forcing;
- multiband CO₂ correlated-k radiation and radiatively active prescribed dust;
- prognostic surface temperature and multilayer regolith conduction;
- surface heat and momentum exchange, boundary-layer mixing, and dry convection;
- surface CO₂ condensation and sublimation with latent heat and atmospheric-mass
  exchange;
- restartable JAX integrations, NetCDF output, maps, diurnal diagnostics, and
  MCD comparison tooling.

Supporting another planet requires a new celestial package containing its body
constants, composition, optical properties, surface datasets, condensable cycles,
and model assembly. It should reuse the framework dynamics and applicable physical
operators rather than fork them.

The repository also retains a simpler Torch-based global-mean Mars model for fast
experiments and intervention prototypes. It is a separate numerical model, not a
replacement for or continuation of the 3-D JAX state.

## Current implementation: Mars

The generic framework is exercised through a present-day Mars GCM. This is research
software under active development, not yet a validated Mars climatology. The
deterministic physics components and their local conservation properties are
substantially implemented, but the coupled model still needs seasonal spin-up and
quantitative validation.

The validation order is:

1. Viking Lander 1 and 2 seasonal surface-pressure cycles;
2. global atmospheric-plus-cap CO₂ conservation;
3. MCS zonal-mean atmospheric temperature;
4. TES/THEMIS surface-temperature cycles;
5. MCD three-dimensional temperature and wind diagnostics;
6. matched AmesGCM runs for model-to-model debugging.

MCD and AmesGCM are model references, not observational truth. Short GCM runs are
labelled as spin-up transients and should not be interpreted as equilibrated
climatology. Water, water-ice clouds, CO₂-cloud microphysics, interactive dust
lifting, photochemistry, and thermospheric escape are not part of the current
three-dimensional baseline.

## Architecture

Reusable numerical machinery and physical operators are kept separate from each
planet's configuration and datasets:

```text
package/src/
├── framework/
│   ├── gcm/                    # Dinosaur adapter, coordinates, dynamics,
│   │                           # units, integration, restart and benchmarks
│   └── physics/                # Reusable differentiable column operators
│                               # and correlated-k radiation machinery;
│                               # currently bundles the Mars/Ames reference table
├── celestials/
│   └── planets/
│       └── mars/               # Mars constants, forcing, MOLA/TES/dust/MCD,
│                               # maps and seasonal model
├── engine/                     # Separate Torch global-mean integration path
└── interventions/              # Experimental intervention definitions
```

The dependency direction is deliberate:

```text
Planet implementation (currently Mars)
              ↓
Reusable framework physics
              ↓
Framework GCM and Dinosaur/JAX
```

Framework modules do not import Mars. Mars depends on the framework and composes
the generic pieces with Mars-specific physics and data. There is no top-level
`src.gcm3d` package.

## Extending the framework

A new planet implementation should provide:

- a `BodyConstants` instance for dynamics and nondimensionalisation;
- atmospheric thermodynamic and radiative properties;
- orbital and stellar forcing;
- surface, terrain, and aerosol datasets where available;
- planet-specific condensable or volatile cycles;
- an assembly module that selects reusable framework operators;
- validation observations and explicit acceptance criteria.

Generic solvers belong under `src.framework`; planet-specific coefficients,
parameterizations, and datasets belong under `src.celestials.planets.<planet>`.
The currently bundled Ames coefficient asset is colocated with the generic
correlated-k loader for reproducibility; moving optical tables behind an injected
planet-data interface remains an architectural cleanup. Framework code otherwise
does not import a concrete celestial implementation.

## Installation

The repository uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone https://github.com/BioMedAI-UCSC/terraforming.git
cd terraforming

# Install the workspace, development tools, and optional Dinosaur GCM stack.
uv sync --dev --all-extras
```

The base package can be installed without Dinosaur when only the Torch model is
needed. The three-dimensional model requires the `gcm3d` optional dependency:

```bash
pip install 'terraforming[gcm3d]'
```

External MOLA and surface datasets are staged separately. See
[the data-staging guide](docs/scripts/gcm3d-data.md).

## Run the Mars GCM

The command-line interface is `tform`:

```bash
# Default Mars GCM map run with deterministic physics.
uv run tform mars maps

# Select resolution, vertical layers, season and duration.
uv run tform mars maps \
  --truncation T21 \
  --layers 12 \
  --ls 270 \
  --dt 300 \
  --steps 1000

# Resolve the moving day/night terminator.
uv run tform mars maps --diurnal --dt 300

# Explicit dry-dycore diagnostic.
uv run tform mars maps --no-physics
```

Outputs are written under `outputs/gcm3d_maps/` and include plotted fields and a
NetCDF dataset suitable for analysis or benchmark comparison. Long integrations
use separate versioned restart checkpoints.

For restartable long-running physics ablations:

```bash
uv run python scripts/run_gcm3d_ablation.py \
  path/to/surface_properties.nc \
  --laptop \
  --resume
```

## Visualizer and benchmarks

The browser UI can run or stop GCM simulations, display the evolving field grid,
inspect diurnal outputs, load existing NetCDF runs, and compare matched fields
against MCD or uploaded reference data.

```bash
uv run tform serve
```

MCD downloads are cached beneath the selected output directory so matching data
can be reused. Maps and comparison figures can be exported as PNG with legends.

## Python API

The canonical imports make the generic/Mars boundary explicit:

```python
from src.celestials.planets.mars import MARS_BODY_3D
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import run_maps, save_netcdf

forcing = radiative_forcing(
    diurnal=True,
    co2_radiation_enabled=True,
)

fields = run_maps(
    body=MARS_BODY_3D,
    forcing=forcing,
    co2_forcing=co2_forcing(),
    truncation="T21",
    n_layers=12,
    dt_seconds=300.0,
    n_steps=1000,
)

save_netcdf(fields, "outputs/mars_gcm.nc")
```

Generic GCM components are available from `src.framework.gcm`; reusable column
operators and state containers are under `src.framework.physics`.

## Development

```bash
uv sync --dev --all-extras

# Core package tests, excluding long integrations.
uv run --project package python -m pytest package/tests -m "not slow"

# CLI and server tests.
uv run python -m pytest cli/tests -m "not slow"

# Documentation preview and static checking.
uv run mkdocs serve
uv run pyright
```

JAX compilation can be expensive on laptops. Prefer the supplied low-resolution
or `--laptop` configurations for smoke tests and use restart checkpoints for
seasonal integrations.

## Project scope

The long-term goal is a reusable framework for differentiable climate modelling
across planets and moons. The near-term milestone is narrower: demonstrate that
framework with a validated dry-Mars GCM and a defensible experiment such as
gradient-based calibration of soil, dust, or boundary-layer parameters against
observations.

The near-term goal is not a terraforming bifurcation result. Intervention forcing
becomes scientifically meaningful only after the supported planet model closes its
coupled energy and mass budgets and passes a documented validation scorecard.

## License

The license is currently undetermined. Until a license is added, all rights are
reserved by the authors ([BioMedAI-UCSC](https://github.com/BioMedAI-UCSC)).
