# gcm3d

> A planet-agnostic, differentiable 3-D general-circulation core built on the NeuralGCM **dinosaur** dycore, plus the Mars column physics (radiation, surface energy, regolith, PBL, CO₂ condensation) that turns it into a mid-fidelity Mars climate model.

The GCM is the primary model for new work. It resolves dynamics and column
physics on a spherical-harmonic 3-D grid over MOLA topography and supports JAX
differentiation and batching. The torch global-mean model is deprecated;
see [migration and compatibility](../../deprecated-global-mean.md).

## Current configuration and evidence

`gcm3d` is the optional extra; reusable code lives in `src.framework.gcm`,
`src.framework.physics`, and `src.framework.neural`. The removed `src.gcm3d`
import path is unsupported.

| Entry point | Default behavior |
| --- | --- |
| `run_maps()` without forcing | Dry dynamics, T42/L25, 600 s, 200 steps |
| Mars `radiative_forcing()` | Diurnal grey radiation; advanced surface flags and resolved CO₂ radiation disabled |
| `tform mars maps` | Fast preset, daily-mean radiation and energy-limited CO₂ exchange |
| Browser GCM through `tform serve` | Correlated-k radiation, regolith, stability exchange, PBL, dry convection, conservative CO₂ exchange, diffusion, and available TES/Ames data |

The server baseline uses CO₂/dust longwave multipliers 0.25 and surface exchange
multiplier 1.0. Generic forcing multipliers default to 1.0. Matching output across
entry points requires matching data, initial state, duration and all forcing settings.

Local verification at revision `198433d` passed 411 tests with three skips and
built the UI. This is software verification, not observational validation or
proof of climate equilibrium. See [validation and limitations](validation.md).

## Contents

- [Quickstart](quickstart.md) — installation, first map, units and restart.
- [Calibration and ablations](calibration.md) — physical controls and protocols.
- [Temperature-only postprocessing](temperature-only.md) — frozen forecasts and held-out gates.
- [Validation and limitations](validation.md) — interpretation of test and climate results.
- [Philosophy](philosophy.md) — why a second (JAX) substrate exists and what it is *not*.
- [Architecture](architecture.md) — module map, data flow, the dynamics⊕physics seam.
- [API Reference](api.md) — every public function/class and its signature.
- [Implementation Details](implementation.md) — the algorithms and physics, derived and explained.
- [Neural experiments](neural-experiments.md) — trainable components, parameter recovery, and neural radiation examples.

## Related design notes (`docs/ideas/`)
- [`dinosaur-backbone-prototype.md`](../../ideas/dinosaur-backbone-prototype.md) — the 0-D ODE-on-dinosaur proof of concept.
- [`dinosaur-mars-maps.md`](../../ideas/dinosaur-mars-maps.md) — the dry-dynamics maps and physics-coupling walkthrough.
- [`gcm3d-physics-limitations.md`](../../ideas/gcm3d-physics-limitations.md) — the honest ledger: what is verified / approximated / absent.
- [`amesgcm-comparison.md`](../../ideas/amesgcm-comparison.md) — mapping to the NASA Ames MGCM physics chain.

## Install

`gcm3d`'s dycore needs JAX + `dinosaur`, which the torch core does **not**. They
are an **optional extra**:

```bash
pip install 'terraforming[gcm3d]'
```

The pure-Python body abstraction (`from src.framework.gcm import BodyConstants`) is always
importable; every dycore builder raises a friendly "install the gcm3d extra"
message if the extra is absent (single guard in `src/framework/gcm/_dinosaur.py`).

## Quick start

```python
# Dry-dynamics Mars maps over real MOLA terrain
from src.celestials.planets.mars.maps import run_maps, save_maps
fields = run_maps(truncation="T42", n_layers=25, dt_seconds=300.0, n_steps=1000)
save_maps(fields, "outputs/mars_maps")          # NetCDF + one PNG per field

# Add the column radiative energy balance + seasonal CO₂ cap
from src.celestials.planets.mars.gcm import radiative_forcing, co2_forcing
forcing    = radiative_forcing(co2_radiation_enabled=True, diurnal=True)
co2        = co2_forcing()
fields = run_maps(truncation="T21", n_layers=12, dt_seconds=300.0, n_steps=100,
                  forcing=forcing, co2_forcing=co2)
```

The separate `src.celestials.planets.mars.seasonal` module provides a 0-D
seasonal ODE on the same JAX substrate. It requires its own `SeasonalForcing`
with explicit planetary, cap, location and orbital parameters; a 3-D
`RadiativeForcing` cannot be passed to `run_seasonal`.
