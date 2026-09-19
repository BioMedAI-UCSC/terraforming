# gcm3d

> A planet-agnostic, differentiable 3-D general-circulation core built on the NeuralGCM **dinosaur** dycore, plus the Mars column physics (radiation, surface energy, regolith, PBL, CO₂ condensation) that turns it into a mid-fidelity Mars climate model.

The GCM is the **mid-fidelity tier** of the model ladder. The 0-D torch model
(`src.celestials.planets.mars.Mars`) gives a global-mean seasonal cycle; the GCM
resolves the same physics on a spherical-harmonic 3-D grid over real MOLA
topography, and is differentiable and batchable end-to-end via JAX.

## Contents
- [Philosophy](philosophy.md) — why a second (JAX) substrate exists and what it is *not*.
- [Architecture](architecture.md) — module map, data flow, the dynamics⊕physics seam.
- [API Reference](api.md) — every public function/class and its signature.
- [Implementation Details](implementation.md) — the algorithms and physics, derived and explained.

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
fields     = run_maps(forcing=forcing, co2_forcing=co2, **resolve_scale("balanced"))
```

```python
# 0-D seasonal (Ls-indexed) cycle on the same substrate — differentiable
from src.celestials.planets.mars.seasonal import (
    SeasonalForcing, initial_seasonal_state, run_seasonal,
)
traj = run_seasonal(forcing, initial_seasonal_state(210.0, 610.0, 0.0, 4e15),
                    dt_seconds=2000.0, n_steps=30000, sample_every=50)
traj.write_csv("outputs/mars_seasonal.csv")     # sol, Ls, T, P, ice, flux
```
