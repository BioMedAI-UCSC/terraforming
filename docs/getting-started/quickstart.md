# Quickstart

## CLI: Run a preset simulation

The fastest way to run a simulation is with a built-in preset:

```bash
# Single sol (diurnal cycle) at Gale Crater
tform mars run --preset gale-crater --type sol

# One Martian year at current Mars baseline
tform mars run --preset current-mars --type year

# Multi-latitude run (45°N, equator, 40°S)
tform mars run --preset equatorial --type multi

# 4 landmark sites in one run
tform mars run --preset landmark-spots --type spots

# Terraforming intervention: GHG injection over many years
tform mars run --preset terraforming-phase1 --type intervention
```

Results are saved as CSV in `outputs/` and plots are shown automatically. Pass `--no-plot` to suppress plots.

## CLI: Custom YAML config

Create a config file:

```yaml
# my-sim.yaml
planet:
  surface_temperature: 210.0   # K
  surface_pressure: 636.0      # Pa
  albedo: 0.25
  greenhouse_factor: 1.0
  ice_mass: 3.0e15             # kg
  latitude: 0.0
  longitude: 137.0

experiment:
  type: year
  sols: 687
  accuracy: accurate
```

Run it:

```bash
tform mars run --config my-sim.yaml
```

Validate without running:

```bash
tform mars config validate my-sim.yaml
```

## Python API: Basic simulation

```python
from src.celestials import Mars
from src.engine import TimeController, Accuracy, Snapshot

# Create Mars with default (current) state
planet = Mars()

# Integrate for 1 Martian year (~687 Earth days)
tc = TimeController(planet, accuracy=Accuracy.ACCURATE)
snapshots: list[Snapshot] = tc.run(n_sols=687)

# Access results
for snap in snapshots[-10:]:
    print(f"Day {snap.time:.1f}: T={snap.surface_temperature:.1f} K, "
          f"P={snap.surface_pressure:.1f} Pa")
```

## Python API: GHG intervention

```python
from src.celestials import Mars
from src.interventions import InterventionController

planet = Mars()
ctrl = InterventionController(planet)

# Inject 1e9 kg/year of SF6 for 50 years
results = ctrl.run(
    schedule={"SF6": 1e9},  # kg/year
    n_years=50,
)

print(f"Final temperature: {results[-1].surface_temperature:.1f} K")
print(f"Temperature gain:  {results[-1].surface_temperature - 210:.1f} K")
```

## Explore available presets

```bash
tform mars config list
tform mars config show terraforming-phase1
```
