# Presets

Built-in presets are YAML configs bundled with the CLI. They cover representative Mars scenarios from baseline current state to active terraforming.

## Listing Presets

```bash
tform mars config list
tform mars config show <name>
```

## Available Presets

### Baseline Scenarios

| Preset | Description |
|--------|-------------|
| `current-mars` | Present-day Mars: $T \approx 210\,\text{K}$, $P \approx 636\,\text{Pa}$ |
| `gale-crater` | Gale Crater conditions (Curiosity rover site, $-4.5°\text{S}$, $137.4°\text{E}$) |
| `early-mars` | Hypothetical early Mars with denser atmosphere |
| `terraforming-phase1` | Phase 1 target: warming via super-GHG injection |

### Multi-Coordinate Scenarios

| Preset | Description |
|--------|-------------|
| `equatorial` | Three-latitude sweep: 45°N, 0°, 40°S at 137°E |
| `polar` | Polar focus run |

### Landmark Sites

| Preset | Location | Elevation | Notes |
|--------|----------|-----------|-------|
| `olympus-mons` | 18.65°N, 226.2°E | +21 km | Tallest volcano in the solar system |
| `elysium-mons` | 25.02°N, 147.21°E | +14 km | Second tallest Martian volcano |
| `hellas-basin` | 42.4°S, 70.5°E | −7 km | Deepest impact basin; highest pressures |
| `south-polar-cap` | 90°S, 0°E | +2 km | CO₂ and water ice reservoir |
| `landmark-spots` | All four | — | Runs all landmark sites in one call |

## Landmark Site Conditions

Landmark sites have calibrated initial conditions that reflect real elevation-dependent temperature and pressure differences:

$$
P(h) = P_0 \exp\!\left(-\frac{h}{H}\right), \quad H \approx 10.8\,\text{km (Mars)}
$$

$$
T(h) = T_0 - \Gamma \cdot h, \quad \Gamma \approx 2.5\,\text{K/km (Mars)}
$$

Hellas Basin runs significantly warmer and denser than the planetary average; Olympus Mons is colder and at near-vacuum pressure.

## Writing a Custom Preset

Any YAML file following the `SimConfig` schema can be used as a preset via `--config`:

```yaml
planet:
  surface_temperature: 215.0    # K — initial surface temperature
  surface_pressure: 900.0       # Pa — Hellas-like dense atmosphere
  albedo: 0.22                  # dimensionless [0, 1]
  greenhouse_factor: 1.05       # multiplicative warming amplifier (≥ 1.0)
  ice_mass: 1.0e15              # kg — polar cap mass
  latitude: -42.4               # degrees [-90, 90]
  longitude: 70.5               # degrees [0, 360]
  elevation_m: -7000.0          # m — used to set P and T lapse corrections

experiment:
  type: year
  sols: 687
  accuracy: accurate
```

Run it with:

```bash
tform mars run --config my-hellas.yaml
```
