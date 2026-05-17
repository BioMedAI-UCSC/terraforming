# Engine API

The `src.engine` module handles numerical integration of planetary state. The integration strategy is fully decoupled from the physics model — `Planet` knows *what* equations to use, `TimeController` knows *how* to integrate them.

## Accuracy Modes

```python
from src.engine import Accuracy

Accuracy.ACCURATE  # 4th-order Runge-Kutta on full ODE system
Accuracy.FAST      # Reduced-order analytic / relaxation updates
```

## Snapshot

Each timestep produces a `Snapshot` record:

| Field | Type | Description |
|-------|------|-------------|
| `time` | float | Elapsed time in sols |
| `surface_temperature` | float | Surface temperature (K) |
| `surface_pressure` | float | Surface pressure (Pa) |
| `ice_mass` | float | Total polar ice mass (kg) |
| `solar_flux` | float | Instantaneous solar flux (W/m²) |
| `orbital_angle` | float | True anomaly (rad) |
| `greenhouse_factor` | float | Current greenhouse amplification |

## TimeController & Snapshot

::: src.engine
    options:
      members:
        - Accuracy
        - Snapshot
        - TimeController
        - BatchedMars
        - BatchedTimeController
      show_root_heading: true
      show_root_full_path: false
