# Mars API

The `src.celestials` package provides the concrete `Mars` class — a full implementation of the abstract `Planet` interface with NASA-calibrated constants and Mars-specific ODE physics.

## Mars Constants

All constants are derived from the NASA Mars Fact Sheet and stored as `torch.float64` tensors.

| Constant | Value | Unit |
|----------|-------|------|
| `MARS_MASS` | $6.4171 \times 10^{23}$ | kg |
| `MARS_RADIUS` | $3.3895 \times 10^6$ | m |
| `MARS_GRAVITY` | $3.721$ | m/s² |
| `MARS_ROTATION_PERIOD` | $88642.66$ | s |
| `MARS_ORBITAL_PERIOD` | $59355072$ | s (~687 Earth days) |
| `MARS_AXIAL_TILT` | $25.19°$ | deg |
| `MARS_SEMI_MAJOR_AXIS` | $1.524$ | AU |
| `MARS_ECCENTRICITY` | $0.0934$ | — |

## Mars Class

::: src.celestials
    options:
      members:
        - Mars
      show_root_heading: true
      show_root_full_path: false
