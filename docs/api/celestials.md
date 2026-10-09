# Mars API

The primary spatial model uses `src.celestials.planets.mars.gcm` for forcing,
`maps` for integration/exports, and `topography`, `surface` and `dust` for boundary
data. `MARS_BODY_3D` configures the generic dry core. See the
[GCM API](../package/gcm3d/api.md) and [Mars configuration](../architecture/mars.md).

!!! warning "Deprecated global-mean class"
    The torch `Mars` class below is deprecated. Use the GCM for new work;
    constants and existing APIs remain available for compatibility.

## Mars Constants

All constants are derived from the NASA Mars Fact Sheet and stored as `torch.float64` tensors.

| Constant | Value | Unit |
|----------|-------|------|
| `MARS_MASS` | $6.4171 \times 10^{23}$ | kg |
| `MARS_RADIUS` | $3.3895 \times 10^6$ | m |
| `MARS_GRAVITY` | $3.721$ | m/s² |
| `MARS_ROTATION_PERIOD` | $88775.244$ | s (model sol) |
| `MARS_ORBITAL_PERIOD` | $59356800$ | s (~668.62 model sols) |
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
