# Framework API

The primary GCM uses `src.framework.gcm` for body constants, grids, dynamics,
restart and parameterized integration; `src.framework.physics` for column
operators; and `src.framework.neural` for learned components. Begin with the
[GCM API](../package/gcm3d/api.md) and [architecture](../architecture/planet.md).

## Deprecated global-mean framework

!!! warning "Deprecated torch model"
    The `Planet` base class and subsystem dataclasses below support the deprecated
    global-mean workflow. They remain available for existing users and are not the
    state/integration contract of the GCM.

## Planet (Abstract Base Class)

::: src.framework.planet

## State Dataclasses

### Atmosphere

::: src.framework.atmosphere

### Thermal

::: src.framework.thermal

### Water / Cryosphere

::: src.framework.water

### Orbital Parameters

::: src.framework.orbital

### Intrinsic Parameters

::: src.framework.intrinsic

### Radiation

::: src.framework.radiation

### Magnetic

::: src.framework.magnetic
