# Framework API

The `src.framework` module defines the abstract `Planet` base class and all state dataclasses. Concrete planets (e.g. `Mars`) inherit from these and implement planet-specific physics.

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
