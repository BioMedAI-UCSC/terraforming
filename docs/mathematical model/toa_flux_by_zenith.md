# Mathematical Model: TOA Flux by Solar Zenith Angle

This model computes top-of-atmosphere (TOA) incident solar flux on Mars as a function of solar zenith angle.

## Equation

`F_TOA(theta_z) = F_normal * max(0, cos(theta_z))`

Where:
- `F_TOA(theta_z)` is TOA incident solar flux (`W/m2`)
- `F_normal` is normal-incidence TOA flux at current Mars-Sun distance (`W/m2`)
- `theta_z` is solar zenith angle (`deg`)

Equivalent expanded form:

`F_TOA(theta_z) = (S_1AU / r_AU^2) * max(0, cos(theta_z))`

Where:
- `S_1AU` is the solar constant at 1 AU (`W/m2`)
- `r_AU` is Sun-Mars distance (`AU`)

## Baseline numerical example

Using the current baseline context (`Ls ~ 0 deg`):
- `F_normal = 713.246626897437 W/m2`

Computed values:
- `theta_z = 0 deg` -> `F_TOA = 713.2466 W/m2`
- `theta_z = 30 deg` -> `F_TOA = 617.6897 W/m2`
- `theta_z = 45 deg` -> `F_TOA = 504.3415 W/m2`
- `theta_z = 60 deg` -> `F_TOA = 356.6233 W/m2`
- `theta_z = 90 deg` -> `F_TOA ~ 0 W/m2`

## Notes
- `90 deg` gives a tiny nonzero floating-point epsilon in code, but physically it is treated as zero direct shortwave.
- This is direct-beam geometry only; diffuse and terrain effects are not included.
