# 4) Magnetic shield program

## Input
- Solar wind conditions, shield architecture (equatorial superconducting loop), material limits.

## Output
- Shield design point + effectiveness `f_shield(t)` reducing atmospheric escape.

## Modeling approach (equations + parameters)
- Pressure-balance requirement:
  - `P_ram = rho_sw * v_sw^2`
  - `P_mag = B_mp^2 / (2 * mu_0)`
  - Shield condition: `P_mag >= P_ram`
- Dipole field scaling:
  - `B(r) = B_0 * (R_M / r)^3`
- Superconductor geometric constraint (compact form):
  - `B_0 / B_c ~= (pi * d * a^2) / (8 * R_M^3)`
  - where `a` = loop radius, `d` = bundle radius, `B_c` = critical field.
- Cryogenic power:
  - `Q_dot = (k * A / L) * DeltaT`
- Escape reduction coupling:
  - `E_i,eff = E_i,base * (1 - f_shield)`
- Key parameters: `rho_sw`, `v_sw`, `mu_0`, `B_c`, `T_c`, `k`, `A`, `L`, `DeltaT`, uptime `U`.

## Data needed
- Solar wind statistics at Mars orbit, superconducting material properties, thermal insulation performance, failure rates.

## Assumptions
- Equatorial loop geometry is baseline architecture.
- `f_shield` represented as scenario parameter or function of shield uptime/performance.
- Non-superconducting fallback is modeled as high-power penalty scenario.

## Pre-requisites
- Phases 1-3.


