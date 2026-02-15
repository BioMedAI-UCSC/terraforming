# 5) Atmospheric mass budget

## Input
- Volatile sources, import plans, shield effectiveness, regolith exchange coefficients.

## Output
- `M_i(t)`, `P(t)`, net accumulation rates per species.

## Modeling approach (equations + parameters)
- Species-wise mass balance:
  - `dM_i/dt = S_i(t) + I_i(t) - E_i,eff(t) - R_i(M_i,T,...) - C_i(...)`
- Pressure mapping:
  - `P(t) = g_M * sum_i M_i / A_M`
- Parameters: source `S_i`, import `I_i`, escape `E_i`, regolith sink/source `R_i`, chemical conversion `C_i`.

## Data needed
- MAVEN-derived loss rates, volatile inventories, outgassing curves, import logistics, regolith sorption data.

## Assumptions
- Global box model (well-mixed atmosphere) for first implementation.
- Species coupling through `C_i` is reduced-order.

## Pre-requisites
- Phases 1, 3, 4.


