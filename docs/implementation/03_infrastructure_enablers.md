# 3) Infrastructure enablers

## Input
- Capacity envelope + candidate infrastructure designs.

## Output
- Time series for available power, mined mass, and processed materials.

## Modeling approach (equations + parameters)
- Stock-flow ODEs:
  - `dE/dt = G_E(t) - L_E(t) - D_E(t)` (energy stock)
  - `dQ_ore/dt = m_dot_mine(t) - m_dot_proc(t)` (ore stock)
  - `dQ_mat/dt = eta_proc * m_dot_proc(t) - m_dot_use(t)` (usable materials)
- Parameters: generation `G_E`, losses `L_E`, degradation `D_E`, processing efficiency `eta_proc`.

## Data needed
- Solar/nuclear generation profiles, ore grade maps, process efficiencies, maintenance intervals.

## Assumptions
- First-order degradation approximates equipment aging.
- Transport delays can be represented as fixed lag or buffer stock.

## Pre-requisites
- Phases 1-2.


