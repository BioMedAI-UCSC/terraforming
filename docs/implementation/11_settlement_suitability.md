# 11) Settlement suitability

## Input
- Outputs from all prior phases + risk and logistics constraints.

## Output
- Ranked candidate settlement zones and robustness scores.

## Modeling approach (equations + parameters)
- Multi-criteria score:
  - `S_site = sum_k w_k * z_k`
- Robustness under uncertainty:
  - `R_site = Pr(S_site >= S_min)` from Monte Carlo over uncertain inputs.
- Parameters: criterion weights `w_k`, normalized metrics `z_k`, threshold `S_min`.

## Data needed
- Terrain hazards, resource proximity, climate reliability, infrastructure distance/cost.

## Assumptions
- Weighting scheme is stakeholder-defined and scenario-dependent.
- Correlations across risk factors can be sampled with simplified copulas.

## Pre-requisites
- Phases 1-10.


