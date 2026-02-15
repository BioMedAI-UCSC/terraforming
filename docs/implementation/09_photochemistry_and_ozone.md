# 9) Photochemistry and ozone

## Input
- Composition trajectories, UV flux, atmospheric temperature profile assumptions.

## Output
- Ozone column depth trajectory and UV-at-surface attenuation estimate.

## Modeling approach (equations + parameters)
- Reduced Chapman-like chemistry:
  - `d[O]/dt = J1[O2] - k1[O][O2][M] + J3[O3] - k3[O][O3]`
  - `d[O3]/dt = k1[O][O2][M] - J3[O3] - k3[O][O3]`
- Parameters: photolysis rates `J1, J3`, reaction rates `k1, k3`, third-body concentration `[M]`.

## Data needed
- Solar UV spectrum at Mars, reaction cross-sections and rates, dust/aerosol attenuation priors.

## Assumptions
- Start with 1D column chemistry and effective mixing.
- Heterogeneous chemistry on dust is parameterized.

## Pre-requisites
- Phases 6, 8.


