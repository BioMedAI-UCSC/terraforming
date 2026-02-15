# 2) Resources and capability

## Input
- Mission constraints: budget, launch cadence, compute, staffing.

## Output
- Capacity envelope `C_cap(t)` for simulation and deployment throughput.

## Modeling approach (equations + parameters)
- Throughput constraint model:
  - `u_k(t) <= min(C_power_k(t), C_mass_k(t), C_labor_k(t), C_compute_k(t))`
- Parameters: capacities per subsystem `C_*`, task demand coefficients `d_{k,*}`.

## Data needed
- Cost curves, launch vehicle specs, workforce assumptions, hardware availability.

## Assumptions
- Capacity is piecewise-constant over planning windows.
- No hard geopolitical/supply-chain disruptions in baseline scenario.

## Pre-requisites
- Phase 1 baseline for realistic demand estimates.


