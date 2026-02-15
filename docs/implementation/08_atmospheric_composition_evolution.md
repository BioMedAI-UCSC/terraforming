# 8) Atmospheric composition evolution

## Input
- Mass-budget outputs, oxygen production plans, chemical conversion yields.

## Output
- Time series of composition fractions `x_i(t)` and partial pressures `p_i(t)`.

## Modeling approach (equations + parameters)
- Mole-fraction dynamics:
  - `x_i = M_i / sum_j M_j`
  - `dx_i/dt = (1/M_tot) * (dM_i/dt - x_i * dM_tot/dt)`
- Partial pressure:
  - `p_i = x_i * P`
- Parameters: conversion efficiencies (for O2 pathways), leakage factors, control schedules.

## Data needed
- ISRU process specs, electrolysis/sabatier yields, storage/transport losses.

## Assumptions
- Atmosphere remains well mixed on modeled timescales.
- Controlled injection schedules are executable as planned.

## Pre-requisites
- Phases 3, 5, 6, 7.


