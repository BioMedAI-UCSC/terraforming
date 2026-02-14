# Mars Measurement Register

## Purpose
Track all Mars measurements used by the simulation, including provenance, uncertainty, and what is required to improve certainty.

## How To Use
- One row per measured variable (or measurement campaign, if preferred)
- Keep entries versioned as new missions/data products arrive
- Link each variable to models/plans it materially influences
- Maintain a priority ranking to guide future mission instrumentation

## Measurement Register (Template)
Use this table as the canonical register.

| measurement_id | variable_name | domain | current_estimate | units | uncertainty_value | uncertainty_type | confidence_level | measured_when | measured_by | measurement_method | source_reference | spatial_coverage | temporal_coverage | model_dependency | impact_on_decisions | certainty_improvement_required | proposed_improvement_method | estimated_improvement_cost | estimated_improvement_timeline | feasibility | priority_rank | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MARS-0001 | CO2 in polar caps | cryosphere/atmosphere | TBD | kg | TBD | interval | TBD | YYYY-MM-DD | mission/instrument | radar + gravimetry (example) | DOI/URL | polar region | seasonal snapshot | Atmos mass budget, climate warming | High: changes required imports and warming path | Reduce interval width by >=50% | new orbital radar campaign + in-situ validation | TBD | TBD | Medium | 1 | open | example seed row |

## Field Definitions
- `measurement_id`: stable identifier, e.g. `MARS-0001`
- `variable_name`: measured quantity
- `domain`: atmosphere, cryosphere, geology, magnetic, radiation, etc.
- `current_estimate`: best current value used by model
- `uncertainty_value`: numeric width/variance/error term
- `uncertainty_type`: interval, standard deviation, confidence interval, posterior, etc.
- `confidence_level`: e.g. 68%, 95%, or mission-defined quality class
- `measured_when`: date or date range of acquisition
- `measured_by`: mission and/or instrument
- `measurement_method`: retrieval/inference method
- `source_reference`: paper/data catalog/mission archive
- `spatial_coverage`: local/regional/global footprint
- `temporal_coverage`: instant/seasonal/multi-year
- `model_dependency`: which model modules consume this variable
- `impact_on_decisions`: qualitative or quantitative influence on plan ranking
- `certainty_improvement_required`: target precision needed for decision confidence
- `proposed_improvement_method`: mission concept or analysis upgrade
- `estimated_improvement_cost`: rough order-of-magnitude cost
- `estimated_improvement_timeline`: expected time to improved estimate
- `feasibility`: low/medium/high (technical + programmatic)
- `priority_rank`: 1 is highest priority
- `status`: open, in-progress, improved, superseded

## Ranking Policy (Suggested)
Rank by expected decision value, not curiosity alone.

- Priority score components:
  - decision impact (weight high)
  - current uncertainty magnitude
  - tractability/cost to improve
  - time-to-impact on plan selection
- Recompute rank each major model release

## Review Cadence
- Update after every relevant mission data release
- Formal review once per quarter
- Freeze register snapshot for each published plan comparison run

