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
| MARS-0002 | Atmospheric escape rate (species-wise) | atmosphere/space | TBD | kg/s | TBD | TBD | TBD | YYYY-MM-DD | MAVEN (NGIMS/STATIC/SWIA/SEP) | in-situ ion/neutral measurements + modeling | https://lasp.colorado.edu/maven/sdc/public/ | global (upper atmosphere) | multi-year | Atmos mass budget | High: sets retention and shielding needs | Reduce uncertainty to <=30% per species | reprocess MAVEN data + cross-mission validation | TBD | TBD | Medium | 2 | open | prioritize CO2, O, N loss |
| MARS-0003 | Solar EUV/UV flux at Mars orbit | radiation/space | TBD | W/m^2 | TBD | TBD | TBD | YYYY-MM-DD | MAVEN (EUV) | spectrometer + irradiance model | https://lasp.colorado.edu/maven/sdc/public/ | global | seasonal + solar cycle | Climate warming, photochemistry | Medium: drives upper-atmos heating and ozone | Reduce uncertainty to <=20% monthly mean | MAVEN EUV data assimilation | TBD | TBD | High | 3 | open | align to reference epoch |
| MARS-0004 | Solar wind dynamic pressure at Mars | space weather | TBD | nPa | TBD | TBD | TBD | YYYY-MM-DD | MAVEN (SWIA/MAG) | plasma + magnetometer | https://lasp.colorado.edu/maven/sdc/public/ | upstream | multi-year | Magnetic shield program, escape | High: drives shielding requirement | Reduce uncertainty to <=20% | reprocess MAVEN + heliospheric models | TBD | TBD | Medium | 4 | open | tie to shielding uptime |
| MARS-0005 | Subsurface ice distribution | cryosphere | TBD | kg or % volume | TBD | TBD | TBD | YYYY-MM-DD | MRO/SHARAD, MARSIS, Odyssey GRS | radar sounding + neutron spectroscopy | TBD (PDS Geosciences) | regional/global | multi-year | Hydrology and melting | High: defines water inventory | Reduce uncertainty to <=25% | radar inversion + gravimetric constraints | TBD | TBD | Medium | 5 | open | map depth and purity |
| MARS-0006 | Regolith thermal inertia map | geology/regolith | TBD | J m^-2 K^-1 s^-1/2 | TBD | TBD | TBD | YYYY-MM-DD | TES/THEMIS | thermal IR retrievals | TBD (PDS Geosciences) | global | seasonal | Climate warming, hydrology | Medium: affects heat storage and melt | Reduce uncertainty to <=15% | reprocess IR datasets | TBD | TBD | High | 6 | open | co-register with albedo |
| MARS-0007 | Regolith albedo and mineralogy | geology/regolith | TBD | unitless / wt% | TBD | TBD | TBD | YYYY-MM-DD | TES/CRISM | spectral inversion | TBD (PDS Geosciences) | global/regional | multi-year | Climate warming, soil readiness | Medium: affects forcing + soil chemistry | Reduce uncertainty to <=20% | improved spectral mixing models | TBD | TBD | Medium | 7 | open | separate dust vs rock |

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

