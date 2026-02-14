# Mars Terraforming Objectives and Evaluation Spec

## Purpose
Define how candidate terraforming plans are evaluated and compared in a consistent, reproducible way.

## Scope
- Horizon: centuries (default: 0-500 years; configurable)
- Plans evaluated: target 5-10
- Time representation: semi-continuous trajectories with allowed discrete events
- Primary body: Mars (framework should remain portable to other bodies)

## Required Outputs Per Plan
- Time-series state trajectory (atmosphere, climate, hydrology, etc.)
- Event log for discrete interventions (what, when, trigger/condition)
- Per-step and cumulative:
  - energy use (J, Wh, or TWh; choose one standard)
  - resource use (mass, volume, throughput by category)
  - monetary cost (constant-year currency + discount assumptions)
  - planetary disruption indicators
- Constraint violations (if any) with timestamps

## Objective Families
Each plan must be scored along these dimensions. Weights can be scenario-specific.

1. Energy efficiency
   - Minimize cumulative energy and peak power demand

2. Resource efficiency
   - Minimize mining effort on Mars, on Earth or on other asteroids
   - Minimize imported mass and scarce material usage
   - Minimize supply chain fragility (optional sub-metric)

3. Economic efficiency
   - Minimize total lifecycle cost and annualized burden
   - Include uncertainty range for high-variance cost terms

4. Planetary disruption minimization
   - Minimize irreversible ecological/geological disturbance
   - Penalize high-risk interventions (e.g., major impact events)

5. Mission robustness
   - Maximize resilience to parameter uncertainty and delays
   - Minimize failure sensitivity to single-point assumptions

## Hard Constraints (Go/No-Go)
Plans that fail hard constraints are rejected or labeled non-viable.
- Physical consistency checks (e.g. )
- Safety thresholds exceeded (needs pre-defined values)
- Required infrastructure dependencies not satisfied

## Comparison Modes
Use both modes to avoid bias from any single ranking style.

1. Pareto comparison
   - Build an objective vector for each viable plan:
     - `[energy, resources, cost, disruption, robustness_penalty]`
     - all dimensions converted to "lower is better"
   - Normalize each objective across the current candidate set so dimensions are comparable
   - Apply hard-constraint filtering first; infeasible plans are excluded from Pareto analysis
   - Dominance rule:
     - Plan A dominates Plan B if A is no worse in every objective and strictly better in at least one
   - Non-dominated plans form the Pareto front (rank 1)
   - Optional deeper ranking:
     - remove rank-1 plans, recompute to get rank-2, rank-3 fronts
   - Tie-break policy inside the same front:
     - use crowding distance/diversity so selected plans are spread across tradeoff space
   - Required output artifact:
     - `pareto_report` containing dominated pairs, front membership, and tradeoff notes per frontier plan
   - Decision usage:
     - do not force a single "best" here; Pareto narrows to efficient options, then policy weights choose final plan

2. Weighted composite score
   - Normalize each metric to [0,1]
   - Scenario-specific weights produce a single ranking
   - Publish full weight vector used for each ranking

## Uncertainty and Sensitivity Policy
- Evaluate each plan under baseline/optimistic/conservative assumptions
- Run uncertainty propagation (method defined in implementation)
- Report:
  - median outcome
  - uncertainty interval (e.g., 5-95%)
  - top sensitivity drivers

## Required Artifact Format (Per Plan)
Use a consistent plan ID and run manifest for reproducibility.

- `plan_id`
- `assumption_set_id`
- `model_version`
- `config_snapshot`
- `seed` (if stochastic)
- `trajectory_outputs`
- `event_log`
- `objective_metrics`
- `constraint_report`
- `sensitivity_report`

## Minimum Deliverables for Milestone M1
- 5-10 candidate plans generated
- Pareto set identified
- At least 2 weighted ranking views (different policy priorities)
- Clear recommendation:
  - best overall plan(s)
  - best low-risk plan
  - best low-cost plan

## Open Decisions To Finalize
- Exact metric definitions and units
- Discount rate and currency year
- Disruption index formula
- Hard constraint thresholds
- Default uncertainty method and sample budget

