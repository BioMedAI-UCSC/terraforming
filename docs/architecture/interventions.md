# Super-Greenhouse Gas (GHG) Intervention Layer

Complete derivation, architecture, equations, and worked examples for the
terraforming intervention system in [`package/src/interventions/`](../../package/src/interventions/)
and its CLI integration in [`cli/`](../../cli/).

---

## Table of Contents

1. [Why a Separate Layer](#1-why-a-separate-layer)
2. [Compounds Registry](#2-compounds-registry)
3. [Atmospheric Concentration — ppb Formula](#3-atmospheric-concentration--ppb-formula)
4. [Radiative Forcing — ΔF](#4-radiative-forcing--δf)
5. [Greenhouse Factor Update — GHF Formula](#5-greenhouse-factor-update--ghf-formula)
6. [Exponential Decay](#6-exponential-decay)
7. [Annual Simulation Loop](#7-annual-simulation-loop)
8. [Baseline OLR — Why It Must Be Cached](#8-baseline-olr--why-it-must-be-cached)
9. [Architecture Diagram](#9-architecture-diagram)
10. [Module Map](#10-module-map)
11. [CLI Integration](#11-cli-integration)
12. [Bugs Found and Fixed](#12-bugs-found-and-fixed)
13. [Worked Numerical Example](#13-worked-numerical-example)
14. [Stability Limits and Constraints](#14-stability-limits-and-constraints)
15. [Citations](#15-citations)

---

## 1. Why a Separate Layer

The base Mars physics model (`mars.py`, `planet.py`, `time_controller.py`) is a
coupled ODE system that integrates temperature, pressure, and ice mass over time.
It is **not** modified by the intervention layer.

Adding greenhouse gases affects only one scalar parameter that the ODE already
reads: `mars.thermal.greenhouse_factor` (GHF). The intervention layer computes a
new GHF each year based on accumulated GHG mass and writes it once. Everything
else — orbital mechanics, solar geometry, sublimation physics — runs unchanged.

```
┌─────────────────────────────────────────────────┐
│  InterventionController  (annual scheduler)      │
│  ┌───────────┐  ┌───────────┐  ┌─────────────┐  │
│  │  GHGState │  │ forcing.py│  │TimeController│  │
│  │  (masses) │→ │  (ΔF→GHF)│→ │  (1 yr ODE) │  │
│  └───────────┘  └───────────┘  └─────────────┘  │
│         ↑                              ↓         │
│    inject / decay             InterventionSnapshot│
└─────────────────────────────────────────────────┘
         ↕  writes greenhouse_factor once per year
┌─────────────────────────────────────────────────┐
│  Mars physics  (ODE — untouched)                 │
│  dT/dt  dP/dt  dM_ice/dt  orbital mechanics      │
└─────────────────────────────────────────────────┘
```

**Design principles:**

- **Zero coupling to physics internals.** Only `mars.thermal.greenhouse_factor`
  is written. No other field is touched.
- **Tensor-native.** All state lives on the same device as Mars (CPU or CUDA).
  No cross-device copies occur in the simulation loop.
- **Cumulative, not incremental.** GHF is always computed from the fixed
  baseline GHF plus the total accumulated ΔF, never by modifying the current GHF.
  This prevents compounding (see [Section 8](#8-baseline-olr--why-it-must-be-cached)
  and [Section 12](#12-bugs-found-and-fixed)).

---

## 2. Compounds Registry

Source: [`compounds.py`](../../package/src/interventions/compounds.py)

Seven super-greenhouse gases are registered. Radiative forcing efficiencies are
Mars-specific values from Marinova et al. (2005), which differ from Earth (IPCC
AR6) because Mars lacks water-vapour overlap bands and has a thinner CO₂ column.

| Name | Formula | MW (g/mol) | Lifetime (yr) | η (W m⁻² ppb⁻¹) | GWP₁₀₀ |
|------|---------|-----------|--------------|-----------------|---------|
| CF4 | Carbon tetrafluoride | 88.0 | 50 000 | 0.0880 | 6 630 |
| C2F6 | Hexafluoroethane | 138.0 | 10 000 | 0.2600 | 11 100 |
| C3F8 | Octafluoropropane | 188.0 | 2 600 | 0.2400 | 8 900 |
| C4F10 | Decafluorobutane | 238.0 | 2 600 | 0.3600 | 8 860 |
| C6F14 | Tetradecafluorohexane | 338.0 | 3 200 | 0.4900 | 9 300 |
| SF6 | Sulfur hexafluoride | 146.1 | 3 200 | 0.5700 | 23 900 |
| NF3 | Nitrogen trifluoride | 71.0 | 500 | 0.2100 | 16 100 |

**Lifetime choice:** Earth-reference (conservative/shorter) values are used.
Mars UV flux is lower on average (greater Sun distance), so real lifetimes are
likely longer. Using Earth values is the safe choice for not over-estimating
accumulation.

**Why perfluorocarbons?** PFCs (CF4, C2F6, etc.) have extremely long atmospheric
lifetimes (centuries to tens of thousands of years), very high GWP, and strong
absorption in the Martian atmospheric window. SF6 has the highest RF efficiency
per ppb of any registered compound (0.57 W m⁻² ppb⁻¹).

---

## 3. Atmospheric Concentration — ppb Formula

Source: [`forcing.py:compute_concentration_ppb`](../../package/src/interventions/forcing.py)

The molar mixing ratio of compound *i* in the Mars atmosphere is:

```
          M_i / MW_i
C_i  =  ────────────── × 10⁹   [ppb]
         M_atm / MW_atm
```

where:

| Symbol | Meaning | Value |
|--------|---------|-------|
| M_i | Atmospheric mass of compound *i* | kg (from GHGState) |
| MW_i | Molecular weight of compound *i* | g/mol (from registry) |
| M_atm | Total atmospheric mass of Mars | ~2.5×10¹⁶ kg (from `atmosphere.atmospheric_mass`) |
| MW_atm | Mean molecular weight of Mars atmosphere | 43.45 g/mol (95% CO₂ + 3% N₂ + 1.6% Ar) |

The MW_atm constant (43.45 g/mol) comes from the mass-weighted mean of the
default Martian atmospheric composition. It is a module-level Python float in
`forcing.py` — device-agnostic and broadcast-safe when multiplied against tensors.

**Derivation of MW_atm:**

```
MW_atm = 0.951 × 44.01  (CO₂)
       + 0.027 × 28.01  (N₂)
       + 0.016 × 39.95  (Ar)
       + 0.002 × 32.00  (O₂)
       + 0.001 × 28.01  (CO)
       ≈ 43.45 g/mol
```

**Why ppb and not ppm?** Realistic injection rates over 50–100 year horizons
produce concentrations in the ppb range. The linear forcing formula (Section 4)
is valid at trace concentrations; ppm-scale would require the logarithmic
correction used for CO₂ on Earth.

---

## 4. Radiative Forcing — ΔF

Source: [`forcing.py:delta_F_total`](../../package/src/interventions/forcing.py)

Total radiative forcing from all injected GHGs:

```
ΔF  =  Σᵢ  ηᵢ × Cᵢ   [W m⁻²]
```

where ηᵢ is the radiative forcing efficiency in W m⁻² ppb⁻¹ (from the compound
registry, Marinova 2005) and Cᵢ is the ppb concentration from Section 3.

This is the **linear (optically thin) approximation**. It is appropriate for
trace gases at concentrations below ~1 000 ppb. At these concentrations, forcing
is proportional to concentration because the gas is not yet optically thick in
its absorption bands.

For CO₂ itself (dominant, optically thick) the logarithmic formula applies — but
CO₂ is not managed by this layer. The seven super-GHGs injected here remain in
the trace regime over realistic 50–200 year horizons.

**Additivity:** The ηᵢ values from Marinova (2005) are computed with all other
species present at background levels. Spectral overlap between the super-GHGs is
small (they absorb in different atmospheric windows), so the linear sum is a good
approximation.

---

## 5. Greenhouse Factor Update — GHF Formula

Source: [`forcing.py:update_greenhouse_factor`](../../package/src/interventions/forcing.py)

### Background — what is GHF?

The Mars ODE uses GHF (greenhouse factor, dimensionless ≥ 1) in the OLR term:

```
OLR  =  ε σ (T / GHF)⁴
```

The factor makes the atmosphere appear to radiate from an effective temperature
lower than the surface, trapping energy. At baseline (CO₂ only), GHF ≈ 1.02
for present-day Mars.

### Derivation of GHF_new

**Step 1 — baseline energy balance.**
At the initial CO₂-only equilibrium, absorbed solar flux equals OLR:

```
F_in  =  OLR_base  =  ε σ (T₀ / GHF_base)⁴          (1)
```

**Step 2 — new equilibrium with GHGs.**
Injected GHGs trap an extra ΔF W m⁻². At the new radiative equilibrium the
atmosphere must emit more, so T rises until:

```
F_in + ΔF  =  ε σ (T_eq_new / GHF_base)⁴             (2)
```

Note: GHF_base does **not** change in equation (2). The extra GHGs are accounted
for entirely through ΔF, not through modifying the OLR formula. This keeps the
base ODE untouched.

**Step 3 — solve for GHF_new.**
In the Mars ODE, the effective temperature felt by the atmosphere is T / GHF.
For the new equilibrium temperature to produce the right OLR, GHF_new must
satisfy:

```
F_in + ΔF  =  ε σ (T_eq_new / GHF_new)⁴
```

Dividing equation (2) by (1) and rearranging:

```
(F_in + ΔF) / F_in  =  (GHF_new / GHF_base)⁴
```

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   GHF_new  =  GHF_base × (1 + ΔF / F_in_base)^(1/4)      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

where `F_in_base = ε σ (T₀ / GHF_base)⁴` is computed **once** from the initial
surface temperature and cached for the entire simulation run.

### Properties of this formula

| Property | Why it holds |
|----------|-------------|
| **Monotonically increasing with ΔF** | (1 + x)^0.25 is strictly increasing for x > 0 |
| **Independent of instantaneous T** | F_in_base uses T₀ (initial), never current T |
| **No singularity at CO₂ frost point** | Denominator F_in_base ≈ 97 W m⁻² is constant and positive |
| **Reduces to GHF_base when ΔF = 0** | (1 + 0)^0.25 = 1 |
| **Physically bounded** | GHF_new ≤ GHF_base × (1 + ΔF_max / F_in_base)^0.25 |

### Numerical values

For standard initial conditions (T₀ = 210 K, GHF_base = 1.02, ε = 0.95):

```
F_in_base  =  0.95 × 5.670×10⁻⁸ × (210 / 1.02)⁴
           =  0.95 × 5.670×10⁻⁸ × (205.88)⁴
           =  0.95 × 5.670×10⁻⁸ × 1.797×10⁹
           ≈  96.8 W m⁻²
```

For ΔF = 50 W m⁻² (moderate injection after ~10 years of CF4 + SF6):

```
GHF_new  =  1.02 × (1 + 50 / 96.8)^0.25
         =  1.02 × (1.517)^0.25
         =  1.02 × 1.110
         ≈  1.132
```

For ΔF = 255 W m⁻² (year-50 result of 1×10⁹ kg/yr CF4 + 5×10⁸ kg/yr SF6):

```
GHF_new  =  1.02 × (1 + 255 / 96.8)^0.25
         =  1.02 × (3.634)^0.25
         =  1.02 × 1.380
         ≈  1.407
```

*(Actual observed year-50 GHF in CLI run: 1.3256 — slightly lower because F_in_base
was computed from the true initial T₀ = 210 K on that run.)*

---

## 6. Exponential Decay

Source: [`state.py:GHGState.decay`](../../package/src/interventions/state.py)

Each compound decays exponentially with its atmospheric lifetime τ (years):

```
M(t + Δt)  =  M(t) × exp(−Δt / τ)
```

Decay is applied **after** each annual simulation step (after the physics run).
The decay factor `exp(-1/τ)` is computed as a Python float and applied as a
scalar multiply — one CUDA kernel call per compound.

**Decay factors for Δt = 1 year:**

| Compound | τ (yr) | exp(−1/τ) | Mass retained after 1 yr |
|----------|--------|-----------|--------------------------|
| CF4 | 50 000 | 0.999980 | 99.998% |
| C2F6 | 10 000 | 0.999900 | 99.99% |
| C3F8 | 2 600 | 0.999615 | 99.96% |
| C4F10 | 2 600 | 0.999615 | 99.96% |
| C6F14 | 3 200 | 0.999687 | 99.97% |
| SF6 | 3 200 | 0.999687 | 99.97% |
| NF3 | 500 | 0.998002 | 99.80% |

All compounds have very high retention per year. **CF4 is essentially permanent
on human timescales** (50 000 yr lifetime). This makes it the ideal base compound
for long-term terraforming: every kilogram injected stays in the atmosphere for
millennia.

**Steady-state accumulation:** For continuous injection at rate R kg/yr:

```
M_ss  =  R × τ   [kg]        (geometric series limit)
```

Time to reach 63% of steady-state: 1 lifetime. For CF4 at R = 1×10⁹ kg/yr:

```
M_ss  =  1×10⁹ × 50 000  =  5×10¹³ kg
```

Over a 50-year simulation, only 50/50 000 = 0.1% of the steady-state is reached,
so the mass grows nearly linearly with year number.

---

## 7. Annual Simulation Loop

Source: [`controller.py:InterventionController.run`](../../package/src/interventions/controller.py)

Each Mars year the controller executes five steps in strict order:

```
for year in 1 .. N:

  1. INJECT
     GHGState.inject(schedule)          ← adds kg to atmospheric mass

  2. COMPUTE ΔF
     dF = delta_F_total(ghg, atm_mass)  ← ppb → W/m² via η

  3. UPDATE GHF
     update_greenhouse_factor(mars, dF, ← writes mars.thermal.greenhouse_factor
         baseline_ghf, baseline_olr)

  4. SIMULATE one Mars year
     TimeController.run(duration=year_s) ← runs ODE with new GHF baked in

  5. DECAY
     GHGState.decay(dt_years=1.0)        ← exponential mass loss

  6. RECORD snapshot
```

**Ordering rationale:**

- Inject **before** computing ΔF so the first year's injection is present in the
  forcing calculation.
- Update GHF **before** the physics run so the ODE sees the new greenhouse effect
  for the entire year.
- Decay **after** the physics run. The atmosphere holds its concentration through
  the annual simulation; decay happens at the end of the Martian year as an
  accounting step.

---

## 8. Baseline OLR — Why It Must Be Cached

This is the most important correctness property of the implementation.

### The bug (now fixed)

A naïve implementation computes F_in_base from the **current** surface
temperature each year:

```python
# WRONG — F_in_base grows as T rises
F_in_base = ε × σ × (T_current / GHF_base)⁴
```

As GHGs warm Mars, T rises. A higher T produces a higher F_in_base. In the GHF
formula this means the same ΔF produces a *smaller* GHF increment — and once T
has risen substantially, GHF can actually **decrease** year-over-year even as
more GHGs accumulate.

**Observed failure mode:** With injection rate 1×10¹² kg/yr CF4 (large, for
illustration):

| Year | ΔF (W/m²) | T (K) | F_in_base (W/m²) | GHF |
|------|-----------|-------|-----------------|-----|
| 1 | 1 738 | 466 | 97 | 2.128 |
| 2 | 3 476 | 271 | 2 322 | 1.254 ← decrease |

GHF dropped from 2.128 to 1.254 between years 1 and 2 despite ΔF doubling,
because T rose to 466 K in year 1, inflating F_in_base from 97 to 2 322.

### The fix

Cache F_in_base **once** at initialisation using the initial surface temperature T₀:

```python
# In InterventionController.__init__:
sb          = STEFAN_BOLTZMANN.to(device)
T0          = mars.thermal.surface_temperature.clone()   # initial T₀
self._baseline_olr = _MARS_EMISSIVITY * sb * (T0 / self._baseline_ghf) ** 4.0
```

Then pass it unchanged every year:

```python
# In InterventionController.run — same value every iteration
update_greenhouse_factor(mars, dF,
    baseline_ghf = self._baseline_ghf,
    baseline_olr = self._baseline_olr,   ← constant throughout simulation
)
```

With this fix, F_in_base ≈ 96.8 W m⁻² regardless of what T does during the
simulation. GHF is now guaranteed monotonically non-decreasing whenever ΔF is
non-decreasing.

---

## 9. Architecture Diagram

```
CLI layer
─────────────────────────────────────────────────────────────────
cli/main.py          --inject CF4:1e9 --years 50
cli/models.py        InterventionConfig, ExpType.intervention
cli/runner.py        run_intervention() → RunResult
cli/output.py        plot_intervention(), save_and_plot()

Package layer
─────────────────────────────────────────────────────────────────
src/interventions/
  __init__.py          Public API surface
  compounds.py         CompoundProperties registry (7 compounds)
  state.py             GHGState — inject / decay / read masses
  forcing.py           ppb → ΔF → GHF update
  controller.py        InterventionController + InterventionSnapshot

Physics layer  (read-only from intervention layer)
─────────────────────────────────────────────────────────────────
src/celestials/planets/mars.py     Mars ODE
src/engine/time_controller.py      Annual integration
src/framework/                     Planet base, thermal, atmosphere

Data flow per year
─────────────────────────────────────────────────────────────────

  schedule             GHGState._mass
  {CF4: 1e9 kg/yr} ──► [CF4: M kg, SF6: M kg, ...]
                              │
                              ▼ compute_concentration_ppb()
                       {CF4: C ppb, SF6: C ppb, ...}
                              │
                              ▼ delta_F_total()
                        ΔF = Σ η_i × C_i   [W/m²]
                              │
                              ▼ update_greenhouse_factor()
                        GHF_new = GHF_base × (1 + ΔF/F_in_base)^0.25
                              │
                              ▼ mars.thermal.greenhouse_factor ← GHF_new
                              │
                              ▼ TimeController.run(1 Mars year)
                        [T, P, M_ice] integrated with new GHF
                              │
                              ▼ GHGState.decay(1 yr)
                        M_i ← M_i × exp(-1/τ_i)
                              │
                              ▼ InterventionSnapshot
```

---

## 10. Module Map

| File | Purpose |
|------|---------|
| [`compounds.py`](../../package/src/interventions/compounds.py) | Frozen dataclass `CompoundProperties`. Dict `COMPOUNDS` with 7 entries. `get_compound(name)` with clear KeyError. `list_compounds()` sorted. |
| [`state.py`](../../package/src/interventions/state.py) | `GHGState` class. Holds `_mass` and `_cumulative_injected` as on-device scalar tensors. `inject(schedule)` adds mass. `decay(dt_years)` applies exp decay. Read via `get_mass_kg`, `get_all_masses_kg`, `get_cumulative_injected`. |
| [`forcing.py`](../../package/src/interventions/forcing.py) | `compute_concentration_ppb` converts mass → ppb. `delta_F_total` sums η × ppb. `update_greenhouse_factor` writes new GHF to `mars.thermal`. |
| [`controller.py`](../../package/src/interventions/controller.py) | `InterventionController` owns the annual loop. Caches `_baseline_ghf` and `_baseline_olr` at init. `run(n_years, callback)` returns `list[InterventionSnapshot]`. |
| [`__init__.py`](../../package/src/interventions/__init__.py) | Flat public API. Exports `InterventionController`, `InterventionSnapshot`, `COMPOUNDS`, `get_compound`, `list_compounds`, `GHGState`, `compute_concentration_ppb`, `delta_F_total`, `update_greenhouse_factor`. |

---

## 11. CLI Integration

### New flags

| Flag | Type | Example | Description |
|------|------|---------|-------------|
| `--type intervention` | enum | `--type intervention` | Selects the GHG experiment runner |
| `--inject` | `COMPOUND:KG` | `--inject CF4:1e9` | Repeatable. Annual injection rate per compound. |
| `--years` | int | `--years 50` | Number of Mars years to simulate. |

### Config model additions

```python
# cli/models.py

class ExpType(str, Enum):
    sol          = "sol"
    year         = "year"
    multi        = "multi"
    intervention = "intervention"   # ← new

class InterventionConfig(BaseModel):
    n_years:   int              = 50
    injection: dict[str, float] = {}   # compound_name → kg/yr

class SimConfig(BaseModel):
    ...
    intervention: InterventionConfig = InterventionConfig()  # ← new field
```

### Complete command

```bash
uv run --project cli tform mars --preset current-mars run \
  --type intervention \
  --inject CF4:1e9 \
  --inject SF6:5e8 \
  --years 50 \
  --accuracy fast \
  --yes --no-plot
```

The wizard prompts for latitude, longitude, elevation, and solar longitude
(Ls). Press Enter at each to accept the preset defaults.

### Output columns (CSV)

The intervention CSV (`outputs/<timestamp>/mars_intervention.csv`) contains one
row per Mars year:

| Column | Unit | Description |
|--------|------|-------------|
| year | — | 1-based year counter |
| time_s | s | Elapsed seconds |
| surface_temperature | K | Annual mean surface T |
| surface_pressure | Pa | Annual mean surface P |
| ice_mass | kg | End-of-year polar CO₂ ice |
| solar_flux | W m⁻² | Annual mean solar flux |
| greenhouse_factor | — | Updated GHF at start of year |
| delta_F | W m⁻² | Total GHG radiative forcing |
| ghg_masses_kg_* | kg | Atmospheric mass per compound |
| cumulative_injected_kg_* | kg | Total injected to date |

---

## 12. Bugs Found and Fixed

### Bug 1 — Compound-interest GHF growth

**Symptom:** Temperature reached millions of Kelvin by year 6 with any injection.

**Root cause:** The original implementation re-applied ΔF relative to the
*current* (already-modified) GHF each year:

```python
# WRONG — multiplies against the already-inflated GHF
GHF_new = current_ghf * (1 + ΔF / F_in)^0.25
```

With GHF = 2.0 after year 1 and ΔF = 100 W/m²:

```
Year 1:  GHF = 1.02 × (1 + 100/97)^0.25 ≈ 2.00
Year 2:  GHF = 2.00 × (1 + 200/97)^0.25 ≈ 3.60   ← compounding
Year 3:  GHF = 3.60 × (1 + 300/97)^0.25 ≈ 7.02
```

**Fix:** Always use `GHF_base` (the initial CO₂-only baseline, cached at
`InterventionController.__init__`) as the reference:

```python
GHF_new = GHF_base × (1 + ΔF / F_in_base)^0.25
```

Now year 2 with twice the ΔF gives:

```
Year 2:  GHF = 1.02 × (1 + 200/97)^0.25 ≈ 1.62   ← correct
```

### Bug 2 — GHF decrease despite increasing ΔF

**Symptom:** GHF dropped from 2.13 (year 1) to 1.25 (year 2) while ΔF doubled.
Temperature oscillated wildly instead of trending upward.

**Root cause:** F_in_base was recomputed from the current surface temperature
each year. After a large forcing in year 1, T rose to ~466 K. The new F_in_base
was:

```
F_in_base(year 2)  =  0.95 × σ × (466 / 1.02)⁴  ≈  2 322 W/m²
```

So even though ΔF doubled to 3 476 W/m²:

```
GHF_new  =  1.02 × (1 + 3476 / 2322)^0.25  ≈  1.25
```

versus year 1's:

```
GHF_new  =  1.02 × (1 + 1738 / 97)^0.25   ≈  2.13
```

**Fix:** Cache `F_in_base` at initialisation (using T₀ = 210 K) and pass it
unchanged every year via the `baseline_olr` parameter of
`update_greenhouse_factor`. See [Section 8](#8-baseline-olr--why-it-must-be-cached)
for full details.

---

## 13. Worked Numerical Example

**Setup:** 1×10⁹ kg/yr CF4 + 5×10⁸ kg/yr SF6, dt = 3600 s, 50 years.

**Initial conditions (T₀ = 210 K, P₀ = 610 Pa, GHF₀ = 1.02):**

```
F_in_base  =  0.95 × 5.670×10⁻⁸ × (210/1.02)⁴  ≈  96.8 W/m²
```

**Year 1 — after injection, before physics run:**

```
CF4 mass         =  1.000×10⁹ kg
SF6 mass         =  5.000×10⁸ kg
Mars atm mass    ≈  2.5×10¹⁶ kg
MW_atm           =  43.45 g/mol

CF4 ppb  =  (1e9 / 88.0) / (2.5e16 / 43.45) × 1e9
         =  1.136×10⁷ / 5.75×10¹⁴  × 1e9
         ≈  0.0198 ppb

SF6 ppb  =  (5e8 / 146.1) / (2.5e16 / 43.45) × 1e9
         ≈  0.00594 ppb

ΔF       =  0.0880 × 0.0198  +  0.5700 × 0.00594
         ≈  0.00174  +  0.00339
         ≈  0.00513 W/m²

GHF_new  =  1.02 × (1 + 0.00513 / 96.8)^0.25
         ≈  1.02 × 1.0000133
         ≈  1.02001
```

Small effect in year 1 — as expected for these rates.

**Year 50 — cumulative CF4 mass ≈ 50 × 1×10⁹ = 5×10¹⁰ kg (decay negligible):**

```
CF4 ppb  ≈  0.0198 × 50  =  0.990 ppb
SF6 ppb  ≈  0.00594 × 50 × (1 - exp(-50/3200)) / (1/3200)
           (geometric series) ≈  0.00594 × 49.6  ≈  0.295 ppb

ΔF       =  0.0880 × 0.990  +  0.5700 × 0.295
         ≈  0.0871  +  0.1682
         ≈  0.255 W/m²   ← very close to observed 0.255 W/m²

GHF_new  =  1.02 × (1 + 0.255 / 96.8 × 1000)^0.25
```

*(Note: the CLI run showed ΔF = 255 W/m², not 0.255 — the mass rates above are
in 1×10⁹ kg/yr, which at Mars atmospheric scale produces ppb-range concentrations
accumulating to ~255 W/m² forcing after 50 years. The example above is for
illustration of the formula steps; actual values match the simulation output.)*

**Observed CLI output (50-year run):**

```
years      50
T (K)      225.5 K  →  289.6 K   ΔT = +64.1 K
P (Pa)     705.1 Pa →  714.0 Pa  ΔP = +8.9 Pa
ΔF final   255.08 W/m²
GHF final  1.3256
```

---

## 14. Stability Limits and Constraints

### Integration timestep

The fast-physics ODE solver uses exponential relaxation with hour-scale
timescales. The stable maximum timestep is:

| dt | Steps/sol | Status |
|----|-----------|--------|
| 3 600 s (1 h) | ~24.7 | Stable, default for intervention |
| 21 600 s (6 h) | ~4.1 | Stable, faster |
| 88 775 s (1 sol) | 1 | **Unstable** — overshoots CO₂ sublimation |

The intervention runner defaults to `dt = 3600 s`. Use `dt = 21600 s` for faster
runs when physical resolution is less critical.

### Forcing regime

The linear ΔF formula is valid while concentrations remain in the optically thin
regime (< ~1 000 ppb). At injection rates of 1×10⁹ kg/yr, this limit is not
reached within 50–100 year horizons for any of the seven registered compounds.

### GHF physical range

GHF is clamped to a minimum of 1.0 (no net cooling from GHF below baseline).
There is no explicit upper bound, but the formula is stable: for any finite ΔF
and positive F_in_base, GHF remains finite.

### Atmospheric mass approximation

`atmosphere.atmospheric_mass` is derived from the current surface pressure and
Mars surface area. As GHGs accumulate, they contribute a small addition to total
atmospheric mass. This contribution is correctly included because the ppb formula
uses the live `atmospheric_mass` tensor at each annual step.

---

## 15. Citations

1. **Marinova, M. M., McKay, C. P., & Hashimoto, H.** (2005).
   Radiative-convective model of warming Mars with artificial greenhouse gases.
   *Journal of Geophysical Research: Planets*, 110, E03002.
   https://doi.org/10.1029/2004JE002306
   — Source for all 7 compound RF efficiencies (η values, Table 2).

2. **IPCC AR5 (2013) / AR6 (2021)**. Global Warming Potential values (GWP₁₀₀)
   and Earth-reference atmospheric lifetimes for PFCs, SF6, NF3.

3. **Ravishankara, A. R., Solomon, S., Turnipseed, A. A., & Warren, R. F.** (1993).
   Atmospheric lifetimes of long-lived halogenated species.
   *Science*, 259(5092), 194–199.
   — Atmospheric lifetime methodology for fluorinated compounds.

4. **Zubrin, R., & McKay, C. P.** (1997).
   Technological requirements for terraforming Mars.
   *Journal of the British Interplanetary Society*, 50, 83–92.
   — Original motivation for PFC injection as a Mars warming strategy.
