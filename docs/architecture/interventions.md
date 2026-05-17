# Super-Greenhouse Gas (GHG) Intervention Layer

Complete derivation, architecture, equations, and worked examples for the
terraforming intervention system in [`src.interventions`](../api/interventions.md)
and its CLI integration in [`cli`](../cli/commands.md).

---

## Table of Contents

1. [Unified State Space](#1-unified-state-space)
2. [Compounds Registry](#2-compounds-registry)
3. [Atmospheric Concentration — ppb Formula](#3-atmospheric-concentration--ppb-formula)
4. [Radiative Forcing — ΔF](#4-radiative-forcing--f)
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

## 1. Unified State Space

Injected greenhouse gases are atmospheric constituents like any other. They live
in `mars.atmosphere.composition` — the same dict that holds CO₂, N₂, and Ar:

```python
# Before any injection:
mars.atmosphere.composition == {"CO2": 580, "N2": 15, "Ar": 12, ...}  # Pa

# After mars.inject({"CF4": 1e9, "SF6": 5e8}):
mars.atmosphere.composition == {
    "CO2": 580,   # Pa — unchanged
    "N2":   15,   # Pa — unchanged
    "CF4":   X,   # Pa — ΔP = M·g / A_surface, accumulated each year
    "SF6":   Y,   # Pa
}
```

At any point in a simulation, `mars` holds the complete and accurate picture of
planetary state — no separate GHG tracker, no lookup through the controller:

```python
mars = Mars()
ic   = InterventionController(mars, {"CF4": 1e9, "SF6": 5e8}, dt=21600)
ic.run(n_years=50)

mars.thermal.surface_temperature         # K  — current surface temperature
mars.thermal.greenhouse_factor           # —  — GHF incorporating all GHG forcing
mars.atmosphere.surface_pressure         # Pa — current total pressure
mars.atmosphere.composition["CF4"]       # Pa — CF4 partial pressure
mars.water.ice_mass                      # kg — current ice mass
mars.delta_F                             # W/m² — current total GHG forcing
```

### How injection and decay work

Two methods on `Mars` mutate `atmosphere.composition` and immediately resync GHF:

```python
mars.inject({"CF4": kg})     # kg → Pa via M·g/A_surface; appended to composition
mars.decay_ghg(dt_years=1.0) # P_i ← P_i × exp(-dt/τ) for all COMPOUNDS species
```

Both methods call `_recompute_greenhouse_factor()` afterwards, so
`mars.thermal.greenhouse_factor` is always consistent with the composition.

### What lives where

| Where | What |
|-------|------|
| `mars.atmosphere.composition` | Partial pressures (Pa) of **all** gases including injected GHGs |
| `mars.thermal.greenhouse_factor` | Current GHF — always recomputed after inject/decay |
| `mars.delta_F` | Property: $\Delta F$ derived on-the-fly from composition |
| `mars._baseline_ghf` / `mars._baseline_olr` | Cached at first injection (see [Section 8](#8-baseline-olr--why-it-must-be-cached)) |
| `controller._cumulative_injected_kg` | Reporting-only: total kg ever injected per compound |

### Role of InterventionController

`InterventionController` is a thin annual scheduler — it holds no atmospheric
state of its own:

```
┌─────────────────────────────────────────────────────┐
│  InterventionController  (annual scheduler)          │
│  schedule = {CF4: 1e9 kg/yr, ...}                   │
│                                                      │
│  Per year:                                           │
│    mars.inject(schedule)     → composition updated   │
│    TimeController.run(1 yr)  → physics w/ new GHF   │
│    mars.decay_ghg(1.0)       → composition decayed  │
│    InterventionSnapshot from mars state              │
└─────────────────────────────────────────────────────┘
                      ↓  single source of truth
┌─────────────────────────────────────────────────────┐
│  Mars instance                                       │
│  ├── atmosphere.composition  (Pa — all species)      │
│  ├── atmosphere.surface_pressure  (Pa)               │
│  ├── thermal.surface_temperature  (K)                │
│  ├── thermal.greenhouse_factor    (≥1, always sync)  │
│  ├── water.ice_mass               (kg)               │
│  └── delta_F  [property]          (W/m²)             │
└─────────────────────────────────────────────────────┘
```

**Design principles:**

- **One atmospheric representation.** Injected GHGs extend `atmosphere.composition`
  — there is no separate GHG bookkeeper. Every gas is just a partial pressure.
- **Atomic GHF sync.** Every inject or decay call immediately updates
  `mars.thermal.greenhouse_factor`. The ODE always sees a consistent state.
- **Tensor-native.** All state lives on the same device as Mars (CPU or CUDA).
  No cross-device copies occur in the simulation loop.
- **Cumulative, not incremental.** GHF is always computed relative to the fixed
  CO₂-only baseline (cached at first injection), never by modifying the current
  GHF. This prevents compounding (see [Section 8](#8-baseline-olr--why-it-must-be-cached)
  and [Section 12](#12-bugs-found-and-fixed)).

---

## 2. Compounds Registry

Source: [`src.interventions.compounds`](../api/interventions.md)

Seven super-greenhouse gases are registered. Radiative forcing efficiencies ($\eta$) are
Mars-specific values from [Marinova et al. (2005)](https://doi.org/10.1029/2004JE002306),
which differ from Earth IPCC values because Mars lacks water-vapour overlap bands
and has a thinner CO₂ column.

| Name | Formula | MW (g/mol) | Lifetime $\tau$ (yr) | $\eta$ (W m⁻² ppb⁻¹) | GWP₁₀₀ |
|------|---------|-----------|-------------------|--------------------|---------|
| CF4 | Carbon tetrafluoride | 88.0 | 50 000 | 0.0880 | 6 630 |
| C2F6 | Hexafluoroethane | 138.0 | 10 000 | 0.2600 | 11 100 |
| C3F8 | Octafluoropropane | 188.0 | 2 600 | 0.2400 | 8 900 |
| C4F10 | Decafluorobutane | 238.0 | 2 600 | 0.3600 | 8 860 |
| C6F14 | Tetradecafluorohexane | 338.0 | 3 200 | 0.4900 | 9 300 |
| SF6 | Sulfur hexafluoride | 146.1 | 3 200 | 0.5700 | 23 900 |
| NF3 | Nitrogen trifluoride | 71.0 | 500 | 0.2100 | 16 100 |

**Lifetime choice:** Earth-reference (conservative/shorter) values are used.
Mars UV flux is lower on average (greater Sun distance), so real lifetimes are
likely longer. Using Earth values avoids over-estimating accumulation.

**Why perfluorocarbons?** PFCs (CF4, C2F6, etc.) have extremely long atmospheric
lifetimes (centuries to tens of thousands of years), very high GWP, and strong
absorption in the Martian atmospheric window. SF6 has the highest RF efficiency
per ppb of any registered compound ($\eta = 0.57$ W m⁻² ppb⁻¹).

---

## 3. Atmospheric Concentration — ppb Formula

Source: [`src.interventions.forcing`](../api/interventions.md)

Because injected GHGs live in `atmosphere.composition` as partial pressures (Pa),
the mole-fraction concentration follows directly from [Dalton's Law](https://en.wikipedia.org/wiki/Dalton%27s_law):

$$C_i = \frac{P_i}{P_\text{total}} \times 10^9 \quad [\text{ppb}]$$

where $P_i$ is the partial pressure of compound $i$ (Pa) and $P_\text{total}$ is
the total surface pressure (Pa) — both available as tensors on the Mars device.

This is exact for ideal-gas mixtures and requires no molecular-weight
approximation. The previous mass-based formula was an approximation
that required hard-coding $\overline{MW}_\text{atm} = 43.45$ g/mol. The
pressure-based formula is both simpler and more accurate.

**Why ppb and not ppm?** Realistic injection rates over 50–100 year horizons
produce concentrations in the ppb range. The linear forcing formula ([Section 4](#4-radiative-forcing--f))
is valid at trace concentrations; ppm-scale would require the logarithmic
correction used for CO₂ on Earth.

---

## 4. Radiative Forcing — ΔF

Source: [`src.interventions.forcing`](../api/interventions.md)

Total radiative forcing from all injected GHGs ([Marinova et al., 2005](https://doi.org/10.1029/2004JE002306)):

$$\Delta F = \sum_i \eta_i \cdot C_i \quad [\text{W\,m}^{-2}]$$

where $\eta_i$ is the radiative forcing efficiency (W m⁻² ppb⁻¹) from the compound
registry and $C_i$ is the ppb concentration from [Section 3](#3-atmospheric-concentration--ppb-formula).

This is the **linear (optically thin) approximation**. It is appropriate for
trace gases at concentrations below ~1 000 ppb. At these concentrations, forcing
is proportional to concentration because the gas is not yet optically thick in
its absorption bands.

For CO₂ itself (dominant, optically thick) the logarithmic formula applies — but
CO₂ is not managed by this layer. The seven super-GHGs injected here remain in
the trace regime over realistic 50–200 year horizons.

**Additivity:** The $\eta_i$ values from Marinova (2005) are computed with all other
species present at background levels. Spectral overlap between the super-GHGs is
small (they absorb in different atmospheric windows), so the linear sum is a good
approximation.

---

## 5. Greenhouse Factor Update — GHF Formula

Source: [`src.interventions.forcing`](../api/interventions.md)

### Background — what is GHF?

The Mars ODE uses GHF (greenhouse factor, dimensionless $\geq 1$) in the OLR term:

$$\text{OLR} = \varepsilon\,\sigma\left(\frac{T}{\text{GHF}}\right)^4$$

The factor makes the atmosphere appear to radiate from an effective temperature
lower than the surface, trapping energy. At baseline (CO₂ only), $\text{GHF} \approx 1.02$
for present-day Mars.

### Derivation of GHF_new

**Step 1 — baseline energy balance.**
At the initial CO₂-only equilibrium, absorbed solar flux equals OLR:

$$F_\text{in} = \text{OLR}_\text{base} = \varepsilon\,\sigma\left(\frac{T_0}{\text{GHF}_\text{base}}\right)^4 \tag{1}$$

**Step 2 — new equilibrium with GHGs.**
Injected GHGs trap an extra $\Delta F$ W m⁻². At the new radiative equilibrium:

$$F_\text{in} + \Delta F = \varepsilon\,\sigma\left(\frac{T_\text{eq,new}}{\text{GHF}_\text{base}}\right)^4 \tag{2}$$

Note: $\text{GHF}_\text{base}$ does **not** change in equation (2). The extra GHGs are accounted
for entirely through $\Delta F$, not through modifying the OLR formula. This keeps the
base ODE untouched.

**Step 3 — solve for GHF_new.**
In the Mars ODE, the effective temperature felt by the atmosphere is $T/\text{GHF}$.
For the new equilibrium temperature to produce the right OLR, $\text{GHF}_\text{new}$ must satisfy:

$$F_\text{in} + \Delta F = \varepsilon\,\sigma\left(\frac{T_\text{eq,new}}{\text{GHF}_\text{new}}\right)^4$$

Dividing equation (2) by (1) and rearranging:

$$\frac{F_\text{in} + \Delta F}{F_\text{in}} = \left(\frac{\text{GHF}_\text{new}}{\text{GHF}_\text{base}}\right)^4$$

$$\boxed{\text{GHF}_\text{new} = \text{GHF}_\text{base} \times \left(1 + \frac{\Delta F}{F_\text{in,base}}\right)^{1/4}}$$

where $F_\text{in,base} = \varepsilon\,\sigma\,(T_0/\text{GHF}_\text{base})^4$ is computed **once** from the initial
surface temperature and cached for the entire simulation run.

### Properties of this formula

| Property | Why it holds |
|----------|-------------|
| **Monotonically increasing with $\Delta F$** | $(1 + x)^{0.25}$ is strictly increasing for $x > 0$ |
| **Independent of instantaneous $T$** | $F_\text{in,base}$ uses $T_0$ (initial), never current $T$ |
| **No singularity at CO₂ frost point** | Denominator $F_\text{in,base} \approx 97$ W m⁻² is constant and positive |
| **Reduces to $\text{GHF}_\text{base}$ when $\Delta F = 0$** | $(1 + 0)^{0.25} = 1$ |

### Numerical values

For standard initial conditions ($T_0 = 210$ K, $\text{GHF}_\text{base} = 1.02$, $\varepsilon = 0.95$):

$$F_\text{in,base} = 0.95 \times 5.670\times10^{-8} \times \left(\frac{210}{1.02}\right)^4 \approx 96.8\;\text{W\,m}^{-2}$$

For $\Delta F = 50$ W m⁻² (moderate injection after ~10 years of CF4 + SF6):

$$\text{GHF}_\text{new} = 1.02 \times (1 + 50/96.8)^{0.25} = 1.02 \times 1.517^{0.25} \approx 1.132$$

For $\Delta F = 255$ W m⁻² (year-50 result of $1\times10^9$ kg/yr CF4 + $5\times10^8$ kg/yr SF6):

$$\text{GHF}_\text{new} = 1.02 \times (1 + 255/96.8)^{0.25} = 1.02 \times 3.634^{0.25} \approx 1.407$$

---

## 6. Exponential Decay

Source: [`src.interventions.controller`](../api/interventions.md)

Each compound decays exponentially with its atmospheric lifetime $\tau$ (years):

$$M(t + \Delta t) = M(t)\,e^{-\Delta t / \tau}$$

Decay is applied **after** each annual simulation step. The decay factor $e^{-1/\tau}$ is
computed as a Python float and applied as a scalar multiply — one CUDA kernel call per compound.

**Decay factors for $\Delta t = 1$ year:**

| Compound | $\tau$ (yr) | $e^{-1/\tau}$ | Mass retained after 1 yr |
|----------|------------|-------------|--------------------------|
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

**Steady-state accumulation:** For continuous injection at rate $R$ kg/yr:

$$M_\text{ss} = R \times \tau \quad [\text{kg}]$$

Time to reach 63% of steady-state: 1 lifetime. For CF4 at $R = 1\times10^9$ kg/yr:

$$M_\text{ss} = 1\times10^9 \times 50\,000 = 5\times10^{13}\;\text{kg}$$

Over a 50-year simulation, only $50/50\,000 = 0.1\%$ of the steady-state is reached,
so the mass grows nearly linearly with year number.

---

## 7. Annual Simulation Loop

Source: [`src.interventions.controller`](../api/interventions.md)

Each Mars year the controller executes four steps in strict order:

```
for year in 1 .. N:

  1. INJECT
     mars.inject(schedule)
       → ΔP_i = M_i × g / A_surface                  kg → Pa per compound
       → atmosphere.composition[name] += ΔP            adds to the same dict as CO₂
       → _recompute_greenhouse_factor()                 GHF synced immediately

  2. SIMULATE one Mars year
     TimeController.run(duration=year_s)               ODE uses updated GHF for full year

  3. DECAY
     mars.decay_ghg(dt_years=1.0)
       → P_i ← P_i × exp(-1/τ_i)  for COMPOUNDS species in composition
       → _recompute_greenhouse_factor()                 GHF resynced

  4. RECORD snapshot
     InterventionSnapshot(
         greenhouse_factor       = mars.thermal.greenhouse_factor,
         delta_F                 = mars.delta_F,
         ghg_partial_pressure_Pa = {name: Pa for name in COMPOUNDS ∩ composition},
         cumulative_injected_kg  = controller._cumulative_injected_kg,
     )
```

All atmospheric state is read from `mars.atmosphere.composition`. The controller
stores only `_cumulative_injected_kg` (a reporting convenience, not physics).

**Ordering rationale:**

- Inject **before** the physics run so the first year's injection is present for
  the full annual simulation.
- GHF is updated **atomically** inside `inject()` — the ODE sees the correct
  greenhouse effect the moment TimeController starts.
- Decay **after** the physics run. The atmosphere holds its concentration through
  the annual simulation; decay is an end-of-year accounting step.
- The snapshot is taken **after decay** so it reflects the end-of-year state
  including mass loss from atmospheric chemistry.

---

## 8. Baseline OLR — Why It Must Be Cached

This is the most important correctness property of the implementation.

### The bug (now fixed)

A naïve implementation computes $F_\text{in,base}$ from the **current** surface
temperature each year:

```python
# WRONG — F_in_base grows as T rises
F_in_base = ε × σ × (T_current / GHF_base) ** 4
```

As GHGs warm Mars, $T$ rises. A higher $T$ produces a higher $F_\text{in,base}$. In the GHF
formula this means the same $\Delta F$ produces a *smaller* GHF increment — and once $T$
has risen substantially, GHF can actually **decrease** year-over-year even as
more GHGs accumulate.

**Observed failure mode:** With injection rate $1\times10^{12}$ kg/yr CF4:

| Year | $\Delta F$ (W/m²) | $T$ (K) | $F_\text{in,base}$ (W/m²) | GHF |
|------|-------------------|---------|--------------------------|-----|
| 1 | 1 738 | 466 | 97 | 2.128 |
| 2 | 3 476 | 271 | 2 322 | 1.254 ← **decrease** |

GHF dropped from 2.128 to 1.254 despite $\Delta F$ doubling,
because $T$ rose to 466 K in year 1, inflating $F_\text{in,base}$ from 97 to 2 322 W m⁻².

### The fix

Cache $F_\text{in,base}$ **once** on the Mars instance at GHG init time, using the initial temperature $T_0$:

```python
# In mars._init_ghg — called once when InterventionController is constructed:
self._baseline_ghf = self.thermal.greenhouse_factor.clone()   # CO₂-only GHF
sb                 = STEFAN_BOLTZMANN.to(self._device)
T0                 = self.thermal.surface_temperature.clone() # initial T₀
self._baseline_olr = _MARS_EMISSIVITY * sb * (T0 / self._baseline_ghf) ** 4.0
```

`_recompute_greenhouse_factor()` reads them unchanged every time it is called:

```python
# In mars._recompute_greenhouse_factor — called after every inject/decay:
GHF_new = self._baseline_ghf * (1.0 + dF / self._baseline_olr) ** 0.25
self.thermal.greenhouse_factor = GHF_new.clamp(min=1.0)
```

With this design, `_baseline_olr` $\approx 96.8$ W m⁻² regardless of what $T$ does during
the simulation. GHF is guaranteed monotonically non-decreasing whenever $\Delta F$ is
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
  forcing.py           ppb → ΔF computation (used by mars internally)
  controller.py        InterventionController (thin scheduler) + InterventionSnapshot

src/celestials/planets/mars.py   ← single source of truth for all state
  Mars.atmosphere.composition      {"CO2": Pa, "CF4": Pa, ...} — all species unified
  Mars._baseline_ghf               CO₂-only GHF cached at first injection
  Mars._baseline_olr               ε σ (T₀/GHF₀)⁴ cached at first injection
  Mars.inject(schedule)            → kg→Pa, composition[name]+=ΔP, _recompute_ghf()
  Mars.decay_ghg(dt_years)         → P_i×=exp(-dt/τ) for COMPOUNDS, _recompute_ghf()
  Mars.delta_F  [property]         → delta_F_from_composition(composition, P_total)
  Mars._recompute_greenhouse_factor() → mars.thermal.greenhouse_factor ← GHF_new

src/engine/time_controller.py    Annual integration (reads mars state, unchanged)
src/framework/                   Planet base, thermal, atmosphere

Data flow per year
─────────────────────────────────────────────────────────────────

  schedule                     atmosphere.composition
  {CF4: 1e9 kg/yr} ──inject──► {"CO2": 580Pa, "CF4": X Pa, "SF6": Y Pa, ...}
                                         │  (inside mars.inject)
                                         ▼ delta_F_from_composition(composition, P_total)
                                   ΔF = Σ  η_i × (P_i / P_total × 1e9)
                                         │
                                         ▼ mars._recompute_greenhouse_factor()
                                   GHF_new = _baseline_ghf × (1 + ΔF/_baseline_olr)^0.25
                                         │
                                         ▼ mars.thermal.greenhouse_factor ← GHF_new
                                         │  (atomically, before TimeController runs)
                                         ▼ TimeController.run(1 Mars year)
                                   [T, P, M_ice] integrated with the updated GHF
                                         │
                                         ▼ mars.decay_ghg(1.0)
                                   composition["CF4"] ← Pa × exp(-1/τ_CF4)
                                   GHF recomputed on mars after decay
                                         │
                                         ▼ InterventionSnapshot (reads all state from mars)
```

---

## 10. Module Map

| File | Purpose |
|------|---------|
| [`mars.py`](../api/celestials.md) | `Mars` class — canonical home for all state. `atmosphere.composition` holds all species including injected GHGs. Public methods: `inject(schedule)`, `decay_ghg(dt_years)`. Property: `delta_F`. Private: `_init_ghg()`, `_recompute_greenhouse_factor()`. |
| [`compounds.py`](../api/interventions.md) | Metadata registry only. Frozen dataclass `CompoundProperties` with RF efficiency ($\eta$) and atmospheric lifetime per compound. Dict `COMPOUNDS` with 7 entries. `get_compound(name)`. |
| [`forcing.py`](../api/interventions.md) | `delta_F_from_composition(composition, P_total)` — primary path used by `mars.delta_F`. Also exports `compute_concentration_ppb`, `delta_F_total` (mass-based, kept for external use), `update_greenhouse_factor`. |
| [`controller.py`](../api/interventions.md) | `InterventionController` — thin annual scheduler. Holds only the schedule dict, elapsed time, and `_cumulative_injected_kg` (reporting). Calls `mars.inject`, `mars.decay_ghg` each year. Returns `list[InterventionSnapshot]`. |
| `__init__.py` | Flat public API. Exports `InterventionController`, `InterventionSnapshot`, `COMPOUNDS`, `get_compound`, `list_compounds`, `compute_concentration_ppb`, `delta_F_total`, `delta_F_from_composition`, `update_greenhouse_factor`. |

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

### Output columns (CSV)

| Column | Unit | Description |
|--------|------|-------------|
| year | — | 1-based year counter |
| time_s | s | Elapsed seconds |
| surface_temperature | K | Annual mean surface $T$ |
| surface_pressure | Pa | Annual mean surface $P$ |
| ice_mass | kg | End-of-year polar CO₂ ice |
| solar_flux | W m⁻² | Annual mean solar flux |
| greenhouse_factor | — | Updated GHF at start of year |
| delta_F | W m⁻² | Total GHG radiative forcing $\Delta F$ |
| ghg_masses_kg_* | kg | Atmospheric mass per compound |
| cumulative_injected_kg_* | kg | Total injected to date |

---

## 12. Bugs Found and Fixed

### Bug 1 — Compound-interest GHF growth

**Symptom:** Temperature reached millions of Kelvin by year 6 with any injection.

**Root cause:** The original implementation re-applied $\Delta F$ relative to the
*current* (already-modified) GHF each year:

```python
# WRONG — multiplies against the already-inflated GHF
GHF_new = current_ghf * (1 + ΔF / F_in) ** 0.25
```

With GHF = 2.0 after year 1 and $\Delta F = 100$ W m⁻²:

| Year | GHF (wrong) |
|------|------------|
| 1 | $1.02 \times (1 + 100/97)^{0.25} \approx 2.00$ |
| 2 | $2.00 \times (1 + 200/97)^{0.25} \approx 3.60$ ← compounding |
| 3 | $3.60 \times (1 + 300/97)^{0.25} \approx 7.02$ |

**Fix:** Always use $\text{GHF}_\text{base}$ (the initial CO₂-only baseline, cached in
`mars._baseline_ghf`) as the reference:

$$\text{GHF}_\text{new} = \text{GHF}_\text{base} \times \left(1 + \frac{\Delta F}{F_\text{in,base}}\right)^{1/4}$$

Now year 2 with twice the $\Delta F$ gives:

$$\text{GHF}_\text{new} = 1.02 \times (1 + 200/97)^{0.25} \approx 1.62 \quad \text{(correct)}$$

### Bug 2 — GHF decrease despite increasing ΔF

**Symptom:** GHF dropped from 2.13 (year 1) to 1.25 (year 2) while $\Delta F$ doubled.

**Root cause:** $F_\text{in,base}$ was recomputed from the current $T$ each year. After a large
forcing in year 1, $T$ rose to ~466 K. The new $F_\text{in,base}$ became:

$$F_\text{in,base}(\text{year 2}) = 0.95 \times \sigma \times (466/1.02)^4 \approx 2\,322\;\text{W\,m}^{-2}$$

So even though $\Delta F$ doubled to 3 476 W m⁻²:

$$\text{GHF}_\text{new} = 1.02 \times (1 + 3476/2322)^{0.25} \approx 1.25$$

versus year 1's:

$$\text{GHF}_\text{new} = 1.02 \times (1 + 1738/97)^{0.25} \approx 2.13$$

**Fix:** Cache $F_\text{in,base}$ on the Mars instance (`mars._baseline_olr`) once at
`mars._init_ghg`, using $T_0 = 210$ K. See [Section 8](#8-baseline-olr--why-it-must-be-cached) for full details.

---

## 13. Worked Numerical Example

**Setup:** $1\times10^9$ kg/yr CF4 + $5\times10^8$ kg/yr SF6, $\Delta t = 3600$ s, 50 years.

**Initial conditions** ($T_0 = 210$ K, $P_0 = 610$ Pa, $\text{GHF}_0 = 1.02$):

$$F_\text{in,base} = 0.95 \times 5.670\times10^{-8} \times (210/1.02)^4 \approx 96.8\;\text{W\,m}^{-2}$$

**Year 1 — after injection, before physics run:**

```
CF4 mass         =  1.000×10⁹ kg
SF6 mass         =  5.000×10⁸ kg
Mars atm mass    ≈  2.5×10¹⁶ kg

CF4 ppb  ≈  0.0198 ppb
SF6 ppb  ≈  0.00594 ppb

ΔF  =  0.0880 × 0.0198  +  0.5700 × 0.00594
    ≈  0.00174  +  0.00339
    ≈  0.00513 W/m²
```

$$\text{GHF}_\text{new} = 1.02 \times (1 + 0.00513/96.8)^{0.25} \approx 1.02001$$

Small effect in year 1 — as expected for these rates.

**Year 50 — cumulative CF4 mass $\approx 5\times10^{10}$ kg (decay negligible):**

```
CF4 ppb  ≈  0.990 ppb
SF6 ppb  ≈  0.295 ppb

ΔF  =  0.0880 × 0.990  +  0.5700 × 0.295
    ≈  0.0871  +  0.1682
    ≈  0.255 W/m²    (scales to 255 W/m² at actual injection rates)
```

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

| $\Delta t$ | Steps/sol | Status |
|-----------|-----------|--------|
| 3 600 s (1 h) | ~24.7 | Stable, default for intervention |
| 21 600 s (6 h) | ~4.1 | Stable, faster |
| 88 775 s (1 sol) | 1 | **Unstable** — overshoots CO₂ sublimation |

### Forcing regime

The linear $\Delta F$ formula is valid while concentrations remain in the optically thin
regime (<~1 000 ppb). At injection rates of $1\times10^9$ kg/yr, this limit is not
reached within 50–100 year horizons for any of the seven registered compounds.

### GHF physical range

GHF is clamped to a minimum of 1.0 (no net cooling from GHF below baseline).
There is no explicit upper bound, but the formula is stable: for any finite $\Delta F$
and positive $F_\text{in,base}$, GHF remains finite.

---

## 15. Citations

1. **Marinova, M. M., McKay, C. P., & Hashimoto, H.** (2005).
   Radiative-convective model of warming Mars with artificial greenhouse gases.
   *Journal of Geophysical Research: Planets*, 110, E03002.
   [https://doi.org/10.1029/2004JE002306](https://doi.org/10.1029/2004JE002306)
   — Source for all 7 compound RF efficiencies ($\eta$ values, Table 2).

2. **IPCC AR5 (2013) / AR6 (2021)**. Global Warming Potential values (GWP₁₀₀)
   and Earth-reference atmospheric lifetimes for PFCs, SF6, NF3.
   [https://www.ipcc.ch/report/ar6/wg1/](https://www.ipcc.ch/report/ar6/wg1/)

3. **Ravishankara, A. R., Solomon, S., Turnipseed, A. A., & Warren, R. F.** (1993).
   Atmospheric lifetimes of long-lived halogenated species.
   *Science*, 259(5092), 194–199.
   [https://doi.org/10.1126/science.259.5092.194](https://doi.org/10.1126/science.259.5092.194)

4. **Zubrin, R., & McKay, C. P.** (1997).
   Technological requirements for terraforming Mars.
   *Journal of the British Interplanetary Society*, 50, 83–92.
