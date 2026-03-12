# Mars Physics Model

Complete derivation, dependency graph, and worked examples for every equation in
[`src/celestials/planets/mars.py`](../src/celestials/planets/mars.py) and its
supporting framework in [`src/framework/planet.py`](../src/framework/planet.py).

---

## Table of Contents

1. [State Vector](#1-state-vector)
2. [Planetary Constants](#2-planetary-constants)
3. [Orbital Mechanics — `advance_orbit`](#3-orbital-mechanics--advance_orbit)
4. [Solar Geometry — `compute_derivatives` step 1](#4-solar-geometry)
5. [dT/dt — Surface Energy Balance](#5-dtdt--surface-energy-balance)
6. [dM_ice/dt — Polar CO₂ Sublimation](#6-dm_icedt--polar-co-sublimation)
7. [dP/dt — Atmospheric Pressure](#7-dpdt--atmospheric-pressure)
8. [FAST path — `compute_fast_physics`](#8-fast-path--compute_fast_physics)
9. [Full Dependency Graph](#9-full-dependency-graph)
10. [Model Scope and Known Approximations](#10-model-scope-and-known-approximations)
11. [Citations](#11-citations)

---

## 1. State Vector

The coupled ODE system evolves three variables:

```
y = [T,  P,  M_ice]
     │    │    │
     │    │    └─ Total polar CO₂ ice mass          [kg]
     │    └────── Global mean surface pressure       [Pa]
     └─────────── Surface temperature at (φ, λ)     [K]
```

At every timestep the engine calls `advance_orbit(dt)` first (updates orbital
angle and solar flux), then either:
- `compute_derivatives(y)` → RK4 integrator (ACCURATE path)
- `compute_fast_physics(dt)` → exponential relaxation (FAST path)

---

## 2. Planetary Constants

All constants are `torch.float64` tensors defined at module level.

| Symbol | Name | Value | Source |
|--------|------|-------|--------|
| M | Mass | 6.4171×10²³ kg | NASA Fact Sheet |
| R | Radius | 3.3895×10⁶ m | NASA Fact Sheet |
| g | Surface gravity | 3.72076 m s⁻² | NASA Fact Sheet |
| T_rot | Rotation period | 88 775.244 s (1 sol) | NASA Fact Sheet |
| a | Semi-major axis | 2.27939×10¹¹ m (1.524 AU) | NASA Fact Sheet |
| e | Eccentricity | 0.0934 | NASA Fact Sheet |
| T_orb | Orbital period | 5.93568×10⁷ s (686.97 days) | NASA Fact Sheet |
| ε_tilt | Axial tilt | 25.19° = 0.4396 rad | NASA Fact Sheet |

**Atmospheric composition (default, partial pressures):**

| Species | Pressure (Pa) | Fraction |
|---------|--------------|---------|
| CO₂ | 580 | 95.1% |
| N₂ | 15 | 2.5% |
| Ar | 12 | 2.0% |
| O₂ | 0.8 | 0.1% |
| CO | 0.4 | 0.07% |

---

## 3. Orbital Mechanics — `advance_orbit`

Called at the start of every timestep by the engine. Updates two quantities:
`elapsed_time` and `orbital_angle`, then recomputes `solar_flux`.

### 3.1 Mean Motion

**Equation:**
```
dθ/dt = 2π / T_orb

→  θ(t + dt) = (θ(t)  +  2π · dt / T_orb)  mod  2π
```

**Derivation:** A planet completing one full orbit (2π rad) in time T_orb sweeps
angle at constant rate 2π/T_orb. The `mod 2π` wraps the angle back into [0, 2π)
after each complete orbit, preventing unbounded accumulation.

**Approximation:** This advances θ as the *mean anomaly* (constant angular rate)
but uses it as the *true anomaly* (actual ellipse position). In reality Kepler's
second law requires the planet to move faster at perihelion and slower at aphelion.
For Mars (e = 0.0934) this introduces a phase error of up to ±10° in solar
longitude — roughly ±5–10 sols timing error on seasonal peaks.

**Worked example (dt = 1 s):**
```
Δθ = 2π × 1 / 59 356 800
   = 6.283185 / 59 356 800
   = 1.05853×10⁻⁷ rad  =  6.065×10⁻⁶ °

After one full year (59 356 800 steps):
  θ = 59 356 800 × 1.05853×10⁻⁷ = 2π rad → mod 2π = 0  ✓
```

**Per-timestep values:**

| dt | Δθ (rad) | Δθ (degrees) |
|----|---------|-------------|
| 1 s | 1.059×10⁻⁷ | 6.07×10⁻⁶° |
| 900 s | 9.53×10⁻⁵ | 0.00546° |
| 3 600 s | 3.81×10⁻⁴ | 0.0218° |
| 88 775 s (1 sol) | 9.395×10⁻³ | 0.538° |

### 3.2 Kepler Ellipse — distance_from_sun

**Equation (Kepler's first law):**
```
r(θ) = a(1 − e²) / (1 + e · cos θ)
```

| θ | Location | r (m) | Solar flux F (W/m²) |
|---|---------|-------|-------------------|
| 0 | Perihelion | 2.066×10¹¹ | 717 |
| π | Aphelion | 2.493×10¹¹ | 492 |

Mars receives **45% more solar power at perihelion than aphelion** — the primary
driver of its strong seasonal asymmetry.

### 3.3 Inverse-Square Solar Flux

**Equation:**
```
F = F₀ × (1 AU / r)²

F₀ = 1361 W/m²  (solar constant at 1 AU)
1 AU = 1.496×10¹¹ m
```

**At perihelion (θ = 0):**
```
F = 1361 × (1.496×10¹¹ / 2.066×10¹¹)² = 1361 × 0.525 = 714 W/m²
```

**At aphelion (θ = π):**
```
F = 1361 × (1.496×10¹¹ / 2.493×10¹¹)² = 1361 × 0.360 = 490 W/m²
```

### 3.4 Data Flow

```
dt
 ├─► elapsed_time += dt          (integration clock, used for hour angle)
 │
 ├─► θ += 2π·dt/T_orb  mod 2π   (mean motion)
 │          │
 │     r = a(1−e²)/(1+e·cosθ)   (Kepler ellipse)
 │          │
 └─► F = F₀·(1AU/r)²            (written to radiation.solar_flux)
```

---

## 4. Solar Geometry

Computed inside `compute_derivatives` before dT/dt. Determines how much of the
solar flux actually hits a surface patch at latitude φ, longitude λ, time t.

### 4.1 Solar Longitude Ls

```
Ls_perihelion = 251°  (Mars convention: Ls=0° is northern spring equinox)
Ls = θ + 251°
```

Ls maps the orbital angle to the aerocentric solar longitude — the standard Mars
seasonal calendar. Ls = 251° is perihelion (deep southern summer).

### 4.2 Solar Declination δ

```
δ = arcsin(sin(ε_tilt) · sin(Ls))
```

δ is the latitude of the sub-solar point — where the Sun is directly overhead.

**At perihelion (Ls = 251°):**
```
sin(25.19°) = 0.42533
sin(251°)   = −sin(71°) = −0.94552

δ = arcsin(0.42533 × −0.94552) = arcsin(−0.40222) = −23.71°
```

Sub-solar point is 23.71° south — southern hemisphere summer. At the default
latitude 22°N this is winter — reduced insolation.

**Seasonal declination cycle:**

| Ls | Season | δ |
|----|--------|---|
| 0° | N. spring equinox | 0° |
| 90° | N. summer solstice | +25.19° |
| 180° | N. autumn equinox | 0° |
| 251° | Perihelion | −23.71° |
| 270° | N. winter solstice | −25.19° |

### 4.3 Hour Angle h

```
ω  = 2π / T_rot = 2π / 88 775.244 = 7.0792×10⁻⁵ rad/s
h  = ω · t  −  π  +  λ
```

t=0 → h = −π (midnight). The planet rotates eastward so h increases with time.
λ shifts the reference: eastern longitudes see solar noon earlier.

| t (s) | h (rad) | Local time |
|-------|---------|-----------|
| 0 | −π | Midnight |
| T_rot/4 | −π/2 | Dawn |
| T_rot/2 | 0 | **Solar noon** |
| 3T_rot/4 | +π/2 | Dusk |
| T_rot | +π → −π | Midnight |

### 4.4 Cosine of Solar Zenith Angle

```
cos(z) = max(0,  sin(φ)·sin(δ)  +  cos(φ)·cos(δ)·cos(h))
```

Clamped to zero when the Sun is below the horizon.

**Worked example — perihelion, φ = 22°N, λ = 0°:**

Pre-compute fixed terms:
```
A = sin(22°)·sin(−23.71°) = 0.37461 × (−0.40222) = −0.15062
B = cos(22°)·cos(−23.71°) = 0.92718 × 0.91558    = +0.84913

cos(z) = max(0,  −0.15062  +  0.84913 · cos(h))
```

| t (s) | h | cos(h) | cos(z) | Sun elevation |
|-------|---|--------|--------|--------------|
| 0 | −180° | −1.000 | 0 | Below horizon |
| 22 194 | −90° | 0.000 | 0 | Below horizon |
| 44 388 | 0° | +1.000 | **0.699** | **44.3°** |
| 66 581 | +90° | 0.000 | 0 | Below horizon |

**Sunrise/sunset hour angle h₀:**
```
cos(h₀) = −tan(φ)·tan(δ) = −tan(22°)·tan(−23.71°) = 0.17732
h₀ = arccos(0.17732) = 79.79°

Day length = 2 × 79.79° / 360° × 88 775 s = 39 360 s ≈ 10.93 hours
```

Winter at 22°N — only 10.93 of 24.66 hours are daylit.

---

## 5. dT/dt — Surface Energy Balance

**Root equation (First Law of Thermodynamics):**
```
C_area · dT/dt = Q_in − Q_out

→  dT/dt = (Q_in − Q_out) / C_area          [K/s]
```

### 5.1 Q_in — Absorbed Solar Radiation

```
Q_in = (1 − α) · F · cos(z)               [W/m²]
```

| Term | Value | Meaning |
|------|-------|---------|
| α | 0.25 | Bond albedo — 25% of incident light reflected |
| F | 492–717 W/m² | Solar flux (from advance_orbit) |
| cos(z) | 0–1 | Geometric projection onto surface |

**At solar noon, perihelion:**
```
Q_in = (1 − 0.25) × 714 × 0.699 = 0.75 × 714 × 0.699 = 374 W/m²
```

**At midnight:**
```
Q_in = 0  (cos(z) clamped to 0)
```

### 5.2 Q_out — Thermal Emission

```
T_eff = T / max(f_gh, 1.0)              (greenhouse reduction)
Q_out = ε · σ · T_eff⁴                  [W/m²]

ε = 0.95  (emissivity — near-blackbody in infrared)
σ = 5.670374×10⁻⁸ W m⁻² K⁻⁴
```

**Greenhouse factor f_gh = 1.02:**

Mars's atmosphere is 166× thinner than Earth's. The thin CO₂ column still
absorbs weakly in the 15 μm infrared band, trapping ~8% of outgoing radiation:

```
Q_out_bare     = ε · σ · 210⁴ = 109.4 W/m²   (no atmosphere)
Q_out_with_fgh = ε · σ · (210/1.02)⁴ = 102.6 W/m²

Trapped = 6.8 W/m²  (~4 K surface warming)
```

Comparison:
| Body | f_gh | Warming |
|------|------|---------|
| Mars | 1.02 | ~4 K |
| Earth | 1.33 | ~33 K |
| Venus | ~3.2 | ~500 K |

### 5.3 C_area — Thermal Inertia

```
C_area = ρ · c_p · d = 1.0×10⁵ J K⁻¹ m⁻²

ρ ≈ 3000 kg/m³    (basaltic regolith)
c_p ≈ 800 J/kg/K  (basalt at 200 K)
d ≈ 0.042 m       (diurnal thermal skin depth)
```

This value was calibrated to reproduce the ~205 K → ~270 K diurnal swing
observed by Curiosity REMS instrument at Gale Crater, Sol 224.

**Important scope note:** C_area is a global constant — it does not vary with
latitude or longitude. Every simulated point has the thermal properties of
Gale Crater's basaltic regolith.

### 5.4 Full Dependency Table

| Variable | Fixed or dynamic | Source |
|----------|-----------------|--------|
| α | Fixed | `radiation.albedo` (init param) |
| F | Dynamic each step | `advance_orbit` → Kepler |
| cos(z) | Dynamic each step | φ, λ, t, δ(θ) |
| ε | Fixed (0.95) | Hardcoded |
| f_gh | Fixed (1.02) | `thermal.greenhouse_factor` |
| T | State variable | ODE solution |
| C_area | Fixed (1.0×10⁵) | Calibrated to REMS Sol 224 |

---

## 6. dM_ice/dt — Polar CO₂ Sublimation

CO₂ ice caps are pinned at the **frost point T_frost = 149 K**. While ice
exists, all net radiative energy goes into phase change rather than warming.

### 6.1 Polar Insolation (Simplified)

At the geographic poles the zenith angle simplifies — the hour angle averages
out over one full sol and the mean daily insolation factor collapses to sin(δ):

```
cos(z)_N = max(0,  +sin(δ))   ← north pole sunlit only when δ > 0
cos(z)_S = max(0,  −sin(δ))   ← south pole sunlit only when δ < 0
```

**Seasonal behaviour:**

| Ls | δ | cos(z)_N | cos(z)_S | Which cap sublimates |
|----|---|---------|---------|---------------------|
| 90° (N. summer) | +25.19° | 0.425 | 0 | North |
| 180° (equinox) | 0° | 0 | 0 | Neither |
| 251° (perihelion) | −23.71° | 0 | 0.402 | South |
| 270° (N. winter) | −25.19° | 0 | 0.425 | South (peak) |

### 6.2 Energy Balance at Frost Point

```
Q_in_pole  = (1 − α) · F · cos(z)_pole       [W/m²]
Q_out_pole = ε · σ · T_frost⁴
           = 0.95 × 5.670374×10⁻⁸ × 149⁴
           = 26.51 W/m²                        (fixed — T pinned at frost point)

ΔQ = Q_in_pole − Q_out_pole
```

ΔQ > 0 → ice **sublimates** (gains energy, turns to gas)
ΔQ < 0 → CO₂ **condenses** (loses energy, freezes from atmosphere)

### 6.3 Mass Rate via Latent Heat

```
A_cap = 0.03 × 4π R²  =  0.03 × 4π × (3.3895×10⁶)²  =  4.328×10¹² m²

L_sub = 5.7×10⁵ J/kg  (CO₂ latent heat of sublimation)

net_sub_N = ΔQ_N × A_cap / L_sub      [kg/s]
net_sub_S = ΔQ_S × A_cap / L_sub      [kg/s]

dM_ice/dt = −(net_sub_N + net_sub_S)
```

Negative sign: positive ΔQ means sublimation = ice is being lost.

**Guard condition:** if M_ice ≤ 0 and dM_ice/dt < 0, clamp to zero.
Cannot sublimate ice that does not exist.

### 6.4 Worked Example — Perihelion (Ls = 251°, south cap peak)

```
δ = −23.71°
cos(z)_S = max(0, −sin(−23.71°)) = max(0, 0.40222) = 0.40222
cos(z)_N = max(0, sin(−23.71°))  = 0

F = 714 W/m²  (perihelion)
Q_in_S = (1 − 0.25) × 714 × 0.40222 = 214.9 W/m²
Q_out_pole = 26.51 W/m²
ΔQ_S = 214.9 − 26.51 = 188.4 W/m²

net_sub_S = 188.4 × 4.328×10¹² / 5.7×10⁵ = 1.431×10⁹ kg/s

dM_ice/dt ≈ −1.431×10⁹ kg/s  (south cap losing ~1.4×10⁹ kg every second)
```

Over one sol (88 775 s):
```
ΔM_ice ≈ −1.431×10⁹ × 88 775 = −1.27×10¹⁴ kg/sol
```

Starting from M_ice = 5×10¹⁵ kg, the south cap would fully sublimate in
~39 sols at peak perihelion insolation (the real cap persists because ΔQ
varies strongly across the orbit).

---

## 7. dP/dt — Atmospheric Pressure

Two contributions to global mean surface pressure:

```
dP/dt = dP_escape  +  dP_sublimation
```

### 7.1 Ice–Atmosphere Mass Exchange (dominant)

Hydrostatic equilibrium gives:
```
P = M_atm · g / A_planet

Conservation:  M_atm + M_ice = constant
→  dM_atm/dt = −dM_ice/dt

dP_sublimation = −dM_ice/dt · g / A_planet
```

**Signs:**
```
Sublimation (M_ice decreasing, dM_ice/dt < 0):
  −dM_ice/dt > 0  →  dP > 0  (pressure rises as CO₂ enters atmosphere) ✓

Condensation (M_ice increasing, dM_ice/dt > 0):
  −dM_ice/dt < 0  →  dP < 0  (pressure falls as CO₂ leaves atmosphere) ✓
```

**Real Mars observations (Viking landers):**

| Season | P (Pa) | Driver |
|--------|--------|--------|
| N. winter / S. summer (perihelion) | ~740 | South cap sublimating |
| N. summer / S. winter | ~560 | South cap condensing |
| Swing | ~25% | Entirely from dM_ice/dt |

### 7.2 Non-thermal Atmospheric Escape (secular)

Mars loses atmosphere primarily via **non-thermal** mechanisms — solar wind
sputtering and photochemical escape — measured by the MAVEN spacecraft.
These processes are independent of surface temperature and pressure.

```
MAVEN_ESCAPE_RATE = 0.2 kg/s   (empirical, Jakosky et al. 2018)

dP_escape = −0.2 · g / A_planet
          = −0.2 × 3.72076 / (4π × (3.3895×10⁶)²)
          = −0.2 × 3.72076 / 1.448×10¹⁴
          = −5.14×10⁻¹⁵ Pa/s
```

**Per sol:**
```
ΔP_escape = 5.14×10⁻¹⁵ × 88 775 = 4.56×10⁻¹⁰ Pa/sol
```

Completely negligible vs the ~180 Pa seasonal swing. Becomes significant only
on geological timescales (millions of years).

**Why not Jeans escape?**

The original code used thermal (Jeans) escape. For CO₂ at T = 210 K, the Jeans
parameter is:
```
λ = G·M·m_CO₂ / (k_B·T·R_exo) ≈ 299

exp(−λ) = exp(−299) ≈ 10⁻¹³⁰   ← underflows to zero in float64
```

CO₂ molecules are too massive to escape Mars thermally. Jeans escape is the
correct mechanism for hydrogen and helium, not CO₂. The empirical MAVEN constant
correctly represents the actual non-thermal mechanisms without creating a false
T → P coupling.

---

## 8. FAST Path — `compute_fast_physics`

An alternative to full RK4 integration for large timesteps. Computes the same
physics analytically via equilibrium + relaxation.

### 8.1 Daily-Mean Insolation Factor

Instead of instantaneous cos(z), computes the daily average analytically:

```
cos(h₀) = clamp(−tan(φ) · tan(δ), −1, 1)
h₀ = arccos(cos(h₀))                        (sunrise hour angle)

insolation_factor = (h₀·sin(φ)·sin(δ)  +  cos(φ)·cos(δ)·sin(h₀)) / π
```

This is the closed-form integral of cos(z) over one full sol.

### 8.2 Radiative Equilibrium Temperature

```
T_eq_base = [(1−α) · F · insolation_factor / (ε · σ)]^(1/4)
T_eq = T_eq_base × f_gh
```

The surface equilibrium temperature — where Q_in = Q_out if the surface had
infinite time to adjust.

**Diurnal swing superimposed:**
```
swing = 50 · cos(φ)     (amplitude drops to zero at poles)
T_eq → T_eq − swing · cos(ω·t + λ)
```

### 8.3 Exponential Relaxation (Newtonian Cooling)

```
τ = C_area / (4 · ε · σ · T³)      (thermal inertia timescale)
T(t+dt) = T_eq + (T − T_eq) · exp(−dt / τ)
```

τ is derived by linearising Q_out = ε·σ·T⁴ around the current temperature:

```
∂Q_out/∂T = 4·ε·σ·T³

τ = C_area / (∂Q_out/∂T) = C_area / (4·ε·σ·T³)
```

**At T = 210 K:**
```
τ = 1.0×10⁵ / (4 × 0.95 × 5.670×10⁻⁸ × 210³)
  = 1.0×10⁵ / (4 × 0.95 × 5.670×10⁻⁸ × 9.261×10⁶)
  = 1.0×10⁵ / 1.992
  = 50 200 s  ≈  0.57 sols
```

The surface relaxes toward equilibrium with a timescale of about half a sol.

### 8.4 Where the Efficiency Actually Comes From

It is **not** fewer ops per call. It is **timestep size**:

| Constraint | ACCURATE (RK4) | FAST (relaxation) |
|-----------|---------------|------------------|
| Stability | Requires dt ≪ τ | Unconditionally stable for any dt |
| Diurnal resolution | Needs dt < T_rot/20 ≈ 4,400 s | Can skip diurnal entirely |
| Min practical dt | ~900–3,600 s | ~1 sol (88,775 s) or larger |
| Calls per step | 4 × compute_derivatives | 1 × compute_fast_physics |

**Steps to simulate 100 Martian years:**

| Mode | dt | Steps | Derivative evals |
|------|-----|-------|-----------------|
| ACCURATE | 3,600 s | 1,648,800 | **6.6M** |
| FAST | 1 sol | 66,860 | 66,860 |
| FAST | 10 sols | 6,686 | 6,686 |
| FAST | 1 year | 100 | 100 |

For century-scale simulations the FAST path is **100–66,000× cheaper**.

**What accuracy is traded away:**

| | ACCURATE | FAST |
|--|---------|------|
| Insolation | Instantaneous cos(z) | Daily-mean insolation factor |
| Diurnal signal | Emerges from ODE | Synthetic 50K·cos(φ) hardcoded |
| Temperature stepping | RK4 (4th-order) | Exact exponential (1st-order linear) |
| Ice and pressure | RK4 sub-steps | First-order Euler |

**When to use each:**

```
Use ACCURATE when:                Use FAST when:
  - dt < 1 sol                     - dt ≥ 1 sol
  - Studying diurnal cycle          - Studying seasonal cycle
  - Validating against REMS data    - Terraforming trajectories
  - Physics accuracy critical       - Many-year runs
```

### 8.5 Dependency Graph — `compute_fast_physics`

```
advance_orbit(dt)  [called by engine before fast physics]
└── orbital_angle, elapsed_time, radiation.solar_flux  ← prerequisites


compute_fast_physics(dt)
│
├── [Step 1: Daily-mean insolation]
│       │
│       ├── Ls = orbital_angle + 251°
│       ├── δ  = arcsin(sin(ε_tilt) · sin(Ls))
│       ├── cos_h0 = clamp(−tan(φ) · tan(δ), −1, 1)
│       ├── h0 = arccos(cos_h0)                          (sunrise hour angle)
│       └── insolation_factor = (h0·sin(φ)·sin(δ) + cos(φ)·cos(δ)·sin(h0)) / π
│                                      │                        │
│                                      φ = _init_latitude (fixed)
│
├── [Step 2: Equilibrium temperature]
│       │
│       ├── absorbed = (1−α) · F · insolation_factor
│       │                   │   │         │
│       │                   │   │    ← Step 1 output
│       │                   │   └── radiation.solar_flux  ← advance_orbit
│       │                   └── radiation.albedo  (fixed)
│       │
│       ├── T_eq_base = (absorbed / (ε·σ))^0.25
│       ├── T_eq      = T_eq_base × greenhouse_factor     (greenhouse scaling)
│       │
│       └── T_eq -= 50·cos(φ) · cos(ω·t + λ)             (synthetic diurnal swing)
│                        │              │   │
│                        φ (fixed)      │   λ = _init_longitude (fixed)
│                                 elapsed_time
│
├── [Step 3: Exponential relaxation → thermal.surface_temperature]
│       │
│       ├── τ = C_area / (4·ε·σ·T_cur³)                  (linearised timescale)
│       │              │              │
│       │         1.0×10⁵         T_cur = current surface_temperature
│       │
│       └── T_new = T_eq + (T_cur − T_eq) · exp(−dt/τ)   (exact ODE solution)
│                    │                              │
│               ← Step 2                           dt
│
├── [Step 4: Ice mass → water.ice_mass]
│       │
│       ├── cos_zenith_N = max(0, +sin(δ))   ← Step 1 δ
│       ├── cos_zenith_S = max(0, −sin(δ))
│       ├── Q_in_N/S = (1−α) · F · cos_zenith_N/S
│       ├── Q_out_pole = ε·σ·149⁴ = 26.51 W/m²   (fixed)
│       ├── dMice = −(net_sub_N + net_sub_S) · dt         (Euler step)
│       └── Guard: clamp if ice_mass=0 and dMice<0
│
└── [Step 5: Pressure → atmosphere.surface_pressure]
        │
        ├── dP_escape      = −0.2 · g / (4πR²) · dt       (MAVEN constant)
        └── dP_sublimation = −dMice · g / (4πR²)          ← Step 4 dMice
```

**Key structural difference from `compute_derivatives`:**

```
compute_derivatives                    compute_fast_physics
────────────────────────────           ────────────────────────────
cos(z) instantaneous at time t         insolation_factor = ∫cos(z)dh / π
T drives dT/dt directly                T_eq is target, T relaxes toward it
h = ω·t−π+λ used explicitly           h averaged out analytically via h0
Called 4× per RK4 step                 Called 1× per step, any dt
```

```
advance_orbit(dt)
├── elapsed_time += dt
└── orbital_angle += 2π·dt/T_orb  mod 2π
        │
        ├── r = a(1−e²)/(1+e·cosθ)
        └── F = F₀·(1AU/r)²  →  radiation.solar_flux


compute_derivatives(y = [T, P, M_ice])
│
├── [dT/dt]  = (Q_in − Q_out) / C_area
│       │
│       ├── Q_in = (1−α)·F·cos(z)
│       │           │   │    │
│       │           │   │    └── cos(z) = max(0, sin(φ)sin(δ) + cos(φ)cos(δ)cos(h))
│       │           │   │                          │      │               │      │
│       │           │   │                          φ      δ               φ      h = ω·t−π+λ
│       │           │   │                                 │                      │
│       │           │   │              δ = arcsin(sin(ε)·sin(Ls))         elapsed_time
│       │           │   │                                    │
│       │           │   │                         Ls = θ + 251°  ←  orbital_angle
│       │           │   │
│       │           │   └── F = radiation.solar_flux  ← advance_orbit
│       │           └── α = radiation.albedo  (fixed)
│       │
│       ├── Q_out = ε·σ·(T/f_gh)⁴
│       │                  │   │
│       │                  T   f_gh = thermal.greenhouse_factor  (fixed 1.02)
│       │
│       └── C_area = 1.0×10⁵  (fixed, calibrated to REMS Sol 224)
│
│
├── [dM_ice/dt]  = −(net_sub_N + net_sub_S)
│       │
│       ├── net_sub = (Q_in_pole − Q_out_pole) · A_cap / L_sub
│       │                │              │          │       │
│       │                │         ε·σ·149⁴       │    5.7×10⁵ J/kg
│       │                │         = 26.51 W/m²   │
│       │                │                    0.03·4πR²
│       │         (1−α)·F·max(0, ±sin(δ))
│       │                    │         │
│       │                    F         δ  ← Ls ← orbital_angle
│       │
│       └── Guard: clamp if M_ice=0 and sublimating
│
│
└── [dP/dt]  = dP_escape + dP_sublimation
        │               │
        │       −dM_ice/dt · g / (4πR²)
        │               │         │
        │          dM_ice/dt    g, R  (fixed intrinsic params)
        │
        └── −0.2 kg/s · g / (4πR²)
                         │
                    MAVEN empirical constant  (Jakosky et al. 2018)
                    independent of T and P
```

### ODE coupling summary

| Equation | Reads from other equations? |
|----------|-----------------------------|
| dT/dt | No — depends on orbital state and own T |
| dM_ice/dt | No — depends on orbital state only |
| dP/dt | **Yes** — depends on dM_ice/dt |

Only one coupling: ice sublimation feeds directly into pressure.
Temperature no longer feeds into pressure (Jeans escape removed).

---

## 10. Model Scope and Known Approximations

| Approximation | Impact | Location |
|---------------|--------|----------|
| Mean anomaly used as true anomaly | ±10° Ls error, ±5–10 sol timing offset on seasonal peaks | `advance_orbit` |
| C_area constant everywhere | All surface points have Gale Crater's thermal inertia regardless of actual location | `compute_derivatives` line 235 |
| Default latitude 22°N not Gale Crater | Solar geometry computed for 22°N but thermal calibration from 4.5°S | `__init__` |
| Polar cos(z) = max(0, ±sin(δ)) | Exact at 90° poles, approximate at cap edges | `compute_derivatives` lines 250–251 |
| A_cap = 3% per pole | Fixed cap area, no seasonal cap extent evolution | lines 247, 352 |
| 0D global pressure | No spatial pressure gradient or local weather | dP/dt formulation |
| f_gh = 1.02 fixed | Greenhouse does not evolve with pressure changes | `thermal.greenhouse_factor` |
| MAVEN escape rate constant | Does not vary with solar activity or season | lines 274, 375 |

---



## 11. Citations

| Reference | Used for |
|-----------|---------|
| NASA Mars Fact Sheet — https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html | All planetary constants |
| Kepler, J. (1609). *Astronomia Nova* | Orbital ellipse r(θ) = a(1−e²)/(1+e·cosθ) |
| Lambert, J.H. (1760). *Photometria* | Lambert's cosine law — Q = F·cos(z) |
| Stefan, J. (1879); Boltzmann, L. (1884) | Thermal emission Q = ε·σ·T⁴ |
| Clausius, R. (1850). First Law of Thermodynamics | dU/dt = Q_in − Q_out |
| Jakosky, B.M. et al. (2018). *Science*, 355(6323). doi:10.1126/science.aan5015 | MAVEN non-thermal escape rate ~0.2 kg/s |
| Kieffer, H.H. et al. (1977). *JGR*, 82(28). | CO₂ frost point 149 K at Mars surface pressure |
| Smith, M.D. (2004). *Icarus*, 167(1), 148–165 | CO₂ latent heat L_sub = 5.7×10⁵ J/kg |
| Haberle, R.M. et al. (1993). *JGR*, 98(E2) | Seasonal pressure swing driven by polar cap exchange |
| Vasavada, A.R. et al. (2017). *JGR Planets*, 122(5) | REMS Gale Crater temperature profiles, Sol 224 calibration |
| Wikipedia — Solar zenith angle — https://en.wikipedia.org/wiki/Solar_zenith_angle | cos(z) spherical geometry formula |
