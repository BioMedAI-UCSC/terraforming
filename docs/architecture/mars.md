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
- `compute_derivatives(y)` → [RK4 integrator](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) (ACCURATE path)
- `compute_fast_physics(dt)` → [exponential relaxation](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) (FAST path)

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

**Physics calibration constants** (tuned to REMS observations):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MARS_THERMAL_INERTIA` | **6.0×10⁴ J K⁻¹ m⁻²** | Surface thermal mass per unit area — controls diurnal temperature amplitude |
| `MARS_POLAR_CAP_FRACTION` | **0.01** | Effective fractional surface area of each sublimating polar cap |
| `MARS_THERMAL_TIDE_PA` | **30.0 Pa** | Half-amplitude of empirical diurnal pressure oscillation |
| `MARS_THERMAL_TIDE_PHASE` | **−0.7π rad** | Phase offset — puts pressure max at ~08:37 LMST |
| `MARS_CO2_FROST_POINT` | 149.0 K | CO₂ condensation/sublimation temperature at Mars surface pressure |
| `MARS_CO2_LATENT_HEAT` | 5.7×10⁵ J kg⁻¹ | Latent heat of CO₂ sublimation |
| `MARS_MAVEN_ESCAPE_RATE` | 0.2 kg s⁻¹ | Non-thermal atmospheric escape (Jakosky et al. 2018) |
| `MARS_SURFACE_EMISSIVITY` | 0.95 | Near-blackbody IR emissivity of basaltic regolith |

---

## 3. Orbital Mechanics — `advance_orbit`

Called at the start of every timestep by the engine. Updates two quantities:
`elapsed_time` and `orbital_angle`, then recomputes `solar_flux`.

### 3.1 Mean Motion

> **Laws applied:** [Kepler's Second Law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Second_law) · [Mean motion](https://en.wikipedia.org/wiki/Mean_motion) · [Mean anomaly](https://en.wikipedia.org/wiki/Mean_anomaly)

**Equation:**
```
dθ/dt = 2π / T_orb

→  θ(t + dt) = (θ(t)  +  2π · dt / T_orb)  mod  2π
```

**Derivation:** A planet completing one full orbit (2π rad) in time T_orb sweeps
angle at constant rate 2π/T_orb. The `mod 2π` wraps the angle back into [0, 2π)
after each complete orbit, preventing unbounded accumulation.

**Approximation:** This advances θ as the *[mean anomaly](https://en.wikipedia.org/wiki/Mean_anomaly)* (constant angular rate)
but uses it as the *[true anomaly](https://en.wikipedia.org/wiki/True_anomaly)* (actual ellipse position). In reality
[Kepler's second law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Second_law) requires the planet to move faster at perihelion and slower at aphelion.
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

> **Laws applied:** [Kepler's First Law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#First_law) · [Elliptic orbit](https://en.wikipedia.org/wiki/Elliptic_orbit) · [Orbital eccentricity](https://en.wikipedia.org/wiki/Orbital_eccentricity)

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

> **Laws applied:** [Inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law) · [Solar irradiance](https://en.wikipedia.org/wiki/Solar_irradiance) · [Solar constant](https://en.wikipedia.org/wiki/Solar_constant)

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

> **Reference:** [Solar longitude](https://en.wikipedia.org/wiki/Solar_longitude) · [Areocentric coordinates](https://en.wikipedia.org/wiki/Mars#Seasons)

```
Ls_perihelion = 251°  (Mars convention: Ls=0° is northern spring equinox)
Ls = θ + 251°
```

Ls maps the orbital angle to the aerocentric solar longitude — the standard Mars
seasonal calendar. Ls = 251° is perihelion (deep southern summer).

### 4.2 Solar Declination δ

> **Laws applied:** [Position of the Sun](https://en.wikipedia.org/wiki/Position_of_the_Sun) · [Solar declination](https://en.wikipedia.org/wiki/Position_of_the_Sun#Declination_of_the_Sun_as_seen_from_Earth) · [Axial tilt](https://en.wikipedia.org/wiki/Axial_tilt)

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

> **Reference:** [Hour angle](https://en.wikipedia.org/wiki/Hour_angle) · [Solar time](https://en.wikipedia.org/wiki/Solar_time)

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

> **Laws applied:** [Solar zenith angle](https://en.wikipedia.org/wiki/Solar_zenith_angle) · [Spherical trigonometry](https://en.wikipedia.org/wiki/Spherical_trigonometry)

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

> **Laws applied:** [First Law of Thermodynamics](https://en.wikipedia.org/wiki/First_law_of_thermodynamics) · [Radiative energy balance](https://en.wikipedia.org/wiki/Earth%27s_energy_budget) · [Ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation)

**Root equation (First Law of Thermodynamics):**
```
C_area · dT/dt = Q_in − Q_out

→  dT/dt = (Q_in − Q_out) / C_area          [K/s]
```

### 5.1 Q_in — Absorbed Solar Radiation

> **Laws applied:** [Lambert's cosine law](https://en.wikipedia.org/wiki/Lambert%27s_cosine_law) · [Albedo](https://en.wikipedia.org/wiki/Albedo)

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

> **Laws applied:** [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) · [Greenhouse effect](https://en.wikipedia.org/wiki/Greenhouse_effect) · [Thermal radiation](https://en.wikipedia.org/wiki/Thermal_radiation)

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

> **Reference:** [Thermal inertia](https://en.wikipedia.org/wiki/Thermal_inertia) · [Thermal skin depth](https://en.wikipedia.org/wiki/Thermal_contact_conductance) · [THEMIS instrument](https://en.wikipedia.org/wiki/2001_Mars_Odyssey#THEMIS)

```
C_area = MARS_THERMAL_INERTIA = 6.0×10⁴ J K⁻¹ m⁻²

Relation to physical properties:
  C_area = ρ · c_p · d

  ρ ≈ 2000–3000 kg/m³   (rocky/sandy regolith, Gale Crater)
  c_p ≈ 800 J/kg/K      (basalt at 200 K)
  d ≈ 0.025–0.037 m     (diurnal thermal skin depth)
```

**Why this value:** THEMIS and TES orbital measurements at Gale Crater give
thermal inertia TI ≈ 200–350 TIU (J m⁻² K⁻¹ s⁻⁰·⁵). Converting:
C_area = TI × √(T_rot/π) ≈ TI × √(88775/π) ≈ TI × 168 → range 3.4–5.9×10⁴.
The value 6.0×10⁴ sits at the upper end, consistent with the rockier sections
of Gale Crater visible in REMS diurnal profiles.

**Calibration history:**

| C_area | Diurnal swing (model) | REMS Sol 224 |
|--------|----------------------|--------------|
| 2.0×10⁴ (old — loose dust) | ~180 K | ~65 K |
| 6.0×10⁴ (current — rocky/sandy) | ~60 K | ~65 K ✓ |

A **3× increase in C_area** → **3× smaller dT/dt** at the same Q_in−Q_out,
producing a 3× narrower diurnal temperature swing.

**Why C_area is the denominator, not a multiplier:**

```
dT/dt = (Q_in − Q_out) / C_area

A slab with more thermal mass (higher C_area) stores more energy
per degree — so the same net flux produces a smaller temperature change.
```

**Important scope note:** C_area is a global constant — every simulated
location has Gale Crater's rocky-sandy thermal properties regardless of
actual surface type.

### 5.4 Full Dependency Table

| Variable | Fixed or dynamic | Source |
|----------|-----------------|--------|
| α | Fixed | `radiation.albedo` (init param) |
| F | Dynamic each step | `advance_orbit` → Kepler |
| cos(z) | Dynamic each step | φ, λ, t, δ(θ) |
| ε | Fixed (0.95) | Hardcoded |
| f_gh | Fixed (1.02) | `thermal.greenhouse_factor` |
| T | State variable | ODE solution |
| C_area | Fixed (**6.0×10⁴**) | Calibrated to REMS Sol 224 Gale Crater |

---

## 6. dM_ice/dt — Polar CO₂ Sublimation

> **Laws applied:** [Latent heat](https://en.wikipedia.org/wiki/Latent_heat) · [Phase transition](https://en.wikipedia.org/wiki/Phase_transition) · [Sublimation](https://en.wikipedia.org/wiki/Sublimation_(phase_transition)) · [Frost point](https://en.wikipedia.org/wiki/Dew_point#Frost_point)

CO₂ ice caps are pinned at the **frost point T_frost = 149 K**. While ice
exists, all net radiative energy goes into phase change rather than warming.

### 6.1 Polar Insolation (Simplified)

> **Reference:** [Polar night](https://en.wikipedia.org/wiki/Polar_night) · [Midnight sun](https://en.wikipedia.org/wiki/Midnight_sun) · [Insolation](https://en.wikipedia.org/wiki/Solar_irradiance#Insolation)

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

> **Laws applied:** [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) · [Latent heat of sublimation](https://en.wikipedia.org/wiki/Enthalpy_of_sublimation) · [Energy balance](https://en.wikipedia.org/wiki/Earth%27s_energy_budget)

```
Q_in_pole  = (1 − α) · F · cos(z)_pole       [W/m²]
Q_out_pole = ε · σ · T_frost⁴
           = 0.95 × 5.670374×10⁻⁸ × 149⁴
           = 26.51 W/m²                        (fixed — T pinned at frost point)

ΔQ = Q_in_pole − Q_out_pole
```

ΔQ > 0 → ice **sublimates** (gains energy, turns to gas)
ΔQ < 0 → CO₂ **condenses** (loses energy, freezes from atmosphere)

### 6.3 Two-Pole Ice Budget with Per-Pole Reservoirs

The model tracks north and south caps **independently**, each with its own ice
reservoir. This is necessary because the north cap depletes in northern summer
(Ls 0–180°) while the south cap persists year-round.

**Initial ice split (set in `setup_properties`):**

```python
if 0° ≤ Ls < 180°:          # Northern summer — north CO₂ already sublimated
    f_north = 0.0
else:                        # Northern winter — north cap growing
    f_north = 0.4 × (Ls − 180°) / 180°   # linearly 0→0.4

f_south = 1.0 − f_north
```

At Ls = 287° (REMS Sol 224): f_north = 0.4×(287−180)/180 = 0.238, f_south = 0.762
— north cap growing in northern winter, south cap at near-peak sublimation.

**Mass rate equations:**

```
A_cap = 0.01 × 4π R²  =  0.01 × 4π × (3.3895×10⁶)²  =  1.443×10¹² m²

L_sub = 5.7×10⁵ J/kg  (CO₂ latent heat of sublimation)

net_sub_N = ΔQ_N × A_cap / L_sub      [kg/s]
net_sub_S = ΔQ_S × A_cap / L_sub      [kg/s]

dM_ice_N/dt = −net_sub_N
dM_ice_S/dt = −net_sub_S
dM_ice/dt   = dM_ice_N/dt + dM_ice_S/dt
```

**Calibration: why 0.01 not 0.03?**

At 0.03 (old value), peak sublimation was ~1.4×10⁹ kg/s, depleting 5×10¹⁵ kg
in ~39 sols — too fast. At 0.01, the peak rate drops to ~4.8×10⁸ kg/s, giving
~120 sol depletion — consistent with the observed seasonal CO₂ cycle timescale.

**Per-pole guard conditions:**

```
Sublimation (dMice_N < 0) blocked if ice_mass_north = 0   ← can't sublimate nothing
Sublimation (dMice_S < 0) blocked if ice_mass_south = 0
Condensation always allowed (CO₂ can always refreeze)
```

### 6.4 Worked Example — Perihelion (Ls = 251°, south cap peak)

```
δ = −23.71°
cos(z)_S = max(0, −sin(−23.71°)) = max(0, 0.40222) = 0.40222
cos(z)_N = max(0, sin(−23.71°))  = 0

F = 714 W/m²  (perihelion)
Q_in_S = (1 − 0.25) × 714 × 0.40222 = 214.9 W/m²
Q_out_pole = 26.51 W/m²
ΔQ_S = 214.9 − 26.51 = 188.4 W/m²

A_cap = 0.01 × 4π × (3.3895×10⁶)² = 1.443×10¹² m²

net_sub_S = 188.4 × 1.443×10¹² / 5.7×10⁵ = 4.77×10⁸ kg/s

dM_ice/dt ≈ −4.77×10⁸ kg/s
```

Over one sol (88 775 s):
```
ΔM_ice ≈ −4.77×10⁸ × 88 775 = −4.23×10¹³ kg/sol
```

Starting from M_ice = 5×10¹⁵ kg, the south cap would fully sublimate in
~118 sols at peak perihelion insolation.

---

## 7. dP/dt — Atmospheric Pressure

> **Laws applied:** [Hydrostatic equilibrium](https://en.wikipedia.org/wiki/Hydrostatic_equilibrium) · [Atmospheric pressure](https://en.wikipedia.org/wiki/Atmospheric_pressure) · [Conservation of mass](https://en.wikipedia.org/wiki/Conservation_of_mass)

Three contributions to global mean surface pressure:

```
dP/dt = dP_escape  +  dP_sublimation  +  dP_tide
```

### 7.1 Ice–Atmosphere Mass Exchange (dominant)

> **Laws applied:** [Hydrostatic equilibrium](https://en.wikipedia.org/wiki/Hydrostatic_equilibrium) · [Ideal gas law](https://en.wikipedia.org/wiki/Ideal_gas_law) · [Barometric formula](https://en.wikipedia.org/wiki/Barometric_formula)

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

> **Reference:** [Atmospheric escape](https://en.wikipedia.org/wiki/Atmospheric_escape) · [Sputtering](https://en.wikipedia.org/wiki/Sputtering) · [Photochemical escape](https://en.wikipedia.org/wiki/Atmospheric_escape#Non-thermal_escape) · [MAVEN mission](https://en.wikipedia.org/wiki/MAVEN)

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

### 7.3 Thermal Tide — Empirical Diurnal Pressure Oscillation

> **Reference:** [Atmospheric tide](https://en.wikipedia.org/wiki/Atmospheric_tide) · [Diurnal cycle](https://en.wikipedia.org/wiki/Diurnal_cycle) · [Harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator)

Real Mars exhibits a ~40–60 Pa diurnal pressure wave at Gale Crater (clearly
visible in REMS data). This **atmospheric thermal tide** is driven by
differential solar heating of the global atmosphere — a wave propagating around
the planet each sol. It cannot emerge from a 1D single-column model.

The parameterisation adds its time-derivative directly to dP/dt:

```
dP_tide/dt = −A × ω × sin(ω·t + φ)

A = MARS_THERMAL_TIDE_PA    = 30.0 Pa
φ = MARS_THERMAL_TIDE_PHASE = −0.7π rad
ω = 2π / T_rot              = 7.0792×10⁻⁵ rad/s
```

This is the analytical derivative of `P_tide(t) = A · cos(ω·t + φ)`, giving
pressure a **zero-mean sinusoidal oscillation of amplitude 30 Pa**.

**Phase derivation — when is pressure maximum?**

P_tide is maximum when cos(ω·t + φ) = 1:
```
ω·t + φ = 0
ω·t = −φ = 0.7π
t = 0.7π / ω = 0.35 × T_rot = 0.35 × 88 775 = 31 071 s after midnight
  = 08:37 LMST  ✓  (matches REMS Sol 224 morning pressure peak)
```

P_tide is minimum when cos(ω·t + φ) = −1:
```
ω·t = π − φ = 1.7π
t = 0.85 × T_rot = 75 459 s = 20:57 LMST  ✓  (REMS evening pressure trough)
```

**Zero mean — no secular drift:**

```
∫₀^T P_tide dt = ∫₀^T A·cos(ω·t+φ) dt = 0

Over any integer number of sols, the tide adds/removes exactly equal amounts
of pressure — it does not bias the long-term seasonal trend.
```

**Amplitude note:** The real REMS tide is ~40–60 Pa peak-to-peak; A = 30 Pa
(half-amplitude) gives 60 Pa peak-to-peak, consistent with clear-day REMS data.

**Why not Jeans escape?**

> **Reference:** [Jeans escape](https://en.wikipedia.org/wiki/Atmospheric_escape#Jeans_escape) · [Maxwell-Boltzmann distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution)

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

> **Methods applied:** [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) · [Stiff ODE](https://en.wikipedia.org/wiki/Stiff_equation) · [Newtonian cooling](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) · [Euler method](https://en.wikipedia.org/wiki/Euler_method)

An alternative to full RK4 integration for large timesteps. Computes the same
physics analytically via equilibrium + relaxation.

### 8.1 Daily-Mean Insolation Factor

> **Reference:** [Insolation](https://en.wikipedia.org/wiki/Solar_irradiance#Insolation) · [Sunrise equation](https://en.wikipedia.org/wiki/Sunrise_equation) · [Definite integral](https://en.wikipedia.org/wiki/Integral)

Instead of instantaneous cos(z), computes the daily average analytically:

```
cos(h₀) = clamp(−tan(φ) · tan(δ), −1, 1)
h₀ = arccos(cos(h₀))                        (sunrise hour angle)

insolation_factor = (h₀·sin(φ)·sin(δ)  +  cos(φ)·cos(δ)·sin(h₀)) / π
```

This is the closed-form integral of cos(z) over one full sol.

### 8.2 Radiative Equilibrium Temperature

> **Laws applied:** [Radiative equilibrium](https://en.wikipedia.org/wiki/Radiative_equilibrium) · [Effective temperature](https://en.wikipedia.org/wiki/Effective_temperature) · [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)

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

> **Methods applied:** [Newton's law of cooling](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) · [Linearisation](https://en.wikipedia.org/wiki/Linearization) · [Exponential decay](https://en.wikipedia.org/wiki/Exponential_decay) · [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) (first-order)

```
τ = C_area / (4 · ε · σ · T³)      (thermal inertia timescale)
T(t+dt) = T_eq + (T − T_eq) · exp(−dt / τ)
```

τ is derived by linearising Q_out = ε·σ·T⁴ around the current temperature
(first-order Taylor expansion of the Stefan-Boltzmann term):

```
∂Q_out/∂T = 4·ε·σ·T³

τ = C_area / (∂Q_out/∂T) = C_area / (4·ε·σ·T³)
```

The solution `T(t) = T_eq + (T₀ − T_eq)·exp(−t/τ)` is the exact analytical
solution to the linearised ODE `dT/dt = −(T − T_eq)/τ`, making the FAST path
unconditionally stable for any dt — unlike RK4 which requires dt ≪ τ.

**At T = 210 K (C_area = 6.0×10⁴):**
```
τ = 6.0×10⁴ / (4 × 0.95 × 5.670×10⁻⁸ × 210³)
  = 6.0×10⁴ / (4 × 0.95 × 5.670×10⁻⁸ × 9.261×10⁶)
  = 6.0×10⁴ / 1.992
  = 30 120 s  ≈  0.34 sols
```

The surface relaxes toward equilibrium with a timescale of about one-third of a sol.

*(Old value with C_area = 2.0×10⁴ gave τ ≈ 10 040 s ≈ 0.11 sols — too fast,
over-amplifying the diurnal swing. The current 6.0×10⁴ gives a more realistic
damping rate.)*

### 8.4 RK4 vs Relaxation — Numerical Integration Trade-offs

> **Methods applied:** [Runge-Kutta 4th order](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) · [Numerical stability](https://en.wikipedia.org/wiki/Numerical_stability) · [Stiff equation](https://en.wikipedia.org/wiki/Stiff_equation) · [Euler method](https://en.wikipedia.org/wiki/Euler_method) (used for ice/pressure in FAST path)

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
├── [Step 4: Ice mass → water.ice_mass_north/south]
│       │
│       ├── cos_zenith_N = max(0, +sin(δ))   ← Step 1 δ
│       ├── cos_zenith_S = max(0, −sin(δ))
│       ├── Q_in_N/S = (1−α) · F · cos_zenith_N/S
│       ├── Q_out_pole = ε·σ·149⁴ = 26.51 W/m²   (fixed)
│       ├── dMice_N/S = −net_sub_N/S · dt         (per-pole Euler step)
│       ├── Guard: clamp each pole independently if that pole's ice = 0
│       └── ice_mass = ice_mass_north + ice_mass_south
│
└── [Step 5: Pressure → atmosphere.surface_pressure]
        │
        ├── dP_escape      = −0.2 · g / (4πR²) · dt       (MAVEN constant)
        ├── dP_sublimation = −dMice · g / (4πR²)          ← Step 4 dMice
        └── dP_tide        = −A·ω·sin(ω·t + φ) · dt       (thermal tide)
                                │    │           │
                                30  7.08×10⁻⁵  −0.7π    elapsed_time
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
│       └── C_area = 6.0×10⁴  (fixed, calibrated to REMS Sol 224 Gale Crater)
│
│
├── [dM_ice/dt]  = dMice_N + dMice_S   (two independent pole budgets)
│       │
│       ├── net_sub_N/S = (Q_in_pole − Q_out_pole) · A_cap / L_sub
│       │                     │              │          │       │
│       │                     │         ε·σ·149⁴       │    5.7×10⁵ J/kg
│       │                     │         = 26.51 W/m²   │
│       │                     │                   0.01·4πR²  (= 1.443×10¹² m²)
│       │          (1−α)·F·max(0, ±sin(δ))
│       │                    │         │
│       │                    F         δ  ← Ls ← orbital_angle
│       │
│       └── Guard: each pole clamped independently if that pole's ice = 0
│
│
└── [dP/dt]  = dP_escape + dP_sublimation + dP_tide
        │               │                       │
        │       −dM_ice/dt · g / (4πR²)         −A·ω·sin(ω·t + φ)
        │               │         │              │    │           │
        │          dM_ice/dt    g, R           30 Pa  ω       −0.7π
        │                                               │
        │                                      elapsed_time (diurnal cycle)
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
| C_area = 6.0×10⁴ constant everywhere | All surface points have Gale Crater's rocky-sandy thermal inertia regardless of actual location | `MARS_THERMAL_INERTIA` |
| Polar cos(z) = max(0, ±sin(δ)) | Exact at 90° poles, approximate at cap edges | `compute_derivatives` polar section |
| A_cap = 1% per pole (fixed) | Fixed sublimating area; no seasonal cap extent evolution | `MARS_POLAR_CAP_FRACTION` |
| Thermal tide empirical (A=30 Pa, φ=−0.7π) | Reproduces REMS diurnal P oscillation but not spatial structure or interannual variability | `MARS_THERMAL_TIDE_PA/PHASE` |
| 0D global pressure | No spatial pressure gradient or local weather | dP/dt formulation |
| f_gh = 1.02 fixed | Greenhouse does not evolve with pressure changes | `thermal.greenhouse_factor` |
| MAVEN escape rate constant | Does not vary with solar activity or season | `MARS_MAVEN_ESCAPE_RATE` |
| North/south ice split by season at init | Linear ramp from Ls=180°→360°; real cap exchange is more complex | `setup_properties` |

### 10.1 Elevation and Topography Treatment

The current `Mars` implementation accepts `elevation_m`, but only uses it once
when constructing the initial surface pressure:

```
P_initial = P_ref · exp(-elevation_m / H)
H = 11 100 m
```

This is a hydrostatic/barometric correction with a fixed Mars CO2 scale height.
It captures the first-order fact that high terrain starts with lower surface
pressure and low terrain starts with higher surface pressure. For example,
relative to the same reference pressure, +5 km terrain starts at about 64% of
datum pressure, -7 km terrain starts at about 188%, and +21 km terrain starts at
about 15%.

What is modeled:

- Initial local pressure is corrected from the supplied reference pressure using
  `elevation_m`.
- That corrected pressure is what enters the evolved state vector as `P`.
- Batched runs inherit the corrected initial pressure because `BatchedMars`
  stacks each `Mars` instance's initialized pressure.

What is absent:

- Elevation is not retained as an explicit model state after initialization.
- The pressure scale height is fixed, not recomputed from temperature,
  composition, or gravity during the run.
- Pressure evolution is 0D/global after initialization. CO2 cap exchange,
  atmospheric escape, and the empirical thermal tide are added directly to the
  local initialized pressure, without recalculating a vertical pressure column.
- No temperature lapse rate or altitude-dependent surface temperature correction
  is applied. Temperature depends on latitude, season, solar flux, albedo,
  emissivity, greenhouse factor, thermal inertia, and diurnal phase, but not
  terrain height.
- No MOLA DEM or other topography lookup is used by `package/src/`. Latitude and
  longitude are accepted as inputs, but longitude does not currently select a
  topographic height or alter local solar time in the implemented equations.
- Slope, aspect, horizon shading, basin cold trapping, and terrain-dependent
  thermal inertia/albedo are not modeled.

Physical impact:

- The model can distinguish a high-altitude and low-altitude site at `t=0` only
  through pressure, and only if the caller supplies `elevation_m` manually.
- Missing lapse-rate physics can be a large local-temperature error over Mars'
  relief. A dry CO2 adiabatic estimate is roughly 4-5 K/km, so Hellas-like
  basins, datum plains, and Tharsis/Olympus-like highlands could differ by
  tens of kelvin from altitude effects alone. The real near-surface Martian
  lapse rate is variable and can invert, but ignoring altitude removes this
  entire class of behavior.
- CO2 frost stability and sublimation are sensitive to both pressure and
  temperature. Omitting altitude-dependent temperature means the model may
  misplace where CO2 or H2O ice should be stable, especially in deep basins,
  polar scarps, crater floors, and high volcanic terrain.
- Without MOLA coupling, named-location simulations such as Hellas Basin, Gale
  Crater, Olympus Mons, and Elysium Mons are not physically distinct unless
  their latitude, initial pressure/temperature, albedo, thermal inertia, and
  `elevation_m` are provided externally.
- Because pressure is documented elsewhere as a global mean state while the
  elevation correction makes it local at initialization, downstream consumers
  should treat current `surface_pressure` as a reduced-order local diagnostic,
  not a resolved Mars pressure field.

---



## 11. Citations

### Physics Laws and Mathematical Methods

| Law / Method | Section used | Wikipedia |
|---|---|---|
| Kepler's First Law — orbital ellipse r(θ) = a(1−e²)/(1+e·cosθ) | §3.2 | [Kepler's laws](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion) |
| Kepler's Second Law — equal areas in equal times (approximation note) | §3.1 | [Mean motion](https://en.wikipedia.org/wiki/Mean_motion) |
| Mean anomaly / mean motion dθ/dt = 2π/T | §3.1 | [Mean anomaly](https://en.wikipedia.org/wiki/Mean_anomaly) · [True anomaly](https://en.wikipedia.org/wiki/True_anomaly) |
| Inverse-square law — F = F₀·(1AU/r)² | §3.3 | [Inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law) |
| Solar declination — δ = arcsin(sin(ε)·sin(Ls)) | §4.2 | [Position of the Sun](https://en.wikipedia.org/wiki/Position_of_the_Sun) |
| Hour angle — h = ω·t − π + λ | §4.3 | [Hour angle](https://en.wikipedia.org/wiki/Hour_angle) |
| Solar zenith angle — cos(z) = sin(φ)sin(δ) + cos(φ)cos(δ)cos(h) | §4.4 | [Solar zenith angle](https://en.wikipedia.org/wiki/Solar_zenith_angle) |
| First Law of Thermodynamics — C·dT/dt = Q_in − Q_out | §5 | [First law of thermodynamics](https://en.wikipedia.org/wiki/First_law_of_thermodynamics) |
| Lambert's cosine law — Q_in = (1−α)·F·cos(z) | §5.1 | [Lambert's cosine law](https://en.wikipedia.org/wiki/Lambert%27s_cosine_law) |
| Stefan-Boltzmann law — Q_out = ε·σ·T⁴ | §5.2, §6.2 | [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) |
| Thermal inertia — C_area = ρ·c_p·d | §5.3 | [Thermal inertia](https://en.wikipedia.org/wiki/Thermal_inertia) |
| Latent heat of sublimation — ΔE = L·ΔM | §6.2–6.3 | [Latent heat](https://en.wikipedia.org/wiki/Latent_heat) · [Sublimation](https://en.wikipedia.org/wiki/Sublimation_(phase_transition)) |
| Hydrostatic equilibrium — P = M·g / A | §7.1 | [Hydrostatic equilibrium](https://en.wikipedia.org/wiki/Hydrostatic_equilibrium) |
| Atmospheric escape (non-thermal) — sputtering, photochemistry | §7.2 | [Atmospheric escape](https://en.wikipedia.org/wiki/Atmospheric_escape) |
| Jeans escape parameter λ = G·M·m/(k·T·R) | §7.2 | [Jeans escape](https://en.wikipedia.org/wiki/Atmospheric_escape#Jeans_escape) |
| Atmospheric thermal tide — P_tide = A·cos(ω·t + φ) | §7.3 | [Atmospheric tide](https://en.wikipedia.org/wiki/Atmospheric_tide) |
| Sunrise hour angle — cos(h₀) = −tan(φ)·tan(δ) | §8.1 | [Sunrise equation](https://en.wikipedia.org/wiki/Sunrise_equation) |
| Radiative equilibrium temperature — T_eq = [(absorbed)/(ε·σ)]^0.25 | §8.2 | [Radiative equilibrium](https://en.wikipedia.org/wiki/Radiative_equilibrium) |
| Newton's law of cooling / exponential relaxation | §8.3 | [Newton's law of cooling](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) |
| Taylor linearisation of T⁴ → τ = C/(4·ε·σ·T³) | §8.3 | [Linearization](https://en.wikipedia.org/wiki/Linearization) · [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) |
| 4th-order Runge-Kutta (ACCURATE path) | §1, §8.4 | [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) |
| Forward Euler method (ice/pressure in FAST path) | §8.4 | [Euler method](https://en.wikipedia.org/wiki/Euler_method) |
| Numerical stability / stiff ODE | §8.4 | [Stiff equation](https://en.wikipedia.org/wiki/Stiff_equation) · [Numerical stability](https://en.wikipedia.org/wiki/Numerical_stability) |

### Scientific Literature

| Reference | Used for |
|-----------|---------|
| NASA Mars Fact Sheet — https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html | All planetary constants (M, R, g, a, e, T_orb, ε_tilt) |
| Kepler, J. (1609). *Astronomia Nova* | Orbital ellipse law |
| Lambert, J.H. (1760). *Photometria* | Lambert's cosine law — Q = F·cos(z) |
| Stefan, J. (1879); Boltzmann, L. (1884) | Thermal emission Q = ε·σ·T⁴ |
| Clausius, R. (1850). *First Law of Thermodynamics* | dU/dt = Q_in − Q_out |
| Jakosky, B.M. et al. (2018). *Science*, 355(6323). doi:10.1126/science.aan5015 | MAVEN non-thermal escape rate ~0.2 kg/s |
| Kieffer, H.H. et al. (1977). *JGR*, 82(28) | CO₂ frost point 149 K at Mars surface pressure |
| Smith, M.D. (2004). *Icarus*, 167(1), 148–165 | CO₂ latent heat L_sub = 5.7×10⁵ J/kg |
| Haberle, R.M. et al. (1993). *JGR*, 98(E2) | Seasonal pressure swing driven by polar cap exchange |
| Vasavada, A.R. et al. (2017). *JGR Planets*, 122(5) | REMS Gale Crater temperature profiles, Sol 224 calibration |
| Christensen, P.R. et al. (2001). *JGR*, 106(E10). THEMIS | Thermal inertia maps — TI ≈ 200–350 TIU at Gale Crater |
| Wilson, R.J. & Hamilton, K. (1996). *JAS*, 53(9) | Martian atmospheric thermal tides — amplitude and phase |
