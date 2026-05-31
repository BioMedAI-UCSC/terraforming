# Mars Physics Model

Complete derivation, dependency graph, and worked examples for every equation in
[`src.celestials`](../api/celestials.md) and its supporting framework in [`src.framework`](../api/framework.md).

---

## Table of Contents

1. [State Vector](#1-state-vector)
2. [Planetary Constants](#2-planetary-constants)
3. [Orbital Mechanics — advance_orbit](#3-orbital-mechanics--advance_orbit)
4. [Solar Geometry — compute_derivatives step 1](#4-solar-geometry)
5. [dT/dt — Surface Energy Balance](#5-dtdt--surface-energy-balance)
6. [dM_ice/dt — Polar CO₂ Sublimation](#6-dm_icedt--polar-co-sublimation)
7. [dP/dt — Atmospheric Pressure](#7-dpdt--atmospheric-pressure)
8. [FAST path — compute_fast_physics](#8-fast-path--compute_fast_physics)
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
| $M$ | Mass | $6.4171\times10^{23}$ kg | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $R$ | Radius | $3.3895\times10^6$ m | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $g$ | Surface gravity | $3.72076$ m s⁻² | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $T_\text{rot}$ | Rotation period | $88\,775.244$ s (1 sol) | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $a$ | Semi-major axis | $2.27939\times10^{11}$ m (1.524 AU) | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $e$ | Eccentricity | $0.0934$ | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $T_\text{orb}$ | Orbital period | $5.93568\times10^7$ s (686.97 days) | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |
| $\varepsilon_\text{tilt}$ | Axial tilt | $25.19° = 0.4396$ rad | [NASA Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) |

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
| `MARS_THERMAL_INERTIA` | $6.0\times10^4$ J K⁻¹ m⁻² | Surface thermal mass per unit area — controls diurnal temperature amplitude |
| `MARS_POLAR_CAP_FRACTION` | $0.01$ | Effective fractional surface area of each sublimating polar cap |
| `MARS_THERMAL_TIDE_PA` | $30.0$ Pa | Half-amplitude of empirical diurnal pressure oscillation |
| `MARS_THERMAL_TIDE_PHASE` | $-0.7\pi$ rad | Phase offset — puts pressure max at ~08:37 LMST |
| `MARS_CO2_FROST_POINT` | $149.0$ K | CO₂ condensation/sublimation temperature at Mars surface pressure |
| `MARS_CO2_LATENT_HEAT` | $5.7\times10^5$ J kg⁻¹ | Latent heat of CO₂ sublimation |
| `MARS_MAVEN_ESCAPE_RATE` | $0.2$ kg s⁻¹ | Non-thermal atmospheric escape ([Jakosky et al. 2018](https://doi.org/10.1126/science.aan5015)) |
| `MARS_SURFACE_EMISSIVITY` | $0.95$ | Near-blackbody IR emissivity of basaltic regolith |

---

## 3. Orbital Mechanics — `advance_orbit`

Called at the start of every timestep by the engine. Updates `elapsed_time` and
`orbital_angle`, then recomputes `solar_flux`.

### 3.1 Mean Motion

> **Laws applied:** [Kepler's Second Law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Second_law) · [Mean motion](https://en.wikipedia.org/wiki/Mean_motion) · [Mean anomaly](https://en.wikipedia.org/wiki/Mean_anomaly)

$$\frac{d\theta}{dt} = \frac{2\pi}{T_\text{orb}}$$

$$\theta(t + \Delta t) = \left(\theta(t) + \frac{2\pi\,\Delta t}{T_\text{orb}}\right) \bmod 2\pi$$

**Derivation:** A planet completing one full orbit ($2\pi$ rad) in time $T_\text{orb}$ sweeps
angle at constant rate $2\pi/T_\text{orb}$. The $\bmod\,2\pi$ wraps the angle back into $[0, 2\pi)$
after each complete orbit, preventing unbounded accumulation.

**Approximation:** This advances $\theta$ as the [mean anomaly](https://en.wikipedia.org/wiki/Mean_anomaly)
(constant angular rate) but uses it as the [true anomaly](https://en.wikipedia.org/wiki/True_anomaly)
(actual ellipse position). In reality [Kepler's second law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#Second_law) requires the planet to move faster at perihelion and slower at aphelion.
For Mars ($e = 0.0934$) this introduces a phase error of up to ±10° in solar
longitude — roughly ±5–10 sols timing error on seasonal peaks.

**Worked example ($\Delta t = 1$ s):**

```
Δθ = 2π × 1 / 59 356 800
   = 6.283185 / 59 356 800
   = 1.05853×10⁻⁷ rad  =  6.065×10⁻⁶ °

After one full year (59 356 800 steps):
  θ = 59 356 800 × 1.05853×10⁻⁷ = 2π rad → mod 2π = 0  ✓
```

**Per-timestep values:**

| $\Delta t$ | $\Delta\theta$ (rad) | $\Delta\theta$ (degrees) |
|----|---------|-------------|
| 1 s | $1.059\times10^{-7}$ | $6.07\times10^{-6}$° |
| 900 s | $9.53\times10^{-5}$ | $0.00546$° |
| 3 600 s | $3.81\times10^{-4}$ | $0.0218$° |
| 88 775 s (1 sol) | $9.395\times10^{-3}$ | $0.538$° |

### 3.2 Kepler Ellipse — distance_from_sun

> **Laws applied:** [Kepler's First Law](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion#First_law) · [Elliptic orbit](https://en.wikipedia.org/wiki/Elliptic_orbit) · [Orbital eccentricity](https://en.wikipedia.org/wiki/Orbital_eccentricity)

$$r(\theta) = \frac{a\,(1 - e^2)}{1 + e\cos\theta}$$

| $\theta$ | Location | $r$ (m) | Solar flux $F$ (W m⁻²) |
|---|---------|-------|-------------------|
| $0$ | Perihelion | $2.066\times10^{11}$ | 717 |
| $\pi$ | Aphelion | $2.493\times10^{11}$ | 492 |

Mars receives **45% more solar power at perihelion than aphelion** — the primary
driver of its strong seasonal asymmetry.

### 3.3 Inverse-Square Solar Flux

> **Laws applied:** [Inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law) · [Solar irradiance](https://en.wikipedia.org/wiki/Solar_irradiance) · [Solar constant](https://en.wikipedia.org/wiki/Solar_constant)

$$F = F_0 \left(\frac{1\,\text{AU}}{r}\right)^2, \qquad F_0 = 1361\;\text{W\,m}^{-2},\quad 1\,\text{AU} = 1.496\times10^{11}\;\text{m}$$

**At perihelion** ($\theta = 0$, $r = 2.066\times10^{11}$ m):

```
F = 1361 × (1.496×10¹¹ / 2.066×10¹¹)² = 1361 × 0.525 = 714 W/m²
```

**At aphelion** ($\theta = \pi$, $r = 2.493\times10^{11}$ m):

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

Computed inside `compute_derivatives` before $dT/dt$. Determines how much of the
solar flux actually hits a surface patch at latitude $\phi$, longitude $\lambda$, time $t$.

### 4.1 Solar Longitude $L_s$

> **Reference:** [Solar longitude](https://en.wikipedia.org/wiki/Solar_longitude) · [Areocentric coordinates](https://en.wikipedia.org/wiki/Mars#Seasons)

$$L_s = \theta + 251°$$

$L_s$ maps the orbital angle to the areocentric solar longitude — the standard Mars
seasonal calendar. $L_s = 251°$ is perihelion (deep southern summer).

### 4.2 Solar Declination $\delta$

> **Laws applied:** [Position of the Sun](https://en.wikipedia.org/wiki/Position_of_the_Sun) · [Solar declination](https://en.wikipedia.org/wiki/Position_of_the_Sun#Declination_of_the_Sun_as_seen_from_Earth) · [Axial tilt](https://en.wikipedia.org/wiki/Axial_tilt)

$$\delta = \arcsin\!\bigl(\sin\varepsilon_\text{tilt}\cdot\sin L_s\bigr)$$

$\delta$ is the latitude of the sub-solar point — where the Sun is directly overhead.

**At perihelion** ($L_s = 251°$):

```
sin(25.19°) = 0.42533
sin(251°)   = −sin(71°) = −0.94552

δ = arcsin(0.42533 × −0.94552) = arcsin(−0.40222) = −23.71°
```

Sub-solar point is 23.71° south — southern hemisphere summer.

**Seasonal declination cycle:**

| $L_s$ | Season | $\delta$ |
|----|--------|---|
| $0°$ | N. spring equinox | $0°$ |
| $90°$ | N. summer solstice | $+25.19°$ |
| $180°$ | N. autumn equinox | $0°$ |
| $251°$ | Perihelion | $-23.71°$ |
| $270°$ | N. winter solstice | $-25.19°$ |

### 4.3 Hour Angle $h$

> **Reference:** [Hour angle](https://en.wikipedia.org/wiki/Hour_angle) · [Solar time](https://en.wikipedia.org/wiki/Solar_time)

$$\omega = \frac{2\pi}{T_\text{rot}} = 7.0792\times10^{-5}\;\text{rad\,s}^{-1}, \qquad h = \omega t - \pi + \lambda$$

$t=0 \Rightarrow h = -\pi$ (midnight). The planet rotates eastward so $h$ increases with time.
$\lambda$ shifts the reference: eastern longitudes see solar noon earlier.

| $t$ (s) | $h$ (rad) | Local time |
|-------|---------|-----------|
| 0 | $-\pi$ | Midnight |
| $T_\text{rot}/4$ | $-\pi/2$ | Dawn |
| $T_\text{rot}/2$ | $0$ | **Solar noon** |
| $3T_\text{rot}/4$ | $+\pi/2$ | Dusk |
| $T_\text{rot}$ | $+\pi \to -\pi$ | Midnight |

### 4.4 Cosine of Solar Zenith Angle

> **Laws applied:** [Solar zenith angle](https://en.wikipedia.org/wiki/Solar_zenith_angle) · [Spherical trigonometry](https://en.wikipedia.org/wiki/Spherical_trigonometry)

$$\cos z = \max\!\bigl(0,\;\sin\phi\sin\delta + \cos\phi\cos\delta\cos h\bigr)$$

Clamped to zero when the Sun is below the horizon.

**Worked example — perihelion, $\phi = 22°$N, $\lambda = 0°$:**

```
A = sin(22°)·sin(−23.71°) = 0.37461 × (−0.40222) = −0.15062
B = cos(22°)·cos(−23.71°) = 0.92718 × 0.91558    = +0.84913

cos(z) = max(0,  −0.15062  +  0.84913 · cos(h))
```

| $t$ (s) | $h$ | $\cos h$ | $\cos z$ | Sun elevation |
|-------|---|--------|--------|--------------|
| 0 | $-180°$ | $-1.000$ | $0$ | Below horizon |
| 22 194 | $-90°$ | $0.000$ | $0$ | Below horizon |
| 44 388 | $0°$ | $+1.000$ | $\mathbf{0.699}$ | **44.3°** |
| 66 581 | $+90°$ | $0.000$ | $0$ | Below horizon |

**Sunrise/sunset hour angle $h_0$:**

$$\cos h_0 = -\tan\phi\tan\delta = -\tan(22°)\tan(-23.71°) = 0.17732$$

$$h_0 = \arccos(0.17732) = 79.79°$$

$$\text{Day length} = \frac{2 \times 79.79°}{360°} \times 88\,775\;\text{s} = 39\,360\;\text{s} \approx 10.93\;\text{hours}$$

Winter at 22°N — only 10.93 of 24.66 hours are daylit.

---

## 5. dT/dt — Surface Energy Balance

> **Laws applied:** [First Law of Thermodynamics](https://en.wikipedia.org/wiki/First_law_of_thermodynamics) · [Radiative energy balance](https://en.wikipedia.org/wiki/Earth%27s_energy_budget) · [Ordinary differential equation](https://en.wikipedia.org/wiki/Ordinary_differential_equation)

$$\frac{dT}{dt} = \frac{Q_\text{in} - Q_\text{out}}{C_\text{area}} \quad [\text{K\,s}^{-1}]$$

### 5.1 $Q_\text{in}$ — Absorbed Solar Radiation

> **Laws applied:** [Lambert's cosine law](https://en.wikipedia.org/wiki/Lambert%27s_cosine_law) · [Albedo](https://en.wikipedia.org/wiki/Albedo)

$$Q_\text{in} = (1 - \alpha)\,F\cos z \quad [\text{W\,m}^{-2}]$$

| Term | Value | Meaning |
|------|-------|---------|
| $\alpha$ | $0.25$ | Bond albedo — 25% of incident light reflected |
| $F$ | 492–717 W m⁻² | Solar flux (from `advance_orbit`) |
| $\cos z$ | $0$–$1$ | Geometric projection onto surface |

**At solar noon, perihelion:**

```
Q_in = (1 − 0.25) × 714 × 0.699 = 0.75 × 714 × 0.699 = 374 W/m²
```

**At midnight:** $Q_\text{in} = 0$ ($\cos z$ clamped to 0).

### 5.2 $Q_\text{out}$ — Thermal Emission

> **Laws applied:** [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) · [Greenhouse effect](https://en.wikipedia.org/wiki/Greenhouse_effect) · [Thermal radiation](https://en.wikipedia.org/wiki/Thermal_radiation)

$$T_\text{eff} = \frac{T}{\max(f_\text{gh},\,1.0)}, \qquad Q_\text{out} = \varepsilon\,\sigma\,T_\text{eff}^4 \quad [\text{W\,m}^{-2}]$$

where $\varepsilon = 0.95$ (near-blackbody IR emissivity) and $\sigma = 5.670374\times10^{-8}$ W m⁻² K⁻⁴.

**Greenhouse factor $f_\text{gh} = 1.02$:**

Mars's thin CO₂ column absorbs weakly in the 15 μm infrared band, trapping ~8% of outgoing radiation:

```
Q_out_bare     = ε · σ · 210⁴ = 109.4 W/m²   (no atmosphere)
Q_out_with_fgh = ε · σ · (210/1.02)⁴ = 102.6 W/m²

Trapped = 6.8 W/m²  (~4 K surface warming)
```

| Body | $f_\text{gh}$ | Warming |
|------|------|---------|
| Mars | 1.02 | ~4 K |
| Earth | 1.33 | ~33 K |
| Venus | ~3.2 | ~500 K |

### 5.3 $C_\text{area}$ — Thermal Inertia

> **Reference:** [Thermal inertia](https://en.wikipedia.org/wiki/Thermal_inertia) · [Thermal skin depth](https://en.wikipedia.org/wiki/Thermal_contact_conductance) · [THEMIS instrument](https://en.wikipedia.org/wiki/2001_Mars_Odyssey#THEMIS)

$$C_\text{area} = \rho\,c_p\,d = 6.0\times10^4\;\text{J\,K}^{-1}\,\text{m}^{-2}$$

where $\rho \approx 2000$–$3000$ kg m⁻³ (regolith), $c_p \approx 800$ J kg⁻¹ K⁻¹ (basalt at 200 K),
$d \approx 0.025$–$0.037$ m (diurnal thermal skin depth).

**Why this value:** THEMIS and TES measurements at Gale Crater give thermal inertia
TI ≈ 200–350 TIU. Converting: $C_\text{area} = \text{TI} \times \sqrt{T_\text{rot}/\pi} \approx \text{TI} \times 168$,
giving a range of $3.4$–$5.9\times10^4$. The value $6.0\times10^4$ sits at the upper end,
consistent with the rockier sections of Gale Crater visible in REMS diurnal profiles.

**Calibration history:**

| $C_\text{area}$ | Diurnal swing (model) | REMS Sol 224 |
|--------|----------------------|--------------|
| $2.0\times10^4$ (old — loose dust) | ~180 K | ~65 K |
| $6.0\times10^4$ (current — rocky/sandy) | ~60 K | ~65 K ✓ |

A **3× increase in $C_\text{area}$** → **3× smaller $dT/dt$** at the same $Q_\text{in}-Q_\text{out}$.

### 5.4 Full Dependency Table

| Variable | Fixed or dynamic | Source |
|----------|-----------------|--------|
| $\alpha$ | Fixed | `radiation.albedo` (init param) |
| $F$ | Dynamic each step | `advance_orbit` → Kepler |
| $\cos z$ | Dynamic each step | $\phi$, $\lambda$, $t$, $\delta(\theta)$ |
| $\varepsilon$ | Fixed (0.95) | Hardcoded |
| $f_\text{gh}$ | Fixed (1.02) | `thermal.greenhouse_factor` |
| $T$ | State variable | ODE solution |
| $C_\text{area}$ | Fixed ($6.0\times10^4$) | Calibrated to REMS Sol 224 Gale Crater |

---

## 6. dM_ice/dt — Polar CO₂ Sublimation

> **Laws applied:** [Latent heat](https://en.wikipedia.org/wiki/Latent_heat) · [Phase transition](https://en.wikipedia.org/wiki/Phase_transition) · [Sublimation](https://en.wikipedia.org/wiki/Sublimation_(phase_transition)) · [Frost point](https://en.wikipedia.org/wiki/Dew_point#Frost_point)

CO₂ ice caps are pinned at the **frost point $T_\text{frost} = 149$ K**. While ice
exists, all net radiative energy goes into phase change rather than warming.

### 6.1 Polar Insolation (Simplified)

> **Reference:** [Polar night](https://en.wikipedia.org/wiki/Polar_night) · [Midnight sun](https://en.wikipedia.org/wiki/Midnight_sun) · [Insolation](https://en.wikipedia.org/wiki/Solar_irradiance#Insolation)

At the geographic poles the hour angle averages out over one full sol and the mean daily insolation factor collapses to $\sin\delta$:

$$\cos z_N = \max(0,\,{+}\sin\delta), \qquad \cos z_S = \max(0,\,-\sin\delta)$$

**Seasonal behaviour:**

| $L_s$ | $\delta$ | $\cos z_N$ | $\cos z_S$ | Which cap sublimates |
|----|---|---------|---------|---------------------|
| $90°$ (N. summer) | $+25.19°$ | 0.425 | 0 | North |
| $180°$ (equinox) | $0°$ | 0 | 0 | Neither |
| $251°$ (perihelion) | $-23.71°$ | 0 | 0.402 | South |
| $270°$ (N. winter) | $-25.19°$ | 0 | 0.425 | South (peak) |

### 6.2 Energy Balance at Frost Point

> **Laws applied:** [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) · [Latent heat of sublimation](https://en.wikipedia.org/wiki/Enthalpy_of_sublimation) · [Energy balance](https://en.wikipedia.org/wiki/Earth%27s_energy_budget)

$$Q_\text{in,pole} = (1-\alpha)\,F\cos z_\text{pole} \quad [\text{W\,m}^{-2}]$$

$$Q_\text{out,pole} = \varepsilon\,\sigma\,T_\text{frost}^4 = 0.95 \times 5.670374\times10^{-8} \times 149^4 = 26.51\;\text{W\,m}^{-2}$$

$$\Delta Q = Q_\text{in,pole} - Q_\text{out,pole}$$

$\Delta Q > 0$ → ice **sublimates** (gains energy, turns to gas).
$\Delta Q < 0$ → CO₂ **condenses** (loses energy, freezes from atmosphere).

### 6.3 Two-Pole Ice Budget with Per-Pole Reservoirs

The model tracks north and south caps **independently**, each with its own ice
reservoir.

**Initial ice split** (set in `setup_properties`):

```python
if 0° ≤ Ls < 180°:          # Northern summer — north CO₂ already sublimated
    f_north = 0.0
else:                        # Northern winter — north cap growing
    f_north = 0.4 × (Ls − 180°) / 180°   # linearly 0→0.4

f_south = 1.0 − f_north
```

**Mass rate equations:**

$$A_\text{cap} = 0.01 \times 4\pi R^2 = 1.443\times10^{12}\;\text{m}^2, \qquad L_\text{sub} = 5.7\times10^5\;\text{J\,kg}^{-1}$$

$$\dot{M}_\text{sub,N} = \frac{\Delta Q_N \cdot A_\text{cap}}{L_\text{sub}}, \qquad \dot{M}_\text{sub,S} = \frac{\Delta Q_S \cdot A_\text{cap}}{L_\text{sub}}$$

$$\frac{dM_\text{ice,N}}{dt} = -\dot{M}_\text{sub,N}, \qquad \frac{dM_\text{ice,S}}{dt} = -\dot{M}_\text{sub,S}, \qquad \frac{dM_\text{ice}}{dt} = \frac{dM_\text{ice,N}}{dt} + \frac{dM_\text{ice,S}}{dt}$$

**Guard conditions:**

- Sublimation ($dM_\text{ice,N} < 0$) blocked if `ice_mass_north = 0`
- Sublimation ($dM_\text{ice,S} < 0$) blocked if `ice_mass_south = 0`
- Condensation always allowed (CO₂ can always refreeze)

### 6.4 Worked Example — Perihelion ($L_s = 251°$, south cap peak)

```
δ = −23.71°
cos(z)_S = max(0, −sin(−23.71°)) = 0.40222
cos(z)_N = 0

F = 714 W/m²  (perihelion)
Q_in_S = (1 − 0.25) × 714 × 0.40222 = 214.9 W/m²
Q_out_pole = 26.51 W/m²
ΔQ_S = 214.9 − 26.51 = 188.4 W/m²

A_cap = 1.443×10¹² m²

net_sub_S = 188.4 × 1.443×10¹² / 5.7×10⁵ = 4.77×10⁸ kg/s
```

$$\frac{dM_\text{ice}}{dt} \approx -4.77\times10^8\;\text{kg\,s}^{-1}$$

Over one sol (88 775 s):

```
ΔM_ice ≈ −4.77×10⁸ × 88 775 = −4.23×10¹³ kg/sol
```

Starting from $M_\text{ice} = 5\times10^{15}$ kg, the south cap would fully sublimate in
~118 sols at peak perihelion insolation.

---

## 7. dP/dt — Atmospheric Pressure

> **Laws applied:** [Hydrostatic equilibrium](https://en.wikipedia.org/wiki/Hydrostatic_equilibrium) · [Atmospheric pressure](https://en.wikipedia.org/wiki/Atmospheric_pressure) · [Conservation of mass](https://en.wikipedia.org/wiki/Conservation_of_mass)

Three contributions to global mean surface pressure:

$$\frac{dP}{dt} = \frac{dP_\text{escape}}{dt} + \frac{dP_\text{sub}}{dt} + \frac{dP_\text{tide}}{dt}$$

### 7.1 Ice–Atmosphere Mass Exchange (dominant)

> **Laws applied:** [Hydrostatic equilibrium](https://en.wikipedia.org/wiki/Hydrostatic_equilibrium) · [Ideal gas law](https://en.wikipedia.org/wiki/Ideal_gas_law) · [Barometric formula](https://en.wikipedia.org/wiki/Barometric_formula)

From hydrostatic equilibrium: $P = M_\text{atm}\,g / A_\text{planet}$.
Conservation of mass gives $M_\text{atm} + M_\text{ice} = \text{const}$, so:

$$\frac{dP_\text{sub}}{dt} = -\frac{dM_\text{ice}}{dt} \cdot \frac{g}{A_\text{planet}}$$

**Signs:**
- Sublimation ($dM_\text{ice}/dt < 0$) → $dP/dt > 0$ (pressure rises as CO₂ enters atmosphere) ✓
- Condensation ($dM_\text{ice}/dt > 0$) → $dP/dt < 0$ (pressure falls as CO₂ leaves atmosphere) ✓

**Real Mars observations (Viking landers):**

| Season | $P$ (Pa) | Driver |
|--------|--------|--------|
| N. winter / S. summer (perihelion) | ~740 | South cap sublimating |
| N. summer / S. winter | ~560 | South cap condensing |
| Swing | ~25% | Entirely from $dM_\text{ice}/dt$ |

### 7.2 Non-thermal Atmospheric Escape (secular)

> **Reference:** [Atmospheric escape](https://en.wikipedia.org/wiki/Atmospheric_escape) · [Sputtering](https://en.wikipedia.org/wiki/Sputtering) · [MAVEN mission](https://en.wikipedia.org/wiki/MAVEN)

Mars loses atmosphere primarily via **non-thermal** mechanisms — solar wind
sputtering and photochemical escape — measured by the MAVEN spacecraft
([Jakosky et al. 2018](https://doi.org/10.1126/science.aan5015)):

$$\frac{dP_\text{escape}}{dt} = -\frac{0.2\;\text{kg\,s}^{-1} \times g}{A_\text{planet}} = -5.14\times10^{-15}\;\text{Pa\,s}^{-1}$$

Per sol: $\Delta P_\text{escape} \approx 4.56\times10^{-10}$ Pa/sol — completely negligible vs the ~180 Pa
seasonal swing. Becomes significant only on geological timescales (millions of years).

**Why not Jeans escape?**

> **Reference:** [Jeans escape](https://en.wikipedia.org/wiki/Atmospheric_escape#Jeans_escape) · [Maxwell-Boltzmann distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution)

For CO₂ at $T = 210$ K, the Jeans escape parameter is:

$$\lambda = \frac{G M m_{\text{CO}_2}}{k_B\,T\,R_\text{exo}} \approx 299 \implies e^{-\lambda} = e^{-299} \approx 10^{-130}$$

CO₂ molecules are too massive to escape Mars thermally. Jeans escape is the
correct mechanism for hydrogen and helium, not CO₂. The empirical MAVEN constant
correctly represents the actual non-thermal mechanisms.

### 7.3 Thermal Tide — Empirical Diurnal Pressure Oscillation

> **Reference:** [Atmospheric tide](https://en.wikipedia.org/wiki/Atmospheric_tide) · [Diurnal cycle](https://en.wikipedia.org/wiki/Diurnal_cycle) · [Harmonic oscillator](https://en.wikipedia.org/wiki/Harmonic_oscillator)

Real Mars exhibits a ~40–60 Pa diurnal pressure wave at Gale Crater (clearly
visible in REMS data). The parameterisation adds the time-derivative of a sinusoid directly to $dP/dt$:

$$\frac{dP_\text{tide}}{dt} = -A\,\omega\sin(\omega t + \varphi)$$

where $A = 30.0$ Pa, $\varphi = -0.7\pi$ rad, $\omega = 2\pi/T_\text{rot} = 7.0792\times10^{-5}$ rad s⁻¹.

This is the analytical derivative of $P_\text{tide}(t) = A\cos(\omega t + \varphi)$, giving
pressure a **zero-mean sinusoidal oscillation of amplitude 30 Pa**.

**Phase derivation — when is pressure maximum?**

$P_\text{tide}$ is maximum when $\cos(\omega t + \varphi) = 1$:

$$\omega t = -\varphi = 0.7\pi \implies t = 0.35 \times T_\text{rot} = 31\,071\;\text{s} \approx 08{:}37\;\text{LMST} \checkmark$$

$P_\text{tide}$ is minimum at $\omega t = 1.7\pi \implies t = 75\,459\;\text{s} \approx 20{:}57\;\text{LMST}$ ✓

**Zero mean — no secular drift:**

$$\int_0^{T_\text{rot}} P_\text{tide}\,dt = \int_0^{T_\text{rot}} A\cos(\omega t + \varphi)\,dt = 0$$

Over any integer number of sols, the tide adds/removes exactly equal amounts of
pressure — it does not bias the long-term seasonal trend.

---

## 8. FAST Path — `compute_fast_physics`

> **Methods applied:** [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) · [Stiff ODE](https://en.wikipedia.org/wiki/Stiff_equation) · [Newtonian cooling](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) · [Euler method](https://en.wikipedia.org/wiki/Euler_method)

An alternative to full RK4 integration for large timesteps. Computes the same
physics analytically via equilibrium + relaxation.

### 8.1 Daily-Mean Insolation Factor

> **Reference:** [Insolation](https://en.wikipedia.org/wiki/Solar_irradiance#Insolation) · [Sunrise equation](https://en.wikipedia.org/wiki/Sunrise_equation) · [Definite integral](https://en.wikipedia.org/wiki/Integral)

Instead of instantaneous $\cos z$, the fast path computes the daily average analytically:

$$\cos h_0 = \mathrm{clamp}(-\tan\phi\tan\delta,\;-1,\;1), \qquad h_0 = \arccos(\cos h_0)$$

$$\bar{I} = \frac{h_0\sin\phi\sin\delta + \cos\phi\cos\delta\sin h_0}{\pi}$$

This is the closed-form integral of $\cos z$ over one full sol.

### 8.2 Radiative Equilibrium Temperature

> **Laws applied:** [Radiative equilibrium](https://en.wikipedia.org/wiki/Radiative_equilibrium) · [Effective temperature](https://en.wikipedia.org/wiki/Effective_temperature) · [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)

$$T_\text{eq,base} = \left[\frac{(1-\alpha)\,F\,\bar{I}}{\varepsilon\,\sigma}\right]^{1/4}, \qquad T_\text{eq} = T_\text{eq,base} \times f_\text{gh}$$

The surface equilibrium temperature — where $Q_\text{in} = Q_\text{out}$ if the surface had
infinite time to adjust.

**Diurnal swing superimposed:**

$$T_\text{eq} \;\leftarrow\; T_\text{eq} - 50\cos\phi \cdot \cos(\omega t + \lambda)$$

(Amplitude drops to zero at poles; $50$ K is the equatorial diurnal half-amplitude.)

### 8.3 Exponential Relaxation (Newtonian Cooling)

> **Methods applied:** [Newton's law of cooling](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) · [Linearisation](https://en.wikipedia.org/wiki/Linearization) · [Exponential decay](https://en.wikipedia.org/wiki/Exponential_decay) · [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) (first-order)

$$\tau = \frac{C_\text{area}}{4\,\varepsilon\,\sigma\,T^3} \qquad \text{(thermal inertia timescale)}$$

$$T(t+\Delta t) = T_\text{eq} + (T - T_\text{eq})\,e^{-\Delta t/\tau}$$

$\tau$ is derived by linearising $Q_\text{out} = \varepsilon\sigma T^4$ around the current temperature:

$$\frac{\partial Q_\text{out}}{\partial T} = 4\varepsilon\sigma T^3, \qquad \tau = \frac{C_\text{area}}{4\varepsilon\sigma T^3}$$

The exact analytical solution to the linearised ODE $dT/dt = -(T - T_\text{eq})/\tau$ makes the
FAST path unconditionally stable for any $\Delta t$ — unlike RK4 which requires $\Delta t \ll \tau$.

**At $T = 210$ K** ($C_\text{area} = 6.0\times10^4$):

```
τ = 6.0×10⁴ / (4 × 0.95 × 5.670×10⁻⁸ × 210³)
  = 6.0×10⁴ / 1.992
  = 30 120 s  ≈  0.34 sols
```

The surface relaxes toward equilibrium with a timescale of about one-third of a sol.

### 8.4 RK4 vs Relaxation — Numerical Trade-offs

> **Methods applied:** [Runge-Kutta 4th order](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) · [Numerical stability](https://en.wikipedia.org/wiki/Numerical_stability) · [Stiff equation](https://en.wikipedia.org/wiki/Stiff_equation)

The efficiency gain of FAST is **not** fewer ops per call — it is **larger timestep size**:

| Constraint | ACCURATE (RK4) | FAST (relaxation) |
|-----------|---------------|------------------|
| Stability | Requires $\Delta t \ll \tau$ | Unconditionally stable |
| Min practical $\Delta t$ | ~900–3 600 s | ~1 sol (88 775 s) or larger |
| Calls per step | 4 × `compute_derivatives` | 1 × `compute_fast_physics` |

**Steps to simulate 100 Martian years:**

| Mode | $\Delta t$ | Steps | Derivative evals |
|------|-----|-------|-----------------|
| ACCURATE | 3 600 s | 1 648 800 | **6.6M** |
| FAST | 1 sol | 66 860 | 66 860 |
| FAST | 10 sols | 6 686 | 6 686 |
| FAST | 1 year | 100 | 100 |

For century-scale simulations the FAST path is **100–66 000× cheaper**.

**What accuracy is traded away:**

| | ACCURATE | FAST |
|--|---------|------|
| Insolation | Instantaneous $\cos z$ | Daily-mean $\bar{I}$ |
| Diurnal signal | Emerges from ODE | Synthetic $50\,\text{K}\cos\phi$ hardcoded |
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
│       └── I_bar = (h0·sin(φ)·sin(δ) + cos(φ)·cos(δ)·sin(h0)) / π
│                                      │
│                                      φ = _init_latitude (fixed)
│
├── [Step 2: Equilibrium temperature]
│       │
│       ├── absorbed = (1−α) · F · I_bar
│       │                   │   │
│       │                   │   └── radiation.solar_flux  ← advance_orbit
│       │                   └── radiation.albedo  (fixed)
│       │
│       ├── T_eq_base = (absorbed / (ε·σ))^0.25
│       ├── T_eq      = T_eq_base × greenhouse_factor     (greenhouse scaling)
│       │
│       └── T_eq -= 50·cos(φ) · cos(ω·t + λ)             (synthetic diurnal swing)
│
├── [Step 3: Exponential relaxation → thermal.surface_temperature]
│       │
│       ├── τ = C_area / (4·ε·σ·T_cur³)                  (linearised timescale)
│       │
│       └── T_new = T_eq + (T_cur − T_eq) · exp(−dt/τ)   (exact ODE solution)
│
├── [Step 4: Ice mass → water.ice_mass_north/south]
│       │
│       ├── cos_zenith_N = max(0, +sin(δ))
│       ├── cos_zenith_S = max(0, −sin(δ))
│       ├── Q_in_N/S = (1−α) · F · cos_zenith_N/S
│       ├── Q_out_pole = ε·σ·149⁴ = 26.51 W/m²   (fixed)
│       ├── dMice_N/S = −net_sub_N/S · dt         (per-pole Euler step)
│       └── Guard: clamp each pole independently if that pole's ice = 0
│
└── [Step 5: Pressure → atmosphere.surface_pressure]
        │
        ├── dP_escape      = −0.2 · g / (4πR²) · dt       (MAVEN constant)
        ├── dP_sublimation = −dMice · g / (4πR²)
        └── dP_tide        = −A·ω·sin(ω·t + φ) · dt       (thermal tide)
```

---

## 9. Full Dependency Graph

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
│       │           │   │                                 │
│       │           │   │              δ = arcsin(sin(ε)·sin(Ls))
│       │           │   │                                    │
│       │           │   │                         Ls = θ + 251°  ←  orbital_angle
│       │           │   │
│       │           │   └── F = radiation.solar_flux  ← advance_orbit
│       │           └── α = radiation.albedo  (fixed)
│       │
│       ├── Q_out = ε·σ·(T/f_gh)⁴
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
│       │                     │                   0.01·4πR²
│       │          (1−α)·F·max(0, ±sin(δ))
│       │
│       └── Guard: each pole clamped independently if that pole's ice = 0
│
│
└── [dP/dt]  = dP_escape + dP_sublimation + dP_tide
        │               │                       │
        │       −dM_ice/dt · g / (4πR²)         −A·ω·sin(ω·t + φ)
        │
        └── −0.2 kg/s · g / (4πR²)   (MAVEN empirical constant)
```

### ODE coupling summary

| Equation | Reads from other equations? |
|----------|-----------------------------|
| $dT/dt$ | No — depends on orbital state and own $T$ |
| $dM_\text{ice}/dt$ | No — depends on orbital state only |
| $dP/dt$ | **Yes** — depends on $dM_\text{ice}/dt$ |

Only one coupling: ice sublimation feeds directly into pressure.

---
## ODE Term Locations

| Term | System | File | Line |
|---|---|---|---|
| Q_in | Surface Energy Balance | mars.py | 285 |
| Q_out | Surface Energy Balance | mars.py | 287 |
| C_area | Surface Energy Balance | mars.py | 60 |
| dP_escape | Atmospheric Pressure | mars.py | 327 |
| dP_sub | Atmospheric Pressure | mars.py | 328 |
| dP_tide | Atmospheric Pressure | mars.py | 329 |
| A_cap | Polar CO2 Sublimation | mars.py | 294 |
| L_sub | Polar CO2 Sublimation | mars.py | 63 |
| M_sub_N | Polar CO2 Sublimation | mars.py | 302 |
| M_sub_S | Polar CO2 Sublimation | mars.py | 303 |
| dM_ice_N | Polar CO2 Sublimation | mars.py | 305 |
| dM_ice_S | Polar CO2 Sublimation | mars.py | 306 |
| dM_ice | Polor CO2 Sublimation | mars.py | 318 |

## 10. Model Scope and Known Approximations

| Approximation | Impact | Location |
|---------------|--------|----------|
| Mean anomaly used as true anomaly | ±10° $L_s$ error, ±5–10 sol timing offset on seasonal peaks | `advance_orbit` |
| $C_\text{area} = 6.0\times10^4$ constant everywhere | All surface points have Gale Crater's rocky-sandy thermal inertia | `MARS_THERMAL_INERTIA` |
| Polar $\cos z = \max(0, \pm\sin\delta)$ | Exact at 90° poles, approximate at cap edges | polar section |
| $A_\text{cap} = 1\%$ per pole (fixed) | Fixed sublimating area; no seasonal cap extent evolution | `MARS_POLAR_CAP_FRACTION` |
| Thermal tide empirical ($A=30$ Pa, $\varphi=-0.7\pi$) | Reproduces REMS diurnal $P$ oscillation but not spatial structure | `MARS_THERMAL_TIDE_PA/PHASE` |
| 0D global pressure | No spatial pressure gradient or local weather | $dP/dt$ formulation |
| $f_\text{gh} = 1.02$ fixed | Greenhouse does not evolve with pressure changes | `thermal.greenhouse_factor` |
| MAVEN escape rate constant | Does not vary with solar activity or season | `MARS_MAVEN_ESCAPE_RATE` |

---

## 11. Citations

### Physics Laws and Mathematical Methods

| Law / Method | Section | Reference |
|---|---|---|
| Kepler's First Law — $r(\theta) = a(1-e^2)/(1+e\cos\theta)$ | §3.2 | [Kepler's laws](https://en.wikipedia.org/wiki/Kepler%27s_laws_of_planetary_motion) |
| Kepler's Second Law — equal areas in equal times | §3.1 | [Mean motion](https://en.wikipedia.org/wiki/Mean_motion) |
| Mean motion — $d\theta/dt = 2\pi/T$ | §3.1 | [Mean anomaly](https://en.wikipedia.org/wiki/Mean_anomaly) · [True anomaly](https://en.wikipedia.org/wiki/True_anomaly) |
| Inverse-square law — $F = F_0(1\,\text{AU}/r)^2$ | §3.3 | [Inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law) |
| Solar declination — $\delta = \arcsin(\sin\varepsilon\sin L_s)$ | §4.2 | [Position of the Sun](https://en.wikipedia.org/wiki/Position_of_the_Sun) |
| Hour angle — $h = \omega t - \pi + \lambda$ | §4.3 | [Hour angle](https://en.wikipedia.org/wiki/Hour_angle) |
| Solar zenith angle — $\cos z = \sin\phi\sin\delta + \cos\phi\cos\delta\cos h$ | §4.4 | [Solar zenith angle](https://en.wikipedia.org/wiki/Solar_zenith_angle) |
| First Law of Thermodynamics — $C\,dT/dt = Q_\text{in} - Q_\text{out}$ | §5 | [First law of thermodynamics](https://en.wikipedia.org/wiki/First_law_of_thermodynamics) |
| Lambert's cosine law — $Q_\text{in} = (1-\alpha)F\cos z$ | §5.1 | [Lambert's cosine law](https://en.wikipedia.org/wiki/Lambert%27s_cosine_law) |
| Stefan-Boltzmann law — $Q_\text{out} = \varepsilon\sigma T^4$ | §5.2, §6.2 | [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) |
| Thermal inertia — $C_\text{area} = \rho\,c_p\,d$ | §5.3 | [Thermal inertia](https://en.wikipedia.org/wiki/Thermal_inertia) |
| Latent heat of sublimation — $\Delta E = L\,\Delta M$ | §6.2–6.3 | [Latent heat](https://en.wikipedia.org/wiki/Latent_heat) · [Sublimation](https://en.wikipedia.org/wiki/Sublimation_(phase_transition)) |
| Hydrostatic equilibrium — $P = Mg/A$ | §7.1 | [Hydrostatic equilibrium](https://en.wikipedia.org/wiki/Hydrostatic_equilibrium) |
| Atmospheric escape (non-thermal) | §7.2 | [Atmospheric escape](https://en.wikipedia.org/wiki/Atmospheric_escape) |
| Jeans escape parameter $\lambda = GMm/(k_B T R)$ | §7.2 | [Jeans escape](https://en.wikipedia.org/wiki/Atmospheric_escape#Jeans_escape) |
| Atmospheric thermal tide — $P_\text{tide} = A\cos(\omega t + \varphi)$ | §7.3 | [Atmospheric tide](https://en.wikipedia.org/wiki/Atmospheric_tide) |
| Sunrise hour angle — $\cos h_0 = -\tan\phi\tan\delta$ | §8.1 | [Sunrise equation](https://en.wikipedia.org/wiki/Sunrise_equation) |
| Radiative equilibrium — $T_\text{eq} = [(1-\alpha)F\bar{I}/(\varepsilon\sigma)]^{1/4}$ | §8.2 | [Radiative equilibrium](https://en.wikipedia.org/wiki/Radiative_equilibrium) |
| Newton's law of cooling / exponential relaxation | §8.3 | [Newton's law of cooling](https://en.wikipedia.org/wiki/Newton%27s_law_of_cooling) |
| Taylor linearisation — $\tau = C/(4\varepsilon\sigma T^3)$ | §8.3 | [Linearization](https://en.wikipedia.org/wiki/Linearization) · [Taylor series](https://en.wikipedia.org/wiki/Taylor_series) |
| 4th-order Runge-Kutta (ACCURATE path) | §1, §8.4 | [Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) |
| Forward Euler method (ice/pressure in FAST path) | §8.4 | [Euler method](https://en.wikipedia.org/wiki/Euler_method) |

### Scientific Literature

| Reference | Used for |
|-----------|---------|
| [NASA Mars Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html) | All planetary constants ($M$, $R$, $g$, $a$, $e$, $T_\text{orb}$, $\varepsilon_\text{tilt}$) |
| Kepler, J. (1609). *Astronomia Nova* | Orbital ellipse law |
| Lambert, J.H. (1760). *Photometria* | Lambert's cosine law — $Q = F\cos z$ |
| Stefan, J. (1879); Boltzmann, L. (1884) | Thermal emission $Q = \varepsilon\sigma T^4$ |
| Clausius, R. (1850). First Law of Thermodynamics | $dU/dt = Q_\text{in} - Q_\text{out}$ |
| [Jakosky, B.M. et al. (2018). *Science*, 355(6323).](https://doi.org/10.1126/science.aan5015) | MAVEN non-thermal escape rate ~0.2 kg s⁻¹ |
| Kieffer, H.H. et al. (1977). *JGR*, 82(28) | CO₂ frost point 149 K at Mars surface pressure |
| Smith, M.D. (2004). *Icarus*, 167(1), 148–165 | CO₂ latent heat $L_\text{sub} = 5.7\times10^5$ J kg⁻¹ |
| [Haberle, R.M. et al. (1993). *JGR*, 98(E2)](https://doi.org/10.1029/92JE02679) | Seasonal pressure swing driven by polar cap exchange |
| Vasavada, A.R. et al. (2017). *JGR Planets*, 122(5) | REMS Gale Crater temperature profiles, Sol 224 calibration |
| Christensen, P.R. et al. (2001). *JGR*, 106(E10). THEMIS | Thermal inertia maps — TI ≈ 200–350 TIU at Gale Crater |
| Wilson, R.J. & Hamilton, K. (1996). *JAS*, 53(9) | Martian atmospheric thermal tides — amplitude and phase |
