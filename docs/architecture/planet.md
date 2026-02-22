# Planet Simulation Architecture

## High-Level Design (HLD)

### 1. System Overview

The terraforming simulation system models planetary properties and their evolution over time. The architecture uses an abstract `Planet` class as the foundation, with concrete implementations (e.g., `Mars`) that define planet-specific parameters.

### 2. Core Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Simulation Engine                         │
│  ├─ Time Controller (simulation speed, timestep)            │
│  └─ State Manager (snapshot, restore, history)              │
└─────────────────────────────────────────────────────────────┘
                        
┌─────────────────────────────────────────────────────────────┐
│                   Planet (Abstract Base)        
│  ├─ Inherent Time variable                 │
│  ├─ Physical Properties (mass, radius, orbit)               │
│  ├─ Environmental State (temp, pressure, atmosphere)         │
│  ├─ Planetary Systems (magnetic, radiation, wind)            │
│  └─ Temporal System (orbital timer, seasonal cycles)         │
└─────────────────────────────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
    ┌─────────┐       ┌─────────┐      ┌─────────┐
    │  Mars   │       │  Earth  │      │  Venus  │
    └─────────┘       └─────────┘      └─────────┘
```

### 3. Simulation Flow

```
Initialize Planet → Set Initial Conditions → Start Simulation Loop
                                                    │
                    ┌───────────────────────────────┘
                    ▼
            Update Orbital Timer (Δt)
                    │
                    ▼
        Calculate Solar Radiation Input
                    │
                    ▼
    ┌───────────────────────────────────┐
    │   Update All Subsystems (Δt)     │
    │   ├─ Atmospheric System           │
    │   ├─ Thermal System                │
    │   ├─ Magnetic Field System         │
    │   ├─ Radiation Environment         │
    │   └─ Wind/Circulation System       │
    └───────────────────────────────────┘
                    │
                    ▼
        Update Planetary State
                    │
                    ▼
        Record/Display Results
                    │
                    ▼
        Check End Condition → Continue/Stop
```

### 4. Key Systems and Properties

#### 4.1 Core Physical Properties (Immutable/Slowly Changing)
- **Mass** (M): kg
- **Radius** (R): m
- **Gravity** (g): m/s² = GM/R²
- **Orbital Parameters**: semi-major axis, eccentricity, period
- **Axial Tilt**: degrees

#### 4.2 Environmental State Variables (Dynamic)
- **Surface Temperature** (T_surf): K
- **Atmospheric Pressure** (P_atm): Pa
- **Atmospheric Composition**: {CO₂, N₂, O₂, H₂O, Ar, ...} (partial pressures)
- **Water Inventory**: ice mass, liquid mass, vapor mass (kg)
- **Albedo** (α): 0-1

#### 4.3 Planetary Systems
- **Magnetic Field System**
  - Field strength (B): Tesla
  - Magnetosphere boundary: R_magnetopause
- **Radiation Environment**
  - Solar radiation flux (at current orbital distance): W/m²
  - Cosmic ray flux
  - Surface UV intensity
- **Atmospheric Circulation**
  - Wind patterns (velocity fields)
  - Heat transport efficiency
- **Thermal Balance**
  - Incoming solar radiation
  - Outgoing thermal radiation
  - Greenhouse effect

### 5. Time System

#### 5.1 Orbital Timer
- **Simulation Time** (t_sim): Real elapsed time in the simulation (seconds)
- **Planetary Year**: Based on orbital period
- **Sol/Day**: Based on rotation period
- **Orbital Position**: θ (angle around sun, 0-2π)

#### 5.2 Time Control
- **Time Scale Factor** (speed): Ratio of simulation time to wall-clock time
  - Example: speed=1000 → 1 second real-time = 1000 seconds sim-time
- **Timestep** (Δt): Integration timestep for numerical solvers
- **Adaptive timestep**: Adjust Δt based on rate of change

---

## Low-Level Design (LLD)

<details>
<summary><strong>§1. Class Structure</strong></summary>


```python
# Base Classes

class Planet(ABC):
    """Abstract base class for planetary bodies"""
    
    # Core Properties
    mass: float                    # kg
    radius: float                  # m
    rotation_period: float         # seconds
    
    # Orbital Properties
    orbital_params: OrbitalParameters
    
    # State Variables
    state: PlanetaryState
    
    # Systems
    atmosphere_system: AtmosphereSystem
    thermal_system: ThermalSystem
    magnetic_system: MagneticFieldSystem
    radiation_system: RadiationSystem
    wind_system: WindSystem
    
    # Time
    orbital_timer: OrbitalTimer
    
    @abstractmethod
    def initialize_state(self) -> PlanetaryState:
        """Set initial conditions for the planet"""
        pass
    
    @abstractmethod
    def update(self, dt: float) -> None:
        """Update all systems for timestep dt"""
        pass


class Mars(Planet):
    """Mars-specific implementation"""
    
    def initialize_state(self) -> PlanetaryState:
        """Initialize Mars with current conditions"""
        # Set Mars-specific initial values
        pass


# Supporting Classes

class OrbitalParameters:
    semi_major_axis: float         # m
    eccentricity: float            # 0-1
    orbital_period: float          # seconds
    axial_tilt: float              # radians
    
    def distance_from_sun(self, theta: float) -> float:
        """Calculate distance at orbital angle theta"""
        pass


class OrbitalTimer:
    """Manages simulation time and orbital position"""
    
    current_time: float            # simulation seconds since epoch
    orbital_angle: float           # radians (0-2π)
    time_scale: float              # simulation speed multiplier
    
    def advance(self, dt: float) -> None:
        """Advance time by dt seconds"""
        pass
    
    def get_solar_distance(self, orbital_params: OrbitalParameters) -> float:
        """Current distance from sun"""
        pass
    
    def get_day_of_year(self, orbital_period: float) -> float:
        """Day within current orbital year"""
        pass


class PlanetaryState:
    """Complete state snapshot at a given time"""
    
    # Atmospheric
    surface_pressure: float        # Pa
    atmospheric_mass: float        # kg
    composition: dict[str, float]  # gas name → partial pressure (Pa)
    
    # Thermal
    surface_temperature: float     # K
    subsurface_temp_profile: np.ndarray  # Temperature vs depth
    
    # Water
    ice_mass: float                # kg
    liquid_mass: float             # kg
    vapor_mass: float              # kg
    
    # Radiation
    albedo: float                  # 0-1
    greenhouse_factor: float       # dimensionless
    
    # Magnetic
    magnetic_field_strength: float # Tesla at surface
    
    # Wind
    wind_velocity_field: np.ndarray  # 3D velocity field (if spatial)
    
    def copy(self) -> 'PlanetaryState':
        """Deep copy of state"""
        pass
```

</details>

<details>
<summary><strong>§2. System Classes</strong></summary>


```python
class AtmosphereSystem:
    """Models atmospheric composition and pressure"""
    
    def __init__(self, planet: Planet):
        self.planet = planet
    
    def update(self, dt: float, state: PlanetaryState, 
               solar_flux: float) -> PlanetaryState:
        """
        Update atmospheric state
        
        Processes:
        - Atmospheric escape (Jeans escape, thermal escape)
        - Photochemistry
        - Gas addition (outgassing, impacts)
        - Mass balance
        """
        # Calculate escape rates
        escape_rates = self.calculate_escape(state, solar_flux)
        
        # Update composition
        new_composition = self.update_composition(
            state.composition, escape_rates, dt
        )
        
        # Update total mass and pressure
        new_mass = sum(new_composition.values()) * molecular_weight
        new_pressure = self.calculate_pressure(new_mass, state.surface_temperature)
        
        state.atmospheric_mass = new_mass
        state.surface_pressure = new_pressure
        state.composition = new_composition
        
        return state
    
    def calculate_escape(self, state: PlanetaryState, 
                        solar_flux: float) -> dict[str, float]:
        """
        Calculate atmospheric escape rates (kg/s per species)
        
        Uses Jeans escape formula: Φ = n(R) × v̄ × exp(-λ)
        where λ = GMm/(kTR) is the escape parameter
        
        Reference: https://en.wikipedia.org/wiki/Atmospheric_escape
        """
        # Jeans escape: based on molecular mass, temperature, gravity
        pass


class ThermalSystem:
    """Models planetary heat balance"""
    
    def __init__(self, planet: Planet):
        self.planet = planet
    
    def update(self, dt: float, state: PlanetaryState,
               solar_flux: float) -> PlanetaryState:
        """
        Update thermal state using energy balance:
        
        dE/dt = Q_solar_absorbed - Q_thermal_emitted + Q_internal
        
        Q_solar = (1 - α) * F_solar * π * R²
        Q_thermal = ε * σ * T⁴ * 4π * R²
        
        Greenhouse effect modifies effective emission temperature
        """
        # Incoming solar radiation
        Q_in = (1 - state.albedo) * solar_flux * np.pi * self.planet.radius**2
        
        # Outgoing thermal radiation (Stefan-Boltzmann)
        sigma = 5.67e-8  # W/(m²·K⁴)
        T_eff = state.surface_temperature / state.greenhouse_factor
        Q_out = sigma * T_eff**4 * 4 * np.pi * self.planet.radius**2
        
        # Internal heat (radioactive decay, tidal heating)
        Q_internal = self.calculate_internal_heat()
        
        # Net energy change
        dE = (Q_in - Q_out + Q_internal) * dt
        
        # Convert to temperature change
        # Heat capacity of surface + atmosphere
        C = self.effective_heat_capacity(state)
        dT = dE / C
        
        state.surface_temperature += dT
        
        # Update greenhouse factor based on composition
        state.greenhouse_factor = self.calculate_greenhouse(state.composition)
        
        return state
    
    def calculate_greenhouse(self, composition: dict[str, float]) -> float:
        """Calculate greenhouse warming factor from atmospheric composition"""
        # Based on CO₂, H₂O, CH₄ concentrations
        pass


class MagneticFieldSystem:
    """Models planetary magnetic field"""
    
    def __init__(self, planet: Planet):
        self.planet = planet
    
    def update(self, dt: float, state: PlanetaryState) -> PlanetaryState:
        """
        Update magnetic field strength
        
        Simplified model: field strength depends on:
        - Core temperature and composition
        - Rotation rate
        - Convection in liquid core
        
        For Mars: likely inactive dynamo (static weak field)
        """
        # Simplified: assume quasi-static or slowly decaying
        # Could model core cooling, rotation slowdown
        
        decay_rate = 1e-18  # T/s (very slow)
        state.magnetic_field_strength *= (1 - decay_rate * dt)
        
        return state
    
    def get_magnetopause_distance(self, state: PlanetaryState,
                                  solar_wind_pressure: float) -> float:
        """
        Calculate magnetosphere boundary (if present)
        
        Pressure balance: R_mp ∝ (B²/(μ₀ρ_sw V_sw²))^(1/6)
        
        Reference: https://en.wikipedia.org/wiki/Magnetopause
        """
        pass


class RadiationSystem:
    """Models radiation environment"""
    
    def __init__(self, planet: Planet):
        self.planet = planet
        self.solar_constant_1AU = 1361  # W/m² at 1 AU
    
    def calculate_solar_flux(self, distance_from_sun: float) -> float:
        """
        Solar flux at planet's current orbital position
        
        F = F₀ * (1 AU / d)²
        """
        AU = 1.496e11  # meters
        flux = self.solar_constant_1AU * (AU / distance_from_sun)**2
        return flux
    
    def calculate_surface_uv(self, state: PlanetaryState,
                            solar_flux: float) -> float:
        """
        UV radiation at surface
        
        Uses Beer-Lambert Law: I = I₀ × exp(-τ)
        where τ is optical depth from O₂ and O₃ absorption
        
        UV-C (<280nm): Absorbed by O₂
        UV-B (280-315nm): Absorbed by O₃
        UV-A (>315nm): Mostly transmitted
        
        Reference: https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law
        """
        # Atmospheric shielding
        O2_column = state.composition.get('O2', 0) * self.column_density_factor
        O3_column = state.composition.get('O3', 0) * self.column_density_factor
        
        attenuation = np.exp(-O2_column / 1e24 - O3_column / 1e22)
        
        uv_flux = solar_flux * 0.08 * attenuation  # ~8% of solar is UV
        return uv_flux
    
    def calculate_cosmic_ray_flux(self, state: PlanetaryState) -> float:
        """
        Cosmic ray flux at surface
        
        Shielded by:
        1. Magnetic field - deflects charged particles (stronger at equator)
        2. Atmosphere - absorbs/scatters particles (exponential attenuation)
        
        Reference: https://en.wikipedia.org/wiki/Cosmic_ray
        """
        # Magnetic shielding factor (Lorentz force deflection)
        mag_shield = np.exp(-state.magnetic_field_strength / 1e-5)
        
        # Atmospheric shielding (secondary particle cascade)
        atm_shield = np.exp(-state.surface_pressure / 1e4)
        
        baseline_flux = 100  # particles/(m²·s) in interplanetary space
        surface_flux = baseline_flux * mag_shield * atm_shield
        
        return surface_flux


class WindSystem:
    """Models atmospheric circulation"""
    
    def __init__(self, planet: Planet):
        self.planet = planet
    
    def update(self, dt: float, state: PlanetaryState) -> PlanetaryState:
        """
        Update wind patterns
        
        Simplified: global heat redistribution
        More complex: solve Navier-Stokes on sphere
        
        For terraforming: heat transport efficiency parameter
        """
        # Simplified approach: effective heat redistribution
        # Reduces day-night and equator-pole temperature gradients
        
        # Full model would solve atmospheric GCM
        pass
```

</details>

<details>
<summary><strong>§3. System of Equations Summary</strong></summary>


#### 3.1 Atmospheric Mass Balance
```
dM_atm/dt = Ṁ_outgassing + Ṁ_impacts - Ṁ_escape - Ṁ_sequestration
```

Where:
- **Ṁ_escape**: Jeans escape = n * v_th * A_exo * exp(-λ), λ = GMm/(kTR_exo)
- **Ṁ_outgassing**: Volcanic/tectonic release
- **Ṁ_sequestration**: Chemical weathering, polar caps

**Reference**: [Atmospheric Escape - Wikipedia](https://en.wikipedia.org/wiki/Atmospheric_escape)

#### 3.2 Energy Balance (Stefan-Boltzmann Law)
```
C dT/dt = (1 - α) F_solar π R² - ε σ (T/f_gh)⁴ 4π R² + Q_int
```

Where:
- **C**: Heat capacity (atmosphere + surface)
- **α**: Albedo
- **F_solar**: Solar flux at current distance
- **f_gh**: Greenhouse enhancement factor
- **Q_int**: Internal heat sources
- **σ**: Stefan-Boltzmann constant = 5.67 × 10⁻⁸ W/(m²·K⁴)

**References**: 
- [Stefan-Boltzmann Law - Wikipedia](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)
- [Planetary Energy Balance - UCAR](https://scied.ucar.edu/learning-zone/how-climate-works/energy-balance)

#### 3.3 Pressure Calculation (Barometric Formula)
```
P_surf = (ρ_atm g H) = (M_atm g)/(4π R²H)
```

Where H is scale height: **H = kT/(μg)**

**Reference**: [Barometric Formula - Wikipedia](https://en.wikipedia.org/wiki/Barometric_formula)

#### 3.4 Orbital Position (Kepler's Elliptical Orbit)
```
θ(t) = θ₀ + 2π * t / T_orbital
r(θ) = a(1 - e²)/(1 + e cos(θ))
```

Where:
- **a**: Semi-major axis
- **e**: Eccentricity (0 < e < 1 for ellipse)
- **θ**: True anomaly (orbital angle)

**Reference**: [Kepler Orbit - Wikipedia](https://en.wikipedia.org/wiki/Kepler_orbit)

</details>

<details>
<summary><strong>§4. Implementation Specifications</strong></summary>

#### 4.1 Time Integration
- Use **4th-order Runge-Kutta (RK4)** or **adaptive timestep solver** (e.g., Dormand-Prince)
- Typical timestep: 1 hour to 1 day (simulation time)
- Adjust timestep based on maximum rate of change

#### 4.2 Simulation Speed Control
```python
class SimulationEngine:
    def __init__(self, planet: Planet, time_scale: float = 1.0):
        self.planet = planet
        self.time_scale = time_scale  # sim_seconds per real_second
        self.dt = 3600  # 1 hour timestep (simulation time)
    
    def run_for_duration(self, duration: float, callback=None):
        """Run simulation for duration (simulation seconds)"""
        elapsed = 0
        while elapsed < duration:
            # Update orbital position
            self.planet.orbital_timer.advance(self.dt)
            
            # Calculate solar flux
            distance = self.planet.orbital_timer.get_solar_distance(
                self.planet.orbital_params
            )
            solar_flux = self.planet.radiation_system.calculate_solar_flux(distance)
            
            # Update all systems
            self.planet.update(self.dt)
            
            elapsed += self.dt
            
            if callback:
                callback(self.planet.state, elapsed)
    
    def set_speed(self, time_scale: float):
        """Adjust simulation speed (time_scale = sim_time/real_time)"""
        self.time_scale = time_scale
```

#### 4.3 State Persistence
```python
# Save/load state for checkpointing
def save_state(planet: Planet, filename: str):
    """Serialize planet state to file"""
    state_dict = {
        'time': planet.orbital_timer.current_time,
        'orbital_angle': planet.orbital_timer.orbital_angle,
        'state': asdict(planet.state),
    }
    with open(filename, 'wb') as f:
        pickle.dump(state_dict, f)

def load_state(planet: Planet, filename: str):
    """Restore planet state from file"""
    with open(filename, 'rb') as f:
        state_dict = pickle.load(f)
    planet.orbital_timer.current_time = state_dict['time']
    planet.orbital_timer.orbital_angle = state_dict['orbital_angle']
    planet.state = PlanetaryState(**state_dict['state'])
```


</details>

<details>
<summary><strong>§5. Mars-Specific Parameters</strong></summary>


```python
class Mars(Planet):
    def __init__(self):
        # Physical constants
        self.mass = 6.39e23  # kg
        self.radius = 3.3895e6  # m
        self.rotation_period = 88775.244  # seconds (24.6 hours)
        
        # Orbital parameters
        self.orbital_params = OrbitalParameters(
            semi_major_axis=2.279e11,  # m (1.524 AU)
            eccentricity=0.0934,
            orbital_period=5.935e7,  # seconds (687 days)
            axial_tilt=0.4396,  # radians (25.19°)
        )
        
        # Initialize systems
        self.atmosphere_system = AtmosphereSystem(self)
        self.thermal_system = ThermalSystem(self)
        self.magnetic_system = MagneticFieldSystem(self)
        self.radiation_system = RadiationSystem(self)
        self.wind_system = WindSystem(self)
        
        # Orbital timer
        self.orbital_timer = OrbitalTimer()
        
    def initialize_state(self) -> PlanetaryState:
        """Current Mars state"""
        return PlanetaryState(
            # Atmosphere (current)
            surface_pressure=610,  # Pa (0.6% Earth)
            atmospheric_mass=2.5e16,  # kg
            composition={
                'CO2': 580,  # Pa
                'N2': 15,
                'Ar': 12,
                'O2': 0.8,
                'CO': 0.4,
            },
            
            # Thermal
            surface_temperature=210,  # K (-63°C average)
            subsurface_temp_profile=np.array([]),  # TBD
            
            # Water
            ice_mass=5e15,  # kg (polar caps + permafrost)
            liquid_mass=0,
            vapor_mass=1e13,  # kg (atmospheric water vapor)
            
            # Radiation
            albedo=0.25,
            greenhouse_factor=1.02,  # Minimal greenhouse (thin CO₂)
            
            # Magnetic (remnant crustal fields)
            magnetic_field_strength=5e-9,  # Tesla (very weak)
            
            # Wind (placeholder)
            wind_velocity_field=np.array([]),
        )
```

</details>

<details>
<summary><strong>§6. Usage Example</strong></summary>


```python
# Create Mars instance
mars = Mars()

# Create simulation engine (1000x speed: 1 real sec = 1000 sim sec)
sim = SimulationEngine(mars, time_scale=1000)

# Run for 100 Martian years
mars_year = mars.orbital_params.orbital_period
sim.run_for_duration(
    duration=100 * mars_year,
    callback=lambda state, t: print(f"Year {t/mars_year:.1f}: T={state.surface_temperature:.1f}K, P={state.surface_pressure:.1f}Pa")
)

# Save final state
save_state(mars, 'mars_100years.pkl')
```

</details>

---

## Summary

### HLD Key Points
1. **Abstract Planet class** with concrete implementations (Mars, etc.)
2. **Modular system architecture**: Atmosphere, Thermal, Magnetic, Radiation, Wind
3. **Time-dependent simulation** with orbital timer and adjustable speed
4. **State machine** capturing all planetary properties

### LLD Key Points
1. **Clear class hierarchy** with separation of concerns
2. **Physics-based equations** for each subsystem
3. **Numerical integration** with adaptive timestep
4. **State persistence** for checkpointing
5. **Mars-specific parameters** as reference implementation

This architecture provides a solid foundation for simulating planetary terraforming with realistic physical models.

---

## References and Equation Sources

### Core Physics Equations

#### 1. Atmospheric Escape (Jeans Escape)
- **Wikipedia - Atmospheric Escape**: [https://en.wikipedia.org/wiki/Atmospheric_escape](https://en.wikipedia.org/wiki/Atmospheric_escape)
- Describes thermal escape mechanisms for atmospheric gases
- Jeans escape formula: Φ = n(R) × v̄ × exp(-λ), where λ = (GMm)/(kTR)
- **Additional Resources**:
  - Paris Observatory - Atmospheric Escape Theory: [https://lesia.obspm.fr](https://lesia.obspm.fr)
  - Trinity University - Jeans Escape Rate Calculations

#### 2. Stefan-Boltzmann Law & Planetary Energy Balance
- **Wikipedia - Stefan-Boltzmann Law**: [https://en.wikipedia.org/wiki/Stefan–Boltzmann_law](https://en.wikipedia.org/wiki/Stefan–Boltzmann_law)
- Fundamental law: E = σT⁴, where σ = 5.67 × 10⁻⁸ W/(m²·K⁴)
- **UCAR - Planetary Energy Balance**: [https://scied.ucar.edu/learning-zone/how-climate-works/energy-balance](https://scied.ucar.edu/learning-zone/how-climate-works/energy-balance)
- Explains incoming solar radiation vs outgoing thermal radiation
- **Additional Resources**:
  - Rutgers University - Energy Balance Models
  - University of Michigan - Climate Physics

#### 3. Barometric Formula & Scale Height
- **Wikipedia - Barometric Formula**: [https://en.wikipedia.org/wiki/Barometric_formula](https://en.wikipedia.org/wiki/Barometric_formula)
- Describes atmospheric pressure variation with altitude
- Scale height: H = RT/(Mg) or H = kT/(μg)
- Isothermal atmosphere: P(h) = P₀ exp(-h/H)
- Non-isothermal: More complex formulation with temperature lapse rate

#### 4. Orbital Mechanics (Kepler's Laws)
- **Wikipedia - Kepler Orbit**: [https://en.wikipedia.org/wiki/Kepler_orbit](https://en.wikipedia.org/wiki/Kepler_orbit)
- Elliptical orbit equation: r(θ) = a(1 - e²)/(1 + e cos θ)
- Kepler's equation: M = E - e sin(E)
- **University of Virginia - Orbital Mechanics**: Teaching resources on elliptical orbits
- **Orbital Mechanics Space**: [https://orbital-mechanics.space](https://orbital-mechanics.space) - Comprehensive orbital dynamics

#### 5. Greenhouse Effect & Radiative Forcing
- **Wikipedia - Greenhouse Effect**: [https://en.wikipedia.org/wiki/Greenhouse_effect](https://en.wikipedia.org/wiki/Greenhouse_effect)
- Natural warming process: GHGs absorb and re-emit longwave radiation
- Earth without greenhouse effect: ~-18°C; with effect: ~15°C
- **Wikipedia - Radiative Forcing**: [https://en.wikipedia.org/wiki/Radiative_forcing](https://en.wikipedia.org/wiki/Radiative_forcing)
- Quantifies energy imbalance: measured in W/m²
- Positive forcing = warming, negative forcing = cooling
- **NASA - Greenhouse Effect**: [https://climate.nasa.gov](https://climate.nasa.gov) - Overview and current measurements
- **Additional Resources**:
  - UCAR - How Greenhouse Gases Work
  - ESA - Planetary Greenhouse Effects (Earth, Venus, Mars comparison)

#### 6. Magnetopause Standoff Distance
- **Wikipedia - Magnetopause**: [https://en.wikipedia.org/wiki/Magnetopause](https://en.wikipedia.org/wiki/Magnetopause)
- Boundary between planetary magnetic field and solar wind
- Pressure balance equation: R_mp ∝ (B²/(μ₀ρ_sw V_sw²))^(1/6)
- Solar wind dynamic pressure controls location
- **Imperial College London - Magnetosphere Physics**: Detailed magnetopause models
- **UCLA - Space Physics**: Empirical magnetopause models

#### 7. UV Attenuation (Beer-Lambert Law)
- **Wikipedia - Beer-Lambert Law**: [https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law](https://en.wikipedia.org/wiki/Beer%E2%80%93Lambert_law)
- Absorption law: A = εlc or I = I₀ exp(-εlc)
- **Wikipedia - Ultraviolet**: [https://en.wikipedia.org/wiki/Ultraviolet](https://en.wikipedia.org/wiki/Ultraviolet)
- UV-C (100-280 nm): Absorbed by O₂
- UV-B (280-315 nm): Absorbed by O₃ (ozone layer)
- UV-A (315-400 nm): Largely reaches surface
- **NIH - Ozone and UV Absorption**: Mechanisms of atmospheric UV protection
- **UNEP - Ozone Layer**: Environmental effects of ozone depletion

#### 8. Cosmic Ray Flux & Shielding
- **Wikipedia - Cosmic Rays**: [https://en.wikipedia.org/wiki/Cosmic_ray](https://en.wikipedia.org/wiki/Cosmic_ray)
- Primary cosmic rays interact with atmosphere to produce secondary particles
- **EPA - Cosmic Radiation**: [https://www.epa.gov](https://www.epa.gov) - Radiation exposure information
- Flux increases with altitude (less atmospheric shielding)
- Flux increases at poles (weaker magnetic shielding)
- **Magnetic Field Shielding**:
  - Magnetosphere deflects charged particles via Lorentz force
  - Solar wind modulation: Higher solar activity → lower cosmic ray flux
- **Atmospheric Shielding**:
  - Atmosphere absorbs/scatters incoming cosmic rays
  - Secondary particle showers created by interactions with air nuclei

### Planetary Data Sources

#### Mars Physical Parameters
- **NASA Mars Fact Sheet**: [https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html)
  - Mass: 6.39 × 10²³ kg
  - Radius: 3,389.5 km
  - Orbital period: 687 Earth days
  - Surface pressure: ~600 Pa (0.6% Earth)
  - Surface temperature: 210 K average
  - Atmospheric composition: 95% CO₂, 3% N₂, 1.6% Ar

#### General Planetary Science
- **NASA Planetary Data System**: [https://pds.nasa.gov](https://pds.nasa.gov)
- **JPL Solar System Dynamics**: [https://ssd.jpl.nasa.gov](https://ssd.jpl.nasa.gov)
- **ESA Planetary Science Archive**: [https://www.cosmos.esa.int/web/psa](https://www.cosmos.esa.int/web/psa)

### Numerical Methods

#### Runge-Kutta Integration
- **Wikipedia - Runge-Kutta Methods**: [https://en.wikipedia.org/wiki/Runge–Kutta_methods](https://en.wikipedia.org/wiki/Runge–Kutta_methods)
- Fourth-order RK4: Standard for ODE integration
- Adaptive timestep methods: Dormand-Prince (DOPRI5)

### Terraforming Literature
- McKay, C. P., Toon, O. B., & Kasting, J. F. (1991). "Making Mars habitable." *Nature*, 352(6335), 489-496.
- Zubrin, R. M., & McKay, C. P. (1997). "Technological requirements for terraforming Mars." *Journal of the British Interplanetary Society*, 50, 83-92.
- Fogg, M. J. (1995). *Terraforming: Engineering Planetary Environments*. SAE International.

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-16  
**Status**: Design Specification
