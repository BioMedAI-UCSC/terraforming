"""Differentiable default-12-band Ames radiation from column state to fluxes.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
from src.framework.gcm._dinosaur import jnp,scales
from src.framework.gcm.dynamics import reference_temperature
from src.framework.physics.radiation_grid import radiation_grid
from src.framework.physics.optical_depth import gas_subgrid,aggregate_optics
from src.framework.physics.spectral_flux import visible_spectrum,infrared_spectrum
from src.framework.physics.gcm import RadiativeFluxDiagnostics,_declination,solar_flux
from src.framework.physics.correlated_k import radiation_tables_jax
from src.framework.physics.solar_geometry import daylight_geometry,averaged_zenith
from src.framework.physics.collision_absorption import collision_induced_depths
from src.framework.physics.nonlte_cooling import cooling,solar_heating_factor,infrared_blend


def rayleigh_per_pa(gravity_m_s2):
    """Original setspv Rayleigh formula, with pressure measured in Pa."""
    boundaries=jnp.array([2222.22,3087.37,4030.63,5370.57,7651.11,12500.,25000.,41666.67])
    wavelength=1.e4/(.5*(boundaries[1:]+boundaries[:-1]))
    return (8.7/gravity_m_s2)*1.527*(1.+.013/wavelength**2)/wavelength**4/9.423e6


def shortwave_ramp(sols,duration_sols):
    """Released radiation-driver cosine ramp; nonpositive duration disables it."""
    duration=jnp.asarray(duration_sols)
    safe=jnp.where(duration>0.,duration,1.)
    return jnp.where((duration>0.)&(sols<duration),.5*(1.-jnp.cos(sols*jnp.pi/safe)),1.)


def column_fluxes(temperature,surface_temperature,pressure_half,pressure_full,
                  mu0,solar_intensity,albedo,emissivity,gravity_m_s2,*,particle_optics=None,
                  daylight_fraction=1.,radiation_mode='12band',water_mass_fraction=None,
                  cia_molar_fractions=None,stratosphere_temperature=None,scale_surface_flux=False,
                  linear_k_interpolation=False):
    """Ames fillpt -> optical depths -> spectral fluxes -> layer heat flux.

    Inputs use kelvin, Pa and W/m2. solar_intensity has seven bands and includes
    the orbital distance factor. Optional particle_optics gives SW and IR tuples
    (extinction, scattering, asymmetry-weighted scattering) on every subinterval.
    Dust/cloud particle distributions and optical properties are caller inputs.
    """
    grid=radiation_grid(pressure_half,pressure_full,temperature,surface_temperature,stratosphere_temperature)
    water=None
    if water_mass_fraction is not None:
        water=jnp.broadcast_to(water_mass_fraction,jnp.shape(temperature))
        water=jnp.concatenate([jnp.zeros_like(water[:1]),jnp.zeros_like(water[:1]),
            jnp.repeat(water,2,axis=0)],axis=0)
    gases=gas_subgrid(grid,mode=radiation_mode,water_mass_fraction=water,
        linear_k_interpolation=linear_k_interpolation)
    if particle_optics is None:
        particle_optics=tuple((jnp.zeros(gas.shape[:2]+gas.shape[3:]),)*3 for gas in gases)
    if len(particle_optics)!=2:
        raise ValueError("radiation requires both visible and infrared particle optics")
    delta=jnp.diff(grid.pressure,axis=0)
    ray=rayleigh_per_pa(gravity_m_s2)
    ray=ray.reshape((1,7)+(1,)*(delta.ndim-1))*delta[:,None]
    ir_particles=particle_optics[1]
    if cia_molar_fractions is not None:
        if radiation_mode=='12band': raise ValueError('reference collision absorption requires 15-band radiation')
        cia=collision_induced_depths(grid.pressure,grid.temperature,
            co2_molar_fraction=cia_molar_fractions[0],h2_molar_fraction=cia_molar_fractions[1])
        # CIA belongs to every channel but must not change gas-channel merging.
        ir_particles=(ir_particles[0]+cia,*ir_particles[1:])
    sw=aggregate_optics(gases[0],*particle_optics[0],ray)
    ir=aggregate_optics(gases[1],*ir_particles,infrared=True)
    day=jnp.asarray(mu0)>=1.e-4
    safe_mu=jnp.where(day,mu0,1.)
    visible=visible_spectrum(*sw,safe_mu,solar_intensity,albedo,mode=radiation_mode)
    thermal=infrared_spectrum(*ir,grid.pressure,grid.temperature,1.-emissivity,
        mode=radiation_mode,scale_surface_flux=scale_surface_flux)
    swup=jnp.where(day,visible.up*daylight_fraction,0.)
    swdown=jnp.where(day,visible.down*daylight_fraction,0.)
    netup=swup-swdown+thermal.up-thermal.down
    convergence=jnp.diff(netup,axis=0)
    strat_ir=thermal.up[0]-thermal.down[0]-thermal.net_top
    strat_sw=jnp.where(day,(visible.up[0]-visible.down[0]-visible.net_top)*
        solar_heating_factor(grid.pressure[1]),0.)
    return RadiativeFluxDiagnostics(swdown,swup,thermal.up,thermal.down,
        convergence,-netup[-1],-netup[0],stratosphere_flux_convergence_w_m2=strat_ir+strat_sw)


def make_radiation_component(particle_provider,*,radiation_interval_seconds=1848.,
                             radiation_mode='12band',water_vapor_tracer=None,cia_molar_fractions=None,
                             solar_nonlte=True,nonlte_tracers=None,scale_surface_flux=False,
                             linear_k_interpolation=False,shortwave_ramp_sols=-1.):
    """Build the Ames backend callback with an explicit particle-input provider.

    The provider receives (state, coords, specs, body, forcing, radiation_grid)
    and returns both spectral particle tuples, or None for an explicitly clear
    experiment. Geometry follows the source's partial-day averaging and its
    optional daylight-mean cosine/fraction mode. Radiation is evaluated each
    Dinosaur step; the averaging interval remains an explicit reference input.
    """
    import math
    if not math.isfinite(radiation_interval_seconds) or radiation_interval_seconds<=0.:
        raise ValueError('radiation averaging interval must be finite and positive')
    if not math.isfinite(shortwave_ramp_sols):
        raise ValueError('shortwave ramp duration must be finite')
    if nonlte_tracers is not None and len(nonlte_tracers)!=2:
        raise ValueError('non-LTE cooling requires explicit CO2 and atomic oxygen tracer names')
    if linear_k_interpolation and radiation_mode!='12band':
        raise ValueError('linear k interpolation requires 12-band radiation')
    radiation_tables_jax(radiation_mode)
    def component(state,coords,specs,body,forcing):
        grid=coords.horizontal
        pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
        time_scale=float(specs.dimensionalize(1.,scales.units.second).magnitude)
        pressure=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*pressure_scale
        temperature=grid.to_nodal(state.dynamics.temperature_variation)+jnp.asarray(
            reference_temperature(coords,body))[:,None,None]
        half=jnp.asarray(coords.vertical.boundaries)[:,None,None]*pressure
        full=jnp.asarray(coords.vertical.centers)[:,None,None]*pressure
        surface=state.surface_temperature[0]
        fields=state.auxiliary_fields
        strat=(fields['radiation_stratosphere_k'][0] if fields is not None and
            'radiation_stratosphere_k' in fields else jnp.full_like(surface,170.))
        rad_grid=radiation_grid(half,full,temperature,surface,strat)
        particles=particle_provider(state,coords,specs,body,forcing,rad_grid)
        time=state.dynamics.sim_time*time_scale
        declination=_declination(time,forcing)
        latitude=jnp.asarray(grid.latitudes)[None,:]
        if forcing.diurnal:
            fraction=jnp.mod(time/forcing.rotation_period_s,1.)
            half_angle=.5*jnp.pi*radiation_interval_seconds/forcing.rotation_period_s
            mu,daylight=averaged_zenith(fraction,declination,
                jnp.asarray(grid.longitudes)[:,None],latitude,half_angle)
        else:
            _,mu,daylight=daylight_geometry(declination,latitude)
        solar=radiation_tables_jax(radiation_mode)['solar_weights']*solar_flux(time,forcing)
        water=(None if water_vapor_tracer is None else grid.to_nodal(state.dynamics.tracers[water_vapor_tracer]))
        flux=column_fluxes(temperature,surface,half,full,mu,solar,
            forcing.albedo,forcing.emissivity,body.gravity_m_s2,particle_optics=particles,
            daylight_fraction=daylight,radiation_mode=radiation_mode,water_mass_fraction=water,
            cia_molar_fractions=cia_molar_fractions,stratosphere_temperature=strat,
            scale_surface_flux=scale_surface_flux,linear_k_interpolation=linear_k_interpolation)
        correction=jnp.zeros_like(temperature)
        capacity=body.cp_j_kg_k*jnp.diff(half,axis=0)/body.gravity_m_s2
        if solar_nonlte or shortwave_ramp_sols>0.:
            sw_heating=jnp.diff(flux.shortwave_up_w_m2-flux.shortwave_down_w_m2,axis=0)/capacity
        if solar_nonlte:
            correction=correction+(solar_heating_factor(full)-1.)*sw_heating
        ramp=1.
        if shortwave_ramp_sols>0.:
            ramp=shortwave_ramp(time/forcing.rotation_period_s,shortwave_ramp_sols)
            correction=ramp*correction+(ramp-1.)*sw_heating
        if nonlte_tracers is not None:
            co2,oxygen=(grid.to_nodal(state.dynamics.tracers[name]) for name in nonlte_tracers)
            nonlte=cooling(temperature,full,co2,oxygen,radiation_pressure_pa=rad_grid.pressure)
            lw=jnp.diff(flux.longwave_up_w_m2-flux.longwave_down_w_m2,axis=0)/capacity
            correction=correction+infrared_blend(lw,nonlte,half[:-1])-lw
        return flux._replace(nonlte_heating_k_s=correction,shortwave_scale=ramp)
    return component
