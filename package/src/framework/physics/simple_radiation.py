"""Released simple-radiation dust scattering and gray infrared driver.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp,scales
from src.framework.gcm.dynamics import reference_temperature
from src.framework.physics.gcm import RadiativeFluxDiagnostics,_declination,solar_flux
from src.framework.physics.solar_geometry import daylight_geometry,averaged_zenith,radiation_clock


class ScatteringResult(NamedTuple):
    heating_fraction: object
    surface_fraction: object
    outgoing_fraction: object
    up: object
    down: object


def layer_coefficients(optical_depth,single_scattering_albedo,asymmetry,zenith,direct):
    """Original ecoeffs, with exponentials divided by exp(lambda*tau).

    This algebra avoids overflow in thick layers. The source singularities at
    conservative scattering and lambda*zenith=1 remain explicit.
    """
    tau,w,g=map(jnp.asarray,(optical_depth,single_scattering_albedo,asymmetry))
    lam=jnp.sqrt(3.*(1.-w)*(1.-w*g))
    u=1.5*(1.-w*g)/lam
    denominator=1./(1.-(lam*zenith)**2)
    alpha=.75*zenith*w*(1.+g*(1.-w))*denominator
    gamma=.5*w*(1.+3.*g*(1.-w)*zenith**2)*denominator
    e=jnp.exp(-lam*tau)
    norm=(u+1.)**2-(u-1.)**2*e**2
    transmission=4.*u*e/norm
    reflection=(u+1.)*(u-1.)*(1.-e**2)/norm
    beam_transmission=(alpha+gamma)*transmission+((alpha-gamma)*reflection-(alpha+gamma-1.))*direct
    beam_reflection=(alpha-gamma)*transmission*direct+(alpha+gamma)*reflection-(alpha-gamma)
    return transmission,reflection,beam_transmission,beam_reflection


def scatter(optical_depth,single_scattering_albedo,asymmetry,forward_fraction,cosine,albedo):
    """Original scatter_calc, with levels first and arbitrary column axes."""
    tau,w,g,f=map(jnp.asarray,(optical_depth,single_scattering_albedo,asymmetry,forward_fraction))
    w=w*(1.-f)/(1.-w*f);g=(g-f)/(1.-f);tau=(1.-single_scattering_albedo*f)*tau
    cumulative=jnp.concatenate([jnp.zeros_like(tau[:1]),jnp.cumsum(tau,axis=0)],axis=0)
    cosine,albedo=jnp.broadcast_arrays(cosine,albedo)
    secant=35./jnp.sqrt(1224.*cosine**2+1.)
    day=cosine>0.
    direct=jnp.where(day,jnp.exp(-cumulative*secant),0.)
    layer_direct=jnp.where(day,jnp.exp(-tau*secant),0.)
    tr1,rf1,tr,rf=layer_coefficients(tau,w,g,1./secant,layer_direct)
    def upward(carry,fields):
        rup,rup1=carry;trans,reflection,beam,reflect,beam_direct=fields
        diffuse=beam-beam_direct
        denominator=1./(1.-reflection*rup1)
        next_rup1=reflection+trans*rup1*trans*denominator
        next_rup=reflect+trans*(diffuse*rup1+beam_direct*rup)*denominator
        return (next_rup,next_rup1),(next_rup,next_rup1)
    _,(rup,rup1)=jax.lax.scan(upward,(albedo,albedo),(tr1,rf1,tr,rf,layer_direct),reverse=True)
    rup=jnp.concatenate([rup,albedo[None]],axis=0)
    rup1=jnp.concatenate([rup1,albedo[None]],axis=0)
    def downward(carry,fields):
        tdn,rdn=carry;trans,reflection,beam,reflect,beam_at_top=fields
        diffuse=tdn-beam_at_top
        denominator=1./(1.-rdn*reflection)
        next_rdn=reflection+trans*rdn*trans*denominator
        next_tdn=beam_at_top*beam+trans*(diffuse+beam_at_top*rdn*reflect)*denominator
        return (next_tdn,next_rdn),(next_tdn,next_rdn)
    top=jnp.ones_like(albedo);zero=jnp.zeros_like(albedo)
    _,(tdn,rdn)=jax.lax.scan(downward,(top,zero),(tr1,rf1,tr,rf,direct[:-1]))
    tdn=jnp.concatenate([top[None],tdn],axis=0)
    rdn=jnp.concatenate([zero[None],rdn],axis=0)
    diffuse=tdn-direct;denominator=1./(1.-rdn*rup1)
    up=(direct*rup+diffuse*rup1)*denominator
    down=direct+(diffuse+direct*rup*rdn)*denominator
    up=jnp.where(day,up,0.);down=jnp.where(day,down,0.)
    net=up-down
    return ScatteringResult(jnp.diff(net,axis=0),-net[-1],up[0],up,down)


def dust_depth(pressure_half,pressure_full,latitude,optical_depth=.2,*,reference_pressure=610.,
               conrath=.001,latitude_variation=False):
    """Original dust_distrib_fixed, including normalization by surface pressure."""
    p,ph=map(jnp.asarray,(pressure_full,pressure_half))
    zmax=70.-40.*jnp.sin(latitude)**2 if latitude_variation else 35.
    opacity=optical_depth*(.3+.7*jnp.cos(latitude)) if latitude_variation else optical_depth
    profile=jnp.exp(conrath*(1.-(reference_pressure/p)**(70./zmax)))
    profile=jnp.where(p>reference_pressure,1.,profile)
    integral=jnp.cumsum(profile*jnp.diff(ph,axis=0),axis=0)
    normal=opacity*ph[-1]/(reference_pressure*integral[-1])
    cumulative=jnp.concatenate([jnp.zeros_like(integral[:1]),integral*normal],axis=0)
    return jnp.diff(cumulative,axis=0)


def infrared(temperature,surface_temperature,pressure_half,pressure_full,*,
             tau_infrared=1.,pressure_exponent=0.,reference_pressure=610.,
             stefan=5.6734e-8,gravity=3.72,cp=735.94,newton_damping=False):
    """Original lwheat. The released routine ignores its emissivity argument."""
    t,ph,p=map(jnp.asarray,(temperature,pressure_half,pressure_full))
    planck=stefan*t**4
    optical_depth=tau_infrared*(ph/reference_pressure)**(pressure_exponent+1.)
    transmission=jnp.exp(-jnp.diff(optical_depth,axis=0))
    def add(flux,fields):
        trans,emission=fields
        flux=flux*trans+emission*(1.-trans)
        return flux,flux
    surface=stefan*jnp.asarray(surface_temperature)**4
    _,up=jax.lax.scan(add,surface,(transmission,planck),reverse=True)
    _,down=jax.lax.scan(add,jnp.zeros_like(surface),(transmission,planck))
    up=jnp.concatenate([up,surface[None]],axis=0)
    down=jnp.concatenate([jnp.zeros_like(surface[None]),down],axis=0)
    heating=jnp.diff(up-down,axis=0)/jnp.diff(ph,axis=0)*gravity/cp
    if newton_damping:
        if t.shape[0]<4: raise ValueError('released infrared damping requires at least four levels')
        pnot=600.e2*.9447012000e-5
        damping=(jnp.log(p[:4])/jnp.log(pnot))**6/5.e4
        heating=heating.at[:4].add(-damping*(t[:4]-140.))
    return heating,up,down


def column_fluxes(temperature,surface_temperature,pressure_half,pressure_full,cosine,
                  solar_intensity,albedo,emissivity,latitude,sols,*,daylight_fraction=1.,
                  optical_depth=.2,single_scattering_albedo=.92,asymmetry=.55,
                  conrath=.001,latitude_variation=False,reference_pressure=610.,
                  tau_infrared=1.,pressure_exponent=0.,forget_shortwave_heating=False,
                  newton_damping=False,no_downward_infrared=False,
                  gravity=3.72,cp=735.94,stefan=5.6734e-8):
    """Released CUBE_CORE simple driver, including its five-sol SW startup ramp."""
    delp=jnp.diff(pressure_half,axis=0)
    tau=dust_depth(pressure_half,pressure_full,latitude,optical_depth,
        reference_pressure=reference_pressure,conrath=conrath,latitude_variation=latitude_variation)
    scattering=scatter(tau,single_scattering_albedo,asymmetry,asymmetry**2,cosine,albedo)
    intensity=solar_intensity*daylight_fraction*cosine
    ramp=jnp.where(sols<5.,.5*(1.-jnp.cos(sols*jnp.pi/5.)),1.)
    swup=scattering.up*intensity*ramp;swdown=scattering.down*intensity*ramp
    heating,irup,irdown=infrared(temperature,surface_temperature,pressure_half,pressure_full,
        tau_infrared=tau_infrared,pressure_exponent=pressure_exponent,reference_pressure=reference_pressure,
        stefan=stefan,gravity=gravity,cp=cp,newton_damping=newton_damping)
    capacity=cp*delp/gravity
    correction=heating-jnp.diff(irup-irdown,axis=0)/capacity
    if forget_shortwave_heating:
        argument=cosine*700./pressure_full
        root=jnp.where(argument>0.,jnp.sqrt(jnp.where(argument>0.,argument,1.)),0.)
        correction=correction+ramp*(1.3/88775.)*root/(1.+.0075/pressure_full)
    net=swup-swdown+irup-irdown
    # Source removes only the returned surface IR flux, without recomputing heating.
    if no_downward_infrared: irdown=irdown.at[-1].set(0.)
    return RadiativeFluxDiagnostics(swdown,swup,irup,irdown,jnp.diff(net,axis=0),
        -swup[-1]+swdown[-1]+irdown[-1]-irup[-1],-net[0],correction)


def make_radiation_component(*,radiation_interval_seconds=1800.,fixed_calendar_date=-1,
                             daily_mean=False,**options):
    """Drop-in callback for the released simple driver, evaluated every step."""
    if not np.isfinite(radiation_interval_seconds) or radiation_interval_seconds<=0.:
        raise ValueError('simple radiation interval must be finite and positive')
    if not isinstance(fixed_calendar_date,int): raise ValueError('fixed calendar date must be an integer')
    def component(state,coords,specs,body,forcing):
        grid=coords.horizontal
        pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
        time_scale=float(specs.dimensionalize(1.,scales.units.second).magnitude)
        pressure=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*pressure_scale
        temperature=grid.to_nodal(state.dynamics.temperature_variation)+jnp.asarray(
            reference_temperature(coords,body))[:,None,None]
        half=jnp.asarray(coords.vertical.boundaries)[:,None,None]*pressure
        full=jnp.asarray(coords.vertical.centers)[:,None,None]*pressure
        seconds=state.dynamics.sim_time*time_scale
        calendar_seconds,sols,day_fraction=radiation_clock(seconds,forcing.rotation_period_s,fixed_calendar_date)
        declination=_declination(calendar_seconds,forcing)
        latitude=jnp.asarray(grid.latitudes)[None,:]
        if daily_mean:
            _,cosine,fraction=daylight_geometry(declination,latitude)
        else:
            cosine,fraction=averaged_zenith(day_fraction,declination,
                jnp.asarray(grid.longitudes)[:,None],latitude,
                .5*jnp.pi*radiation_interval_seconds/forcing.rotation_period_s)
        return column_fluxes(temperature,state.surface_temperature[0],half,full,cosine,
            solar_flux(calendar_seconds,forcing),forcing.albedo,forcing.emissivity,latitude,sols,
            daylight_fraction=fraction,gravity=body.gravity_m_s2,cp=body.cp_j_kg_k,
            stefan=forcing.stefan_boltzmann,**options)
    return component
