"""Reference wind-stress and convective dust lifting on level-first arrays.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jnp


def stress_flux(stress,threshold,density,latitude_rad,snow,frost,opacity,*,scheme=1,
                efficiency=.0065,gravity=3.72,north_snow_threshold=1.,south_snow_threshold=1.,
                frost_threshold=1.,opacity_threshold=100.):
    """Original flux_stress in kg/m2/s; keep strict source mask boundaries."""
    stress,threshold,rho,lat,snow,frost,tau=map(jnp.asarray,
        (stress,threshold,density,latitude_rad,snow,frost,opacity))
    if scheme in (2,3,4):
        active=stress>=threshold
        safe_stress=jnp.where(active,stress,1.)
        flux=efficiency/gravity*jnp.sqrt(safe_stress**3/rho)
        if scheme!=3:
            ratio=jnp.sqrt(threshold/safe_stress)
            flux=flux*(1.-ratio)*(1.+ratio)**2*(2.61 if scheme==4 else 1.)
    else:
        flux=efficiency*(2.3e-3 if scheme==5 else 2.43e-3)*stress**2*(stress-threshold)/threshold
    blocked=(stress<threshold)|((snow>north_snow_threshold)&(lat>0.))|(
        (snow>south_snow_threshold)&(lat<0.))|(frost>frost_threshold)|(tau>opacity_threshold)
    return jnp.where(blocked,0.,jnp.maximum(flux,0.))


def devil_flux(temperature_bottom,temperature_surface,pressure_bottom,pressure_pbl,pbl_top_layer,
               bottom_layer_index,sensible_heat_flux,latitude_rad,snow,frost,*,dda=True,dth=False,
               fv3=False,heat_flux_threshold=0.,dda_efficiency=1.4e-10,dth_efficiency=.5,
               grain_diameter_m=6e-6,fv3_source=.3,fv3_radius_m=2.5e-6,
               temperature_threshold=22.,north_snow_threshold=1.,south_snow_threshold=1.,
               frost_threshold=1.,gas_constant=189.02,cp=735.94,gravity=3.72):
    """Original three dust-devil flux choices, including their literal units."""
    t,ts,p,pp,top,heat,lat,snow,frost=map(jnp.asarray,(temperature_bottom,temperature_surface,
        pressure_bottom,pressure_pbl,pbl_top_layer,sensible_heat_flux,latitude_rad,snow,frost))
    xsi=gas_constant/cp+1.
    flux=jnp.zeros_like(t)
    if dda or dth:
        # (1-x**a)/(1-x) has the finite limit a at a collapsed PBL.
        ratio=pp/p;collapsed=ratio==1.
        safe=jnp.where(collapsed,.5,ratio)
        b=-jnp.expm1(xsi*jnp.log(safe))/((1.-safe)*xsi*p)
        b=jnp.where(collapsed,1./p,b)
    if dda:
        flux=flux+jnp.where((heat>heat_flux_threshold)&(top<bottom_layer_index),
            dda_efficiency*(1.-b)*heat,0.)
    if dth:
        rho=p/(gas_constant*t)
        limit=jnp.sqrt(1.+15./(2500.*gravity*grain_diameter_m))*jnp.sqrt(2500.*gravity*grain_diameter_m/rho)
        eta=1.-b;eta_h=(t-ts)/ts
        dp=p*(-jnp.expm1(.5*eta/(.5*eta-1.)*(eta_h/(xsi-1.))))
        wind=dp/rho
        flux=flux+jnp.where(wind>=limit,dth_efficiency*(rho*wind**2-15.)/gravity,0.)
    if fv3:
        scaling=(gravity/600.)*1.33*(2500./2.5)*1e-6
        amount=fv3_source*scaling*(fv3_radius_m/1e-6)**2*1.2e-2/200.
        flux=flux+jnp.where(ts-t<temperature_threshold,0.,amount)
    blocked=((snow>north_snow_threshold)&(lat>0.))|((snow>south_snow_threshold)&(lat<0.))|(frost>frost_threshold)
    return jnp.where(blocked,0.,flux)


class LiftingResult(NamedTuple):
    mass_ratio: object
    number_ratio: object
    surface_dust_kg_m2: object
    diagnostics: object


def interactive_step(mass_ratio,number_ratio,temperature,pressure_half,pressure_full,
                     temperature_surface,stress,sensible_heat_flux,pbl_top_layer,pressure_pbl,
                     latitude_rad,snow,frost,opacity,surface_dust_kg_m2,dt_seconds,*,
                     stress_lift=True,convective_lift=True,stress_options=None,devil_options=None,
                     inject_pbl_stress=False,inject_pbl_devils=True,fixed_layers=(0,0),
                     stress_radius_m=2e-6,devil_radius_m=2e-6,limited_reservoir=False,
                     initial_surface_dust_kg_m2=30.,reservoir_factor=20.,threshold_stress=.022,
                     threshold_decrease=1e-11,devil_threshold_increase=.2,gravity=3.72,
                     gas_constant=189.02,cp=735.94):
    """One physical dust mode: source devil/stress order and reservoir feedback.

    The unlimited source reservoir may become negative, as in the reference.
    Limited-reservoir mode changes the stress threshold; it does not cap mass.
    Background scenario control, dual modes and tagged copies are separate.
    """
    q,n,t,p,full,ts,top,reservoir=map(jnp.asarray,(mass_ratio,number_ratio,temperature,
        pressure_half,pressure_full,temperature_surface,pbl_top_layer,surface_dust_kg_m2))
    if q.shape!=t.shape or n.shape!=t.shape or full.shape!=t.shape or p.shape!=(t.shape[0]+1,*t.shape[1:]):
        raise ValueError('dust and atmosphere grids must match')
    if len(fixed_layers)!=2 or not 0<=fixed_layers[0]<=fixed_layers[1]<t.shape[0]:
        raise ValueError('fixed dust layers must be ordered bottom offsets within the column')
    q,n=jnp.maximum(q,1e-31),jnp.maximum(n,1e-31)
    layer_mass=jnp.diff(p,axis=0)/gravity
    level=jnp.arange(t.shape[0]).reshape((t.shape[0],)+(1,)*(t.ndim-1))
    def inject(q,n,amount,radius,pbl):
        chosen=level>=top if pbl else ((level>=t.shape[0]-1-fixed_layers[1])&
            (level<=t.shape[0]-1-fixed_layers[0]))
        carrier=jnp.sum(jnp.where(chosen,layer_mass,0.),axis=0)
        addition=jnp.where(chosen,amount/jnp.where(carrier>0.,carrier,1.),0.)
        valid=(carrier>0.)&(top>=0)&(top<t.shape[0]) if pbl else carrier>0.
        addition=jnp.where(valid,addition,jnp.nan)
        particle_mass=4./3.*jnp.pi*2500.*(radius*jnp.exp(-.5*.63676**2))**3
        return q+addition,n+addition/particle_mass
    devil=jnp.zeros_like(reservoir);wind=jnp.zeros_like(reservoir)
    if convective_lift:
        devil=devil_flux(t[-1],ts,full[-1],pressure_pbl,top,t.shape[0]-1,sensible_heat_flux,
            latitude_rad,snow,frost,gas_constant=gas_constant,cp=cp,gravity=gravity,
            **({} if devil_options is None else devil_options))
        if limited_reservoir:
            increase=threshold_stress*jnp.expm1(2.*reservoir_factor*(initial_surface_dust_kg_m2-reservoir))
            devil=jnp.where(increase>devil_threshold_increase,0.,devil)
        devil=jnp.maximum(devil*dt_seconds,0.)
        q,n=inject(q,n,devil,devil_radius_m,inject_pbl_devils)
        reservoir=reservoir-devil
    threshold=jnp.full_like(reservoir,threshold_stress)
    if stress_lift:
        if limited_reservoir:
            threshold=threshold_stress*jnp.exp(2.*reservoir_factor*(initial_surface_dust_kg_m2-reservoir))-threshold_decrease*dt_seconds
        wind=stress_flux(stress,threshold,full[-1]/(gas_constant*t[-1]),latitude_rad,snow,frost,opacity,
            gravity=gravity,**({} if stress_options is None else stress_options))
        wind=jnp.maximum(wind*dt_seconds,0.)
        q,n=inject(q,n,wind,stress_radius_m,inject_pbl_stress)
        reservoir=reservoir-wind
    return LiftingResult(q,n,reservoir,dict(devil_lift_kg_m2=devil,stress_lift_kg_m2=wind,
        stress_threshold_pa=threshold))
