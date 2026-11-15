"""Reference vapor exchange with surface water frost and liquid reservoirs.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.water_bulk import saturation_ratio,LATENT_HEAT


class WaterSurfaceResult(NamedTuple):
    vapor_ratio: object
    surface_water_kg_m2: object
    upward_flux_kg_m2_s: object
    latent_flux_w_m2: object
    latent_derivative_w_m2_k: object
    diagnostics: object


def surface_exchange(vapor_ratio,pressure_half,surface_temperature,surface_water_kg_m2,
                     exchange_coefficient,pbl_top_layer,dt_seconds,*,gravity=3.72,
                     inject_through_pbl=True,sublimation_factor=1.,scheme='bulk',
                     liquid_vapor_pressure=False,bulk_latent_heat=False):
    """Original update_water[_blk/_fv3] with zero-based PBL top indices.

    Positive flux moves surface water into the air. Sublimation is limited by
    the surface inventory; deposition is limited by bottom-layer vapor. The
    source's bulk latent diagnostic and its derivative are retained literally.
    The moment scheme scales positive flux only and repairs negative frost.
    """
    if scheme not in ('bulk','moment','bin'):
        raise ValueError('water surface scheme must be bulk, moment or bin')
    q,p,t,frost,drag,top=map(jnp.asarray,(vapor_ratio,pressure_half,surface_temperature,
        surface_water_kg_m2,exchange_coefficient,pbl_top_layer))
    n=q.shape[0]
    if p.shape!=(n+1,*q.shape[1:]):
        raise ValueError('water vapor and pressure interfaces must match')
    valid=(top>=0)&(top<n)
    if not inject_through_pbl: top=jnp.full_like(top,n-1)
    levels=jnp.arange(n).reshape((n,)+(1,)*(q.ndim-1))
    selected=levels>=top
    layer_mass=jnp.diff(p,axis=0)/gravity
    pbl_mass=jnp.sum(jnp.where(selected,layer_mass,0.),axis=0)
    ground=saturation_ratio(t,p[-1])
    if scheme=='bulk' and liquid_vapor_pressure:
        celsius=jnp.where(t>=273.16,t-273.16,0.)
        enhancement=1.0007+3.46e-6*p[-1]/100.
        liquid=enhancement*18./(44.*p[-1])*611.21*jnp.exp(17.502*celsius/(240.97+celsius))
        ground=jnp.where(t>=273.16,liquid,ground)
    requested=drag*(ground-q[-1])
    repair=jnp.zeros_like(frost)
    if scheme=='moment':
        repair=jnp.maximum(-frost,0.)
        frost=jnp.maximum(frost,0.)
        requested=jnp.where(requested>0.,requested*sublimation_factor,requested)
    else:
        requested=jnp.where(frost>=0.,sublimation_factor*requested,0.)
    flux=jnp.where(requested>0.,jnp.minimum(requested,frost/dt_seconds),
        jnp.maximum(requested,-q[-1]*layer_mass[-1]/dt_seconds))
    safe_mass=jnp.where(pbl_mass>0.,pbl_mass,1.)
    updated=q+jnp.where((flux>0.)[None]&selected,dt_seconds*flux[None]/safe_mass[None],0.)
    updated=updated.at[-1].add(jnp.where(flux<0.,dt_seconds*flux/layer_mass[-1],0.))
    latent=jnp.zeros_like(t);derivative=jnp.zeros_like(t)
    if scheme=='bulk' and bulk_latent_heat:
        latent=jnp.where(flux!=0.,drag*LATENT_HEAT*(q[-1]-updated[-1]),0.)
        derivative=jnp.where(flux!=0.,-6146.1*drag*LATENT_HEAT*ground/t**2,0.)
    valid=valid&(pbl_mass>0.)
    return WaterSurfaceResult(jnp.where(valid[None],updated,jnp.nan),
        jnp.where(valid,frost-flux*dt_seconds,jnp.nan),flux,latent,derivative,
        dict(requested_upward_flux_kg_m2_s=requested,pbl_mass_kg_m2=pbl_mass,
            repaired_surface_mass_kg_m2=repair,ground_saturation_ratio=ground))


def optical_properties(albedo,emissivity,surface_temperature,co2_ice_kg_m2,
                       water_kg_m2,*,frost_threshold_m=1.e6,ice_albedo=.4,
                       liquid_albedo=.07,water_emissivity=1.,liquid_water=False):
    """Original albedo_calc water override, including its strict frost threshold."""
    covered=(jnp.asarray(water_kg_m2)>frost_threshold_m*1000.)&(jnp.asarray(co2_ice_kg_m2)<=0.)
    water_albedo=jnp.where(liquid_water&(jnp.asarray(surface_temperature)>=273.),
        liquid_albedo,ice_albedo)
    return (jnp.where(covered,water_albedo,albedo),
            jnp.where(covered,water_emissivity,emissivity))
