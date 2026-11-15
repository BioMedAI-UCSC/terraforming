"""Ames-order column physics for the differentiable Dinosaur backend.

Radiation inputs are supplied explicitly. Gas mass losses are returned for
the dynamical core to transport; this column operator does not remap pressure.
"""
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.co2_condensation import simple_condensation
from src.framework.physics.dry_convection import pass2
from src.framework.physics.boundary_layer import column_step
from src.framework.physics.soil_frost import surface_step


class ColumnResult(NamedTuple):
    temperature: object
    u: object
    v: object
    gas_tracers: object
    soil: object
    co2_ice_kg_m2: object
    gas_mass_loss_kg_m2: object
    diagnostics: object


def column_physics(temperature,u,v,gas_tracers,soil,pressure_half,pressure_full,
                   co2_ice_kg_m2,radiative_heating_k_s,downward_surface_flux_w_m2,
                   dt_seconds,body,*,soil_thickness,soil_diffusivity,
                   soil_volumetric_heat_capacity,emissivity,latent_heat=5.9e5,
                   surface_exchange_start):
    """PBL, soil/frost, immediate atmospheric condensation, then convection.

    This is the standard present-Mars gas-column process order. Tracers here
    are gas species only. Optional suspended clouds/dust/water microphysics
    require their own state and operators. Match the surface exchange start
    index to the reference grid before applying the returned gas mass losses.
    """
    if not isinstance(surface_exchange_start,int) or not 0<=surface_exchange_start<temperature.shape[0]:
        raise ValueError("surface mass exchange start must be a valid layer index")
    predicted=jnp.asarray(temperature)+jnp.asarray(radiative_heating_k_s)*dt_seconds
    t,u,v,r,pbl=column_step(predicted,u,v,gas_tracers,pressure_half,pressure_full,soil[0],
        dt_seconds,body.gas_constant_j_kg_k,body.cp_j_kg_k,body.gravity_m_s2,
        surface_co2_ice=co2_ice_kg_m2)
    ground,snow,surface=surface_step(soil,soil_thickness,soil_diffusivity,
        soil_volumetric_heat_capacity,pressure_half[-1],co2_ice_kg_m2,
        downward_surface_flux_w_m2,pbl['heat_flux_w_m2'],
        pbl['heat_flux_surface_derivative_w_m2_k'],emissivity,dt_seconds,latent_heat=latent_heat)
    t,mass,precip,guard=simple_condensation(t,pressure_half,pressure_full,
        body.cp_j_kg_k,body.gravity_m_s2,latent_heat)
    t,u,v,r=pass2(t,u,v,r,pressure_half,pressure_full,body.kappa)
    dp=jnp.diff(pressure_half,axis=0)
    selected=jnp.asarray(np.arange(temperature.shape[0])>=surface_exchange_start)
    selected=selected.reshape((temperature.shape[0],)+(1,)*(dp.ndim-1))
    weights=jnp.where(selected,dp,0.)
    weights=weights/jnp.sum(weights,axis=0,keepdims=True)
    mass=mass+surface['co2_phase_change_kg_m2']*weights
    diagnostics=dict(pbl=pbl,surface=surface,co2_mass_guard=guard,
        atmospheric_co2_precipitation_kg_m2=precip)
    return ColumnResult(t,u,v,r,ground,snow+precip,mass,diagnostics)
