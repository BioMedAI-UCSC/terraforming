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
from src.framework.physics.rayleigh_drag import rayleigh_tendencies
from src.framework.physics.co2_microphysics import cloud_step,phase_change,transport_concentration


class ColumnResult(NamedTuple):
    temperature: object
    u: object
    v: object
    gas_tracers: object
    soil: object
    co2_ice_kg_m2: object
    gas_mass_loss_kg_m2: object
    diagnostics: object
    suspended_co2_ice_ratio: object = None


def column_physics(temperature,u,v,gas_tracers,soil,pressure_half,pressure_full,
                   co2_ice_kg_m2,radiative_heating_k_s,downward_surface_flux_w_m2,
                   dt_seconds,body,*,soil_thickness,soil_diffusivity,
                   soil_volumetric_heat_capacity,emissivity,latent_heat=5.9e5,
                   surface_exchange_start,latitude_rad=0.,rayleigh_options=None,
                   conserve_drag_energy=False,suspended_co2_ice_ratio=None,
                   cloud_nuclei_per_kg=1.e5):
    """Rayleigh/PBL, soil/frost, CO2 deposition or suspended clouds, convection.

    This is the standard present-Mars gas-column process order. Tracers here
    are gas species only. Optional CO2 cloud ice is supplied separately and
    excluded from gas PBL mixing. Convection mixes it with the other tracers,
    as in the source. Match the surface exchange start
    index to the reference grid before applying the returned gas mass losses.
    """
    if not isinstance(surface_exchange_start,int) or not 0<=surface_exchange_start<temperature.shape[0]:
        raise ValueError("surface mass exchange start must be a valid layer index")
    initial_wind=(u,v)
    predicted=jnp.asarray(temperature)+jnp.asarray(radiative_heating_k_s)*dt_seconds
    du,dv=rayleigh_tendencies(u,v,pressure_full,pressure_half[-1],latitude_rad,
        **({} if rayleigh_options is None else rayleigh_options))
    drag_heat=-((u+.5*du*dt_seconds)*du+(v+.5*dv*dt_seconds)*dv)*dt_seconds/body.cp_j_kg_k
    if conserve_drag_energy:
        predicted=predicted+drag_heat
    t,u,v,r,pbl=column_step(predicted,u+du*dt_seconds,v+dv*dt_seconds,gas_tracers,
        pressure_half,pressure_full,soil[0],
        dt_seconds,body.gas_constant_j_kg_k,body.cp_j_kg_k,body.gravity_m_s2,
        surface_co2_ice=co2_ice_kg_m2)
    pbl_wind=(u,v)
    ground,snow,surface=surface_step(soil,soil_thickness,soil_diffusivity,
        soil_volumetric_heat_capacity,pressure_half[-1],co2_ice_kg_m2,
        downward_surface_flux_w_m2,pbl['heat_flux_w_m2'],
        pbl['heat_flux_surface_derivative_w_m2_k'],emissivity,dt_seconds,latent_heat=latent_heat)
    cloud=None
    cloud_diagnostics={}
    if suspended_co2_ice_ratio is None:
        t,mass,precip,guard=simple_condensation(t,pressure_half,pressure_full,
            body.cp_j_kg_k,body.gravity_m_s2,latent_heat)
    else:
        cloud=cloud_step(t,pressure_half,pressure_full,suspended_co2_ice_ratio,ground[0],
            pbl['heat_diffusivity_m2_s'],dt_seconds,gas_constant=body.gas_constant_j_kg_k,
            gravity=body.gravity_m_s2,nuclei_per_kg=cloud_nuclei_per_kg)
        _,formed=phase_change(t,pressure_full,suspended_co2_ice_ratio)
        density=cloud.diagnostics['density_kg_m3']
        deposit_momentum=jnp.stack([transport_concentration(formed*density*wind,
            cloud.diagnostics,dt_seconds)[1] for wind in (u,v)])
        cloud_diagnostics=dict(cloud._asdict())
        cloud_diagnostics['deposition_momentum_kg_m_s']=deposit_momentum
        t,mass,precip=cloud.temperature,cloud.gas_mass_loss_kg_m2,cloud.deposition_kg_m2
        guard=jnp.zeros_like(t,dtype=bool)
        r=jnp.concatenate([r,cloud.ice_ratio[...,None]],axis=-1)
    t,u,v,r=pass2(t,u,v,r,pressure_half,pressure_full,body.kappa)
    suspended=None
    if cloud is not None:
        suspended,r=r[...,-1],r[...,:-1]
    dp=jnp.diff(pressure_half,axis=0)
    selected=jnp.asarray(np.arange(temperature.shape[0])>=surface_exchange_start)
    selected=selected.reshape((temperature.shape[0],)+(1,)*(dp.ndim-1))
    weights=jnp.where(selected,dp,0.)
    weights=weights/jnp.sum(weights,axis=0,keepdims=True)
    surface_mass=surface['co2_phase_change_kg_m2']*weights
    mass=mass+surface_mass
    if cloud is not None:
        gas=dp/body.gravity_m_s2
        # Only drag, surface phase exchange and deposited ice torque the ground.
        ground_impulse=jnp.stack([jnp.sum(gas*(before-after)+surface_mass*after,axis=0)
            for before,after in zip(initial_wind,pbl_wind,strict=True)])
        ground_impulse=ground_impulse+cloud_diagnostics['deposition_momentum_kg_m_s']
        cloud_diagnostics['ground_momentum_exchange_kg_m_s']=ground_impulse
        old_mass=gas*(1.+jnp.asarray(suspended_co2_ice_ratio))
        new_mass=gas-mass+gas*suspended
        cloud_diagnostics['unresolved_momentum_source_kg_m_s']=ground_impulse+jnp.stack([
            jnp.sum(new_mass*after-old_mass*before,axis=0)
            for before,after in zip(initial_wind,(u,v),strict=True)])
    diagnostics=dict(pbl=pbl,surface=surface,co2_mass_guard=guard,
        atmospheric_co2_precipitation_kg_m2=precip,
        co2_cloud=cloud_diagnostics,
        rayleigh=dict(u_tendency_m_s2=du,v_tendency_m_s2=dv,dissipated_heat_k=drag_heat,
            reheating_enabled=conserve_drag_energy))
    return ColumnResult(t,u,v,r,ground,snow+precip,mass,diagnostics,suspended)
