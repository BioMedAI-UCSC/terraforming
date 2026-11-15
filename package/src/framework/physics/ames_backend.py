"""Conservative sigma-grid adaptation of Ames column outputs for Dinosaur.

The column kernels match the reference; sigma remapping, spectral projection
and the Dinosaur dynamical core are separate numerical adaptations.
"""
import dataclasses
import numpy as np

from src.framework.gcm._dinosaur import jax,jnp,scales,spherical_harmonic
from src.framework.gcm.dynamics import primitive_equations,reference_temperature,stepper
from src.framework.physics.ames_column import column_physics
from src.framework.physics.ames_surface import soil_layer_thickness
from src.framework.physics.co2_clouds import _remap_extensive
from src.framework.physics.gcm import (ColumnPhysicsTendencies,column_primitive_equations,
    initial_column_state,positivity_preserving_co2_step,surface_radiative_forcing)


def remap_gas(temperature,u,v,tracers,pressure_pa,mass_loss,dsigma,body):
    """Apply Ames layer mass sources, then remap extensive fields to sigma.

    As in fv_phys's qratio update, gas tracers retain their material mass when
    CO2 is removed. Remapping conserves donor gas enthalpy plus kinetic energy
    and horizontal momentum; resolved mixing loss returns as heat. Nonpositive
    donor masses are invalid, not clipped to manufacture a finite climate.
    """
    temperature,u,v,tracers,pressure_pa,mass_loss=map(jnp.asarray,
        (temperature,u,v,tracers,pressure_pa,mass_loss))
    weights=jnp.asarray(dsigma).reshape((len(dsigma),)+(1,)*(temperature.ndim-1))
    gas=pressure_pa[None]*weights/body.gravity_m_s2
    donor=gas-mass_loss
    valid=jnp.all(donor>0.,axis=0)
    safe=jnp.where(valid[None],donor,gas)
    new_pressure=body.gravity_m_s2*jnp.sum(safe,axis=0)
    target=new_pressure[None]*weights/body.gravity_m_s2
    energy=safe*(body.cp_j_kg_k*temperature+.5*(u*u+v*v))
    extensive=jnp.concatenate([energy[None],(safe*u)[None],(safe*v)[None],
        jnp.moveaxis(gas[...,None]*tracers,-1,0)],axis=0)
    mapped=_remap_extensive(safe,target,extensive)
    new_u,new_v=mapped[1]/target,mapped[2]/target
    new_t=(mapped[0]/target-.5*(new_u**2+new_v**2))/body.cp_j_kg_k
    new_q=jnp.moveaxis(mapped[3:]/target[None],0,-1)
    return (jnp.where(valid[None],new_t,jnp.nan),jnp.where(valid[None],new_u,jnp.nan),
        jnp.where(valid[None],new_v,jnp.nan),jnp.where(valid[None,...,None],new_q,jnp.nan),
        jnp.where(valid,new_pressure,jnp.nan))


def initial_state(dynamics,coords,specs,temperature_k=170.,soil_layers=16):
    """Initialize the existing column state with the reference soil node count."""
    if soil_layers<3:
        raise ValueError("Ames soil requires two evolving nodes and a fixed deep node")
    state=initial_column_state(dynamics,coords,temperature_k,specs)
    ground=jnp.broadcast_to(state.surface_temperature,(soil_layers-1,)+coords.horizontal.nodal_shape)
    return state._replace(ground_temperature=ground)


def build_step(coords,specs,body,forcing,dt_seconds,*,radiation_component,
               soil_diffusivity,soil_volumetric_heat_capacity=1500.*627.9,
               soil_layers=16,orography=None,hyperdiffusion_tau_seconds=None,return_diagnostics=False):
    """Dinosaur dynamics followed by Ames-order reference gas-column physics.

    Supply a verified radiation component and mapped soil diffusivity. Keeping
    them explicit prevents substituting the legacy radiation or uniform soil
    silently. The current mode is immediate atmospheric CO2 deposition, with
    all present state tracers treated as gas/passive tracers. Suspended-cloud
    microphysics and general-mixture thermodynamics require separate modes.
    """
    grid=coords.horizontal
    sigma=np.asarray(coords.vertical.centers,dtype=float)
    dsigma=np.diff(np.asarray(coords.vertical.boundaries,dtype=float))
    below=np.flatnonzero(sigma/sigma[-1]<.991)
    exchange_start=max(0,int(below[-1])-2) if len(below) else 0
    dz=soil_layer_thickness(soil_layers)
    pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    velocity_scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    ref=jnp.asarray(reference_temperature(coords,body))[:,None,None]
    base=primitive_equations(coords,body,specs=specs,orography=orography)

    def no_forcing(state):
        dyn=state.dynamics
        return ColumnPhysicsTendencies(jnp.zeros_like(dyn.vorticity),jnp.zeros_like(dyn.divergence),
            jnp.zeros_like(dyn.temperature_variation),jnp.zeros_like(dyn.log_surface_pressure),
            jnp.zeros_like(state.surface_temperature),jnp.zeros_like(state.co2_ice),
            jnp.zeros_like(state.ground_temperature),{},
            None if state.surface_momentum is None else jnp.zeros_like(state.surface_momentum))

    dry=stepper(column_primitive_equations(base,no_forcing),dt_seconds,specs,coords=coords,
        hyperdiffusion_tau_seconds=hyperdiffusion_tau_seconds)

    def advance(state):
        predicted=dry(state)
        if predicted.ground_temperature.shape[0]!=soil_layers-1:
            raise ValueError("state soil nodes do not match the Ames backend configuration")
        dyn=predicted.dynamics
        temperature=grid.to_nodal(dyn.temperature_variation)+ref
        u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,dyn.vorticity,dyn.divergence)
        u,v=u*velocity_scale,v*velocity_scale
        pressure=jnp.exp(grid.to_nodal(dyn.log_surface_pressure))[0]*pressure_scale
        half=jnp.asarray(coords.vertical.boundaries)[:,None,None]*pressure
        full=jnp.asarray(coords.vertical.centers)[:,None,None]*pressure
        names=tuple(dyn.tracers)
        tracers=(jnp.stack([grid.to_nodal(dyn.tracers[name]) for name in names],axis=-1)
            if names else jnp.empty(temperature.shape+(0,),dtype=temperature.dtype))
        local_forcing=surface_radiative_forcing(predicted,coords,specs,body,forcing)
        radiation=radiation_component(predicted,coords,specs,body,local_forcing)
        layer_capacity=body.cp_j_kg_k*jnp.diff(half,axis=0)/body.gravity_m_s2
        heating=radiation.atmospheric_convergence_w_m2/layer_capacity
        down=(radiation.shortwave_down_w_m2[-1]-radiation.shortwave_up_w_m2[-1]+
            radiation.longwave_down_w_m2[-1])
        soil=jnp.concatenate([predicted.surface_temperature,predicted.ground_temperature],axis=0)
        frost=predicted.co2_ice[0]*pressure_scale/body.gravity_m_s2
        column=column_physics(temperature,u,v,tracers,soil,half,full,frost,heating,down,
            dt_seconds,body,soil_thickness=dz,soil_diffusivity=soil_diffusivity,
            soil_volumetric_heat_capacity=soil_volumetric_heat_capacity,
            emissivity=local_forcing.emissivity,surface_exchange_start=exchange_start)
        t,unew,vnew,q,new_pressure=remap_gas(column.temperature,column.u,column.v,
            column.gas_tracers,pressure,column.gas_mass_loss_kg_m2,dsigma,body)
        vor,div=spherical_harmonic.uv_nodal_to_vor_div_modal(grid,
            (unew-u)/velocity_scale,(vnew-v)/velocity_scale)
        from src.framework.physics.co2_clouds import CO2_CLOUD_TRACER
        unsupported_cloud=(jnp.any(grid.to_nodal(dyn.tracers[CO2_CLOUD_TRACER])!=0.)
            if CO2_CLOUD_TRACER in dyn.tracers else False)
        invalid=jnp.any(column.diagnostics['co2_mass_guard'])|unsupported_cloud
        new_temperature=dyn.temperature_variation+grid.to_modal(t-temperature)
        new_temperature=jnp.where(invalid,jnp.nan,new_temperature)
        dynamics=dataclasses.replace(dyn,temperature_variation=new_temperature,
            vorticity=dyn.vorticity+vor,divergence=dyn.divergence+div,
            log_surface_pressure=dyn.log_surface_pressure+grid.to_modal(jnp.log(new_pressure/pressure)[None]),
            tracers={name:dyn.tracers[name]+grid.to_modal(q[...,i]-tracers[...,i])
                for i,name in enumerate(names)})
        return predicted._replace(dynamics=dynamics,surface_temperature=column.soil[:1],
            ground_temperature=column.soil[1:],co2_ice=(column.co2_ice_kg_m2*body.gravity_m_s2/pressure_scale)[None])

    return positivity_preserving_co2_step(advance,coords,specs,body=body,
        return_diagnostics=return_diagnostics)
