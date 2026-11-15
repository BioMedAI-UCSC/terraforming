"""Conservative sigma-grid adaptation of Ames column outputs for Dinosaur.

The column kernels match the reference; sigma remapping, spectral projection
and the Dinosaur dynamical core are separate numerical adaptations.
"""
import dataclasses
import numpy as np

from src.framework.gcm._dinosaur import jax,jnp,scales,spherical_harmonic
from src.framework.gcm.dynamics import primitive_equations,reference_temperature,stepper
from src.framework.physics.column_physics import column_physics
from src.framework.physics.soil_frost import soil_layer_thickness
from src.framework.physics.co2_clouds import _remap_extensive,CO2_CLOUD_TRACER
from src.framework.physics.co2_microphysics import LATENT_HEAT as CLOUD_LATENT_HEAT
from src.framework.physics.gcm import (ColumnPhysicsTendencies,column_primitive_equations,
    initial_column_state,positivity_preserving_co2_step,surface_radiative_forcing)


def remap_gas(temperature,u,v,tracers,pressure_pa,mass_loss,dsigma,body,*,cloud_index=None):
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
    cloud=(jnp.zeros_like(gas) if cloud_index is None else gas*tracers[...,cloud_index])
    moving_mass=safe+cloud
    energy=safe*body.cp_j_kg_k*temperature+moving_mass*.5*(u*u+v*v)
    extensive=jnp.concatenate([energy[None],(moving_mass*u)[None],(moving_mass*v)[None],
        jnp.moveaxis(gas[...,None]*tracers,-1,0)],axis=0)
    mapped=_remap_extensive(safe,target,extensive)
    target_mass=target+(0. if cloud_index is None else mapped[3+cloud_index])
    new_u,new_v=mapped[1]/target_mass,mapped[2]/target_mass
    new_t=(mapped[0]-.5*target_mass*(new_u**2+new_v**2))/(target*body.cp_j_kg_k)
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


def axial_angular_momentum(state,coords,specs,body):
    """SI axial momentum of gas, surface CO2 and the ground impulse reservoir.

    Includes suspended CO2 ice when present, planetary rotation and zonal wind.
    """
    grid=coords.horizontal
    pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    speed_scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    pressure=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*pressure_scale
    gas=pressure/body.gravity_m_s2
    frost=state.co2_ice[0]*pressure_scale/body.gravity_m_s2
    u,_=spherical_harmonic.vor_div_to_uv_nodal(grid,state.dynamics.vorticity,state.dynamics.divergence)
    weights=jnp.asarray(np.diff(coords.vertical.boundaries))[:,None,None]
    cloud=(grid.to_nodal(state.dynamics.tracers[CO2_CLOUD_TRACER])
        if CO2_CLOUD_TRACER in state.dynamics.tracers else jnp.zeros_like(u))
    moving_mass=gas[None]*weights*(1.+cloud)
    relative=jnp.sum(moving_mass*u*speed_scale,axis=0)
    if state.surface_momentum is not None:
        relative=relative+state.surface_momentum[0]
    lever=body.radius_m*jnp.asarray(grid.cos_lat)
    local=lever*(relative+(jnp.sum(moving_mass,axis=0)+frost)*body.angular_velocity_s*lever)
    area=jnp.asarray(grid.quadrature_weights)
    return jnp.sum(local*area)/jnp.sum(area)*body.surface_area_m2


def exchange_energy(state,coords,specs,body,*,soil_volumetric_heat_capacity=1500.*627.9,
                    latent_heat=5.9e5):
    """SI resolved gas/cloud/soil/frost exchange ledger for reference modes.

    Count gas enthalpy plus relative kinetic energy, cloud latent/kinetic energy,
    evolving soil-node sensible energy, and frost enthalpy cp*T_surface-L.
    The fixed deep node is an external reservoir. This quantity exposes numerical
    repair changes; it is not a conservation theorem for the reference physics,
    whose default drag/convection omit dissipative reheating.
    """
    grid=coords.horizontal
    pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    speed_scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    pressure=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*pressure_scale
    temperature=grid.to_nodal(state.dynamics.temperature_variation)+jnp.asarray(
        reference_temperature(coords,body))[:,None,None]
    u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,state.dynamics.vorticity,state.dynamics.divergence)
    weights=jnp.asarray(np.diff(coords.vertical.boundaries))[:,None,None]
    gas=pressure[None]*weights/body.gravity_m_s2
    energy=jnp.sum(gas*(body.cp_j_kg_k*temperature+.5*(u*u+v*v)*speed_scale**2),axis=0)
    if CO2_CLOUD_TRACER in state.dynamics.tracers:
        cloud=gas*grid.to_nodal(state.dynamics.tracers[CO2_CLOUD_TRACER])
        # The source bulk scheme neglects cloud sensible and potential energy.
        energy=energy+jnp.sum(cloud*(.5*(u*u+v*v)*speed_scale**2-CLOUD_LATENT_HEAT),axis=0)
    soil=jnp.concatenate([state.surface_temperature,state.ground_temperature],axis=0)
    capacity=jnp.asarray(soil_volumetric_heat_capacity)
    if capacity.ndim==1:
        capacity=capacity[:,None,None]
    capacity=jnp.broadcast_to(capacity,soil.shape)
    dz=jnp.asarray(soil_layer_thickness(soil.shape[0]))[:soil.shape[0]-1,None,None]
    energy=energy+jnp.sum(capacity[:-1]*dz*soil[:-1],axis=0)
    frost=state.co2_ice[0]*pressure_scale/body.gravity_m_s2
    energy=energy+frost*(body.cp_j_kg_k*state.surface_temperature[0]-latent_heat)
    area=jnp.asarray(grid.quadrature_weights)
    return jnp.sum(energy*area)/jnp.sum(area)*body.surface_area_m2


def build_step(coords,specs,body,forcing,dt_seconds,*,radiation_component,
               soil_diffusivity,soil_volumetric_heat_capacity=1500.*627.9,
               soil_layers=16,orography=None,hyperdiffusion_tau_seconds=None,return_diagnostics=False,
               rayleigh_options=None,conserve_drag_energy=False,
               suspended_co2_clouds=False,cloud_nuclei_per_kg=1.e5):
    """Dinosaur dynamics followed by Ames-order reference gas-column physics.

    Supply a verified radiation component and mapped soil diffusivity. Keeping
    them explicit prevents substituting the legacy radiation or uniform soil
    silently. Optional suspended CO2 uses the source bulk cloud scheme, with
    ice excluded from gas PBL mixing. The source neglects cloud thermal inertia
    and two-way momentum feedback; energy and angular momentum ledgers expose
    those approximations. General-mixture thermodynamics is separate.
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
        if predicted.surface_momentum is None:
            raise ValueError("column physics requires the ground momentum reservoir")
        dyn=predicted.dynamics
        temperature=grid.to_nodal(dyn.temperature_variation)+ref
        u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,dyn.vorticity,dyn.divergence)
        u,v=u*velocity_scale,v*velocity_scale
        pressure=jnp.exp(grid.to_nodal(dyn.log_surface_pressure))[0]*pressure_scale
        half=jnp.asarray(coords.vertical.boundaries)[:,None,None]*pressure
        full=jnp.asarray(coords.vertical.centers)[:,None,None]*pressure
        names=tuple(dyn.tracers)
        if suspended_co2_clouds and CO2_CLOUD_TRACER not in names:
            raise ValueError('suspended CO2 mode requires the cloud tracer in the initial state')
        gas_names=tuple(name for name in names if name!=CO2_CLOUD_TRACER)
        cloud_ratio=(grid.to_nodal(dyn.tracers[CO2_CLOUD_TRACER])
            if CO2_CLOUD_TRACER in names else None)
        tracers=(jnp.stack([grid.to_nodal(dyn.tracers[name]) for name in names],axis=-1)
            if names else jnp.empty(temperature.shape+(0,),dtype=temperature.dtype))
        gas_tracers=(jnp.stack([grid.to_nodal(dyn.tracers[name]) for name in gas_names],axis=-1)
            if gas_names else jnp.empty(temperature.shape+(0,),dtype=temperature.dtype))
        local_forcing=surface_radiative_forcing(predicted,coords,specs,body,forcing)
        radiation=radiation_component(predicted,coords,specs,body,local_forcing)
        layer_capacity=body.cp_j_kg_k*jnp.diff(half,axis=0)/body.gravity_m_s2
        heating=radiation.atmospheric_convergence_w_m2/layer_capacity
        down=(radiation.shortwave_down_w_m2[-1]-radiation.shortwave_up_w_m2[-1]+
            radiation.longwave_down_w_m2[-1])
        soil=jnp.concatenate([predicted.surface_temperature,predicted.ground_temperature],axis=0)
        frost=predicted.co2_ice[0]*pressure_scale/body.gravity_m_s2
        column=column_physics(temperature,u,v,gas_tracers,soil,half,full,frost,heating,down,
            dt_seconds,body,soil_thickness=dz,soil_diffusivity=soil_diffusivity,
            soil_volumetric_heat_capacity=soil_volumetric_heat_capacity,
            emissivity=local_forcing.emissivity,surface_exchange_start=exchange_start,
            latitude_rad=jnp.asarray(grid.latitudes)[None,:],rayleigh_options=rayleigh_options,
            conserve_drag_energy=conserve_drag_energy,
            suspended_co2_ice_ratio=cloud_ratio if suspended_co2_clouds else None,
            cloud_nuclei_per_kg=cloud_nuclei_per_kg)
        columns={name:column.gas_tracers[...,i] for i,name in enumerate(gas_names)}
        if cloud_ratio is not None:
            columns[CO2_CLOUD_TRACER]=(column.suspended_co2_ice_ratio
                if suspended_co2_clouds else cloud_ratio)
        material=(jnp.stack([columns[name] for name in names],axis=-1)
            if names else column.gas_tracers)
        t,unew,vnew,q,new_pressure=remap_gas(column.temperature,column.u,column.v,
            material,pressure,column.gas_mass_loss_kg_m2,dsigma,body,
            cloud_index=names.index(CO2_CLOUD_TRACER) if suspended_co2_clouds else None)
        old_gas=pressure[None]*jnp.asarray(dsigma)[:,None,None]/body.gravity_m_s2
        new_gas=new_pressure[None]*jnp.asarray(dsigma)[:,None,None]/body.gravity_m_s2
        impulse=jnp.stack([jnp.sum(old_gas*u-new_gas*unew,axis=0),
            jnp.sum(old_gas*v-new_gas*vnew,axis=0)])
        if suspended_co2_clouds:
            # Keep the source's missing two-way cloud reaction in the air ledger.
            impulse=column.diagnostics['co2_cloud']['ground_momentum_exchange_kg_m_s']
        vor,div=spherical_harmonic.uv_nodal_to_vor_div_modal(grid,
            (unew-u)/velocity_scale,(vnew-v)/velocity_scale)
        unsupported_cloud=(jnp.any(grid.to_nodal(dyn.tracers[CO2_CLOUD_TRACER])!=0.)
            if CO2_CLOUD_TRACER in dyn.tracers and not suspended_co2_clouds else False)
        invalid=jnp.any(column.diagnostics['co2_mass_guard'])|unsupported_cloud
        new_temperature=dyn.temperature_variation+grid.to_modal(t-temperature)
        new_temperature=jnp.where(invalid,jnp.nan,new_temperature)
        dynamics=dataclasses.replace(dyn,temperature_variation=new_temperature,
            vorticity=dyn.vorticity+vor,divergence=dyn.divergence+div,
            log_surface_pressure=dyn.log_surface_pressure+grid.to_modal(jnp.log(new_pressure/pressure)[None]),
            tracers={name:dyn.tracers[name]+grid.to_modal(q[...,i]-tracers[...,i])
                for i,name in enumerate(names)})
        return predicted._replace(dynamics=dynamics,surface_temperature=column.soil[:1],
            ground_temperature=column.soil[1:],co2_ice=(column.co2_ice_kg_m2*body.gravity_m_s2/pressure_scale)[None],
            surface_momentum=predicted.surface_momentum+impulse)

    if not return_diagnostics:
        return positivity_preserving_co2_step(advance,coords,specs,body=body)

    def audited(state):
        raw=advance(state)
        final,diagnostics=positivity_preserving_co2_step(lambda _:raw,coords,specs,
            body=body,return_diagnostics=True)(state)
        diagnostics=diagnostics._replace(
            raw_exchange_energy_j=exchange_energy(raw,coords,specs,body,
                soil_volumetric_heat_capacity=soil_volumetric_heat_capacity),
            corrected_exchange_energy_j=exchange_energy(final,coords,specs,body,
                soil_volumetric_heat_capacity=soil_volumetric_heat_capacity),
            raw_axial_angular_momentum_kg_m2_s=axial_angular_momentum(raw,coords,specs,body),
            corrected_axial_angular_momentum_kg_m2_s=axial_angular_momentum(final,coords,specs,body))
        return final,diagnostics
    return audited
