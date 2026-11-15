"""Conservative sigma-grid adaptation of Ames column outputs for Dinosaur.

The column kernels match the reference; sigma remapping, spectral projection
and the Dinosaur dynamical core are separate numerical adaptations.
"""
import dataclasses
import numpy as np

from src.framework.gcm._dinosaur import jax,jnp,scales,spherical_harmonic
from src.framework.gcm.dynamics import primitive_equations,reference_temperature,stepper
from src.framework.physics.column_physics import column_physics,BulkWaterColumnState
from src.framework.physics.soil_frost import soil_layer_thickness
from src.framework.physics.co2_clouds import _remap_extensive,CO2_CLOUD_TRACER
from src.framework.physics.co2_microphysics import LATENT_HEAT as CLOUD_LATENT_HEAT
from src.framework.physics.water_bulk import WATER_VAPOR_TRACER,WATER_CLOUD_TRACER
from src.framework.physics.water_moments import MOMENT_PARTICLE_TRACERS
from src.framework.physics.particle_optics import dust_optics
from src.framework.physics.radiation_grid import radiation_grid
from src.framework.physics.dust_lifting import reference_efficiencies
from src.framework.physics.dust_control import (LocalTimeState,seasonal_sample,local_time_step,
    validate_local_time_scenario)
from src.framework.physics.nonorographic_drag import DragState,initial_drag_state,source_parameters,schedule
from src.framework.physics.gcm import (ColumnPhysicsTendencies,column_primitive_equations,
    initial_column_state,positivity_preserving_co2_step,surface_radiative_forcing,_true_anomaly)


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


def initial_state(dynamics,coords,specs,temperature_k=170.,soil_layers=16,*,bulk_water=False,
                  initial_water_surface_kg_m2=0.,initial_water_vapor_ratio=0.,
                  radiation_update_seconds=None,moment_water=False,
                  initial_dust_mass_ratio=0.,initial_dust_number_per_kg=0.,
                  initial_dust_surface_kg_m2=30.,nonorographic_drag_options=None,dust_background_options=None):
    """Initialize the existing column state with the reference soil node count."""
    if soil_layers<3:
        raise ValueError("Ames soil requires two evolving nodes and a fixed deep node")
    if bulk_water and moment_water:
        raise ValueError('bulk water and moment water cannot be enabled together')
    state=initial_column_state(dynamics,coords,temperature_k,specs)
    ground=jnp.broadcast_to(state.surface_temperature,(soil_layers-1,)+coords.horizontal.nodal_shape)
    if radiation_update_seconds is not None:
        if not np.isfinite(radiation_update_seconds) or radiation_update_seconds<=0.:
            raise ValueError('radiation update interval must be finite and positive')
        shape=state.surface_temperature.shape
        fields=dict(radiation_heating_k_s=jnp.zeros((len(coords.vertical.centers),)+shape[1:]),
            radiation_surface_down_w_m2=jnp.zeros(shape),radiation_initialized=jnp.zeros(shape),
            radiation_stratosphere_k=jnp.full(shape,170.))
        state=state._replace(auxiliary_fields=fields)
    if dust_background_options is not None and dust_background_options.get('local_time_scenario') is not None:
        zero=jnp.zeros(state.surface_temperature.shape)
        fields={} if state.auxiliary_fields is None else dict(state.auxiliary_fields)
        fields.update(dust_local_time_opacity=zero,dust_local_time_bin=zero,
            dust_local_time_initialized=jnp.asarray(0.))
        state=state._replace(auxiliary_fields=fields)
    if nonorographic_drag_options is not None:
        template=jnp.zeros((len(coords.vertical.centers),*coords.horizontal.nodal_shape),dtype=state.surface_temperature.dtype)
        drag=initial_drag_state(template,update_seconds=nonorographic_drag_options.get('update_seconds',0),
            offset_seconds=nonorographic_drag_options.get('offset_seconds',0))
        fields={} if state.auxiliary_fields is None else dict(state.auxiliary_fields)
        fields.update(nonorographic_u_tendency_m_s2=drag.u_tendency,nonorographic_v_tendency_m_s2=drag.v_tendency,
            nonorographic_alarm_seconds=drag.alarm_seconds,nonorographic_frequency_seconds=drag.frequency_seconds)
        state=state._replace(auxiliary_fields=fields)
    if not (bulk_water or moment_water): return state._replace(ground_temperature=ground)
    tracers={WATER_VAPOR_TRACER:coords.horizontal.to_modal(jnp.broadcast_to(
        jnp.asarray(initial_water_vapor_ratio),(len(coords.vertical.centers),)+coords.horizontal.nodal_shape)),
        WATER_CLOUD_TRACER:jnp.zeros_like(dynamics.temperature_variation),**state.dynamics.tracers}
    if moment_water:
        for name,value in zip(MOMENT_PARTICLE_TRACERS,
                              (initial_dust_mass_ratio,initial_dust_number_per_kg,0.,0.),strict=True):
            if name not in tracers:
                tracers[name]=coords.horizontal.to_modal(jnp.broadcast_to(jnp.asarray(value),
                    (len(coords.vertical.centers),)+coords.horizontal.nodal_shape))
    shape=state.surface_temperature.shape
    fields=dict({} if state.auxiliary_fields is None else state.auxiliary_fields,water_surface_kg_m2=jnp.broadcast_to(jnp.asarray(initial_water_surface_kg_m2),shape),
        water_exchange_coefficient_kg_m2_s=jnp.zeros(shape),
        water_pbl_top_layer=jnp.full(shape,len(coords.vertical.centers)-1,dtype=state.surface_temperature.dtype))
    if moment_water:
        fields['dust_surface_kg_m2']=jnp.broadcast_to(jnp.asarray(initial_dust_surface_kg_m2),shape)
        fields['dust_reference_opacity']=jnp.zeros((2,*shape[1:]))
        fields['dust_coagulation_mass_change_kg_m2']=jnp.zeros(shape)
    return state._replace(dynamics=dataclasses.replace(state.dynamics,tracers=tracers),
        ground_temperature=ground,auxiliary_fields=fields)


def water_inventory(state,coords,specs,body):
    """SI water mass in gas/ice tracers and the surface reservoir, independently of CO2."""
    grid=coords.horizontal
    scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    ps=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*scale
    gas=ps[None]*jnp.asarray(np.diff(coords.vertical.boundaries))[:,None,None]/body.gravity_m_s2
    ratios=sum((grid.to_nodal(state.dynamics.tracers[name]) for name in
        (WATER_VAPOR_TRACER,WATER_CLOUD_TRACER) if name in state.dynamics.tracers),jnp.zeros_like(gas))
    water=jnp.sum(gas*ratios,axis=0)
    if state.auxiliary_fields is not None and 'water_surface_kg_m2' in state.auxiliary_fields:
        water=water+state.auxiliary_fields['water_surface_kg_m2'][0]
    weights=jnp.asarray(grid.quadrature_weights)
    return jnp.sum(water*weights)/jnp.sum(weights)*body.surface_area_m2


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
               suspended_co2_clouds=False,cloud_nuclei_per_kg=1.e5,
               bulk_water=False,water_surface_options=None,water_cloud_options=None,form_bulk_clouds=True,
               terrain_drag_options=None,radiation_update_seconds=None,
               moment_water=False,moment_cloud_options=None,moment_sedimentation=True,
               dust_lifting_options=None,dust_optical_bands=12,dust_opacity_channel=1,
               coagulation_options=None,dust_background_options=None,nonorographic_drag_options=None):
    """Dinosaur dynamics followed by Ames-order reference gas-column physics.

    Supply a verified radiation component and mapped soil diffusivity. Keeping
    them explicit prevents substituting the legacy radiation or uniform soil
    silently. Optional suspended CO2 uses the source bulk cloud scheme, with
    ice excluded from gas PBL mixing. The source neglects cloud thermal inertia
    and two-way momentum feedback; energy and angular momentum ledgers expose
    those approximations. General-mixture thermodynamics is separate.
    """
    grid=coords.horizontal
    if bulk_water and moment_water:
        raise ValueError('bulk water and moment water cannot be enabled together')
    water_cycle=bulk_water or moment_water
    if coagulation_options is not None and not moment_water:
        raise ValueError('dust coagulation requires initialized moment tracers')
    dust_control=dust_lifting_options is not None or dust_background_options is not None
    background_scenario=None;local_time_scenario=None
    if dust_background_options is not None:
        if dust_lifting_options is not None and dust_lifting_options.get('dual_mode',0)<=0:
            raise ValueError('combined interactive/background dust requires a positive dual mode')
        dust_background_options=dict(dust_background_options)
        background_scenario=dust_background_options.pop('scenario',None)
        background_segment=dust_background_options.pop('segment',False)
        local_time_scenario=dust_background_options.pop('local_time_scenario',None)
        if background_scenario is None or {'seasons','opacity'}-set(background_scenario):
            raise ValueError('background dust requires mapped seasonal opacity and solar-longitude axes')
        shape=np.shape(background_scenario['seasons'])
        if len(shape)!=1 or shape[0]<2 or np.shape(background_scenario['opacity'])!=(shape[0],*grid.nodal_shape):
            raise ValueError('background dust seasonal maps must match the Dinosaur nodal grid')
        axis=np.asarray(background_scenario['seasons']);maps=np.asarray(background_scenario['opacity'])
        if not np.all(np.isfinite(axis)) or np.any(np.diff(axis)<=0.) or axis[0]<0. or axis[-1]!=360. or not np.all(np.isfinite(maps)) or np.any(maps<0.):
            raise ValueError('background dust requires increasing seasonal nodes ending at 360 and finite nonnegative opacity')
        if local_time_scenario is not None:
            validate_local_time_scenario(local_time_scenario,axis,grid.nodal_shape)
        dust_background_options.setdefault('extinction_efficiency',reference_efficiencies(
            dust_background_options.get('effective_radius_m',2e-6),bands=dust_optical_bands)[dust_opacity_channel-1])
    if dust_control:
        if not moment_water:
            raise ValueError('interactive dust requires initialized moment tracers')
        if dust_optical_bands not in (12,15) or dust_opacity_channel not in (1,2):
            raise ValueError('dust optics require 12/15 bands and visible/infrared channel 1/2')
    if moment_water:
        water_surface_options=dict({} if water_surface_options is None else water_surface_options,
            scheme='moment')
    sigma=np.asarray(coords.vertical.centers,dtype=float)
    dsigma=np.diff(np.asarray(coords.vertical.boundaries,dtype=float))
    if nonorographic_drag_options is not None:
        nonorographic_drag_options=dict(nonorographic_drag_options)
        if 'top_pressure_pa' not in nonorographic_drag_options:
            raise ValueError('non-orographic drag requires an explicit diagnostic top pressure')
        schedule(nonorographic_drag_options.get('update_seconds',0),nonorographic_drag_options.get('offset_seconds',0))
        source_options={name:nonorographic_drag_options.pop(name) for name in
            ('source_pressure_pa','base_amplitude','equatorial_amplitude','equatorial_width')
            if name in nonorographic_drag_options}
        source_index,amplitude=source_parameters(sigma*body.reference_surface_pressure_pa,
            jnp.asarray(grid.latitudes)[None,:],**source_options)
        nonorographic_drag_options.setdefault('source_index',source_index)
        nonorographic_drag_options.setdefault('source_amplitude',amplitude)
    below=np.flatnonzero(sigma/sigma[-1]<.991)
    exchange_start=max(0,int(below[-1])-2) if len(below) else 0
    dz=soil_layer_thickness(soil_layers)
    pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    velocity_scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    time_scale=float(specs.dimensionalize(1.,scales.units.second).magnitude)
    if radiation_update_seconds is not None:
        ratio=radiation_update_seconds/dt_seconds
        if not np.isfinite(ratio) or ratio<1. or not np.isclose(ratio,round(ratio),rtol=0.,atol=1e-10):
            raise ValueError('radiation interval must be an integer number of physics steps')
        if not np.isclose(radiation_update_seconds,round(radiation_update_seconds),rtol=0.,atol=1e-10):
            raise ValueError('radiation update interval must use whole seconds')
        radiation_interval=round(radiation_update_seconds)
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
        if water_cycle:
            required={'water_surface_kg_m2','water_exchange_coefficient_kg_m2_s','water_pbl_top_layer'}
            if state.auxiliary_fields is None or required-set(state.auxiliary_fields):
                raise ValueError('bulk water requires initialized water reservoirs and exchange history')
            if {WATER_VAPOR_TRACER,WATER_CLOUD_TRACER}-set(state.dynamics.tracers):
                raise ValueError('bulk water requires vapor and cloud tracers in the initial state')
        if moment_water and set(MOMENT_PARTICLE_TRACERS)-set(state.dynamics.tracers):
            raise ValueError('moment water requires dust mass/number and cloud number/core tracers')
        if moment_water and 'dust_surface_kg_m2' not in state.auxiliary_fields:
            raise ValueError('moment water requires the initialized dust surface reservoir')
        if dust_control and 'dust_reference_opacity' not in state.auxiliary_fields:
            raise ValueError('interactive dust requires initialized reference-opacity history')
        if coagulation_options is not None and 'dust_coagulation_mass_change_kg_m2' not in state.auxiliary_fields:
            raise ValueError('dust coagulation requires the initialized mass-change ledger')
        if nonorographic_drag_options is not None:
            required={'nonorographic_u_tendency_m_s2','nonorographic_v_tendency_m_s2',
                'nonorographic_alarm_seconds','nonorographic_frequency_seconds'}
            if state.auxiliary_fields is None or required-set(state.auxiliary_fields):
                raise ValueError('non-orographic drag requires initialized tendencies and timer state')
        if local_time_scenario is not None:
            required={'dust_local_time_opacity','dust_local_time_bin','dust_local_time_initialized'}
            if state.auxiliary_fields is None or required-set(state.auxiliary_fields):
                raise ValueError('local-time dust requires initialized opacity and bin history')
        if radiation_update_seconds is not None:
            required={'radiation_heating_k_s','radiation_surface_down_w_m2',
                'radiation_initialized','radiation_stratosphere_k'}
            if state.auxiliary_fields is None or required-set(state.auxiliary_fields):
                raise ValueError('radiation schedule requires initialized heating and stratosphere state')
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
        particles=(CO2_CLOUD_TRACER,WATER_CLOUD_TRACER)+ (MOMENT_PARTICLE_TRACERS if moment_water else ())
        gas_names=tuple(name for name in names if name not in particles)
        cloud_ratio=(grid.to_nodal(dyn.tracers[CO2_CLOUD_TRACER])
            if CO2_CLOUD_TRACER in names else None)
        tracers=(jnp.stack([grid.to_nodal(dyn.tracers[name]) for name in names],axis=-1)
            if names else jnp.empty(temperature.shape+(0,),dtype=temperature.dtype))
        gas_tracers=(jnp.stack([grid.to_nodal(dyn.tracers[name]) for name in gas_names],axis=-1)
            if gas_names else jnp.empty(temperature.shape+(0,),dtype=temperature.dtype))
        local_forcing=surface_radiative_forcing(predicted,coords,specs,body,forcing)
        layer_capacity=body.cp_j_kg_k*jnp.diff(half,axis=0)/body.gravity_m_s2
        def evaluate(_):
            radiation=radiation_component(predicted,coords,specs,body,local_forcing)
            heating=radiation.atmospheric_convergence_w_m2/layer_capacity
            if radiation.nonlte_heating_k_s is not None:
                heating=heating+radiation.nonlte_heating_k_s
            down=(radiation.shortwave_down_w_m2[-1]-radiation.shortwave_up_w_m2[-1]+
                radiation.longwave_down_w_m2[-1])
            strat_rate=body.gravity_m_s2/body.cp_j_kg_k/100.*radiation.stratosphere_flux_convergence_w_m2
            return heating,down,jnp.broadcast_to(strat_rate,pressure.shape)
        radiation_fields=None
        if radiation_update_seconds is None:
            heating,down,_=evaluate(None)
        else:
            fields=predicted.auxiliary_fields
            seconds=jnp.mod(jnp.rint(predicted.dynamics.sim_time*time_scale).astype(jnp.int64),
                round(body.solar_day_seconds))
            refresh=jnp.all(fields['radiation_initialized']==0.)|(jnp.mod(seconds,radiation_interval)==0)
            def held(_):
                return fields['radiation_heating_k_s'],fields['radiation_surface_down_w_m2'][0],jnp.zeros_like(pressure)
            heating,down,strat_rate=jax.lax.cond(refresh,evaluate,held,operand=None)
            old_strat=fields['radiation_stratosphere_k'][0]
            strat=old_strat+strat_rate*radiation_update_seconds
            positive_top=half[0]>0.
            frost=3182.48/(23.3494-jnp.log(jnp.where(positive_top,half[0]*.01,1.)))
            floor=jnp.where(positive_top,frost,0.)
            strat=jnp.where(refresh,jnp.maximum(strat,floor),old_strat)
            radiation_fields=dict(radiation_heating_k_s=heating,
                radiation_surface_down_w_m2=down[None],radiation_initialized=jnp.ones_like(down[None]),
                radiation_stratosphere_k=strat[None])
        dust_depth=None
        if dust_control:
            def dust_diagnostic(_):
                radgrid=radiation_grid(half,full,temperature,predicted.surface_temperature[0])
                optics=dust_optics(grid.to_nodal(dyn.tracers['dust_mass'])[...,None],
                    grid.to_nodal(dyn.tracers['dust_number'])[...,None],radgrid.pressure,
                    bands=dust_optical_bands,gravity=body.gravity_m_s2)
                ir_index=3 if dust_optical_bands==12 else 6
                return jnp.stack([jnp.sum(optics.visible[0][:,5],axis=0),
                    jnp.sum((optics.infrared[0]-optics.infrared[1])[:,ir_index],axis=0)])
            dust_depth=(dust_diagnostic(None) if radiation_update_seconds is None else
                jax.lax.cond(refresh,dust_diagnostic,lambda _:predicted.auxiliary_fields['dust_reference_opacity'],None))
        background_options=None;local_time_fields=None
        if background_scenario is not None:
            axis=jnp.asarray(background_scenario['seasons'])
            maps=jnp.asarray(background_scenario['opacity'])
            ls=jnp.mod((_true_anomaly(predicted.dynamics.sim_time*time_scale,forcing)+
                forcing.ls_perihelion_rad)*180./jnp.pi,360.)
            opacity,change,_,_=seasonal_sample(axis,maps,ls,segment=background_segment)
            background_options={**dust_background_options,'scenario_opacity':opacity,'scenario_change':change}
            if local_time_scenario is not None:
                fields=predicted.auxiliary_fields
                history=LocalTimeState(fields['dust_local_time_opacity'][0],fields['dust_local_time_bin'][0])
                arguments=(maps,local_time_scenario['fractions'],axis,local_time_scenario['bins_hours'],
                    ls,jnp.asarray(grid.longitudes)[:,None],predicted.dynamics.sim_time*time_scale/forcing.rotation_period_s)
                local_opacity,fraction,history=jax.lax.cond(fields['dust_local_time_initialized']>0.,
                    lambda _:local_time_step(*arguments,history),lambda _:local_time_step(*arguments,None),None)
                background_options.update(scenario_opacity=local_opacity,local_time_fraction=fraction,
                    dual_scenario_opacity=opacity)
                local_time_fields=dict(dust_local_time_opacity=history.opacity[None],
                    dust_local_time_bin=history.bin_index[None].astype(fields['dust_local_time_bin'].dtype),
                    dust_local_time_initialized=jnp.asarray(1.))
        soil=jnp.concatenate([predicted.surface_temperature,predicted.ground_temperature],axis=0)
        frost=predicted.co2_ice[0]*pressure_scale/body.gravity_m_s2
        water_state=None
        gravity_wave_state=None
        if nonorographic_drag_options is not None:
            fields=predicted.auxiliary_fields
            gravity_wave_state=DragState(fields['nonorographic_u_tendency_m_s2'],fields['nonorographic_v_tendency_m_s2'],
                fields['nonorographic_alarm_seconds'],fields['nonorographic_frequency_seconds'])
        if water_cycle:
            fields=predicted.auxiliary_fields
            water_state=BulkWaterColumnState(gas_names.index(WATER_VAPOR_TRACER),
                grid.to_nodal(dyn.tracers[WATER_CLOUD_TRACER]),fields['water_surface_kg_m2'][0],
                fields['water_exchange_coefficient_kg_m2_s'][0],fields['water_pbl_top_layer'][0])
        # Soil sees post-exchange water; radiation above sees the old reservoir.
        soil_forcing=surface_radiative_forcing(predicted,coords,specs,body,
            dataclasses.replace(forcing,water_frost_radiative_properties=False))
        water_optics=(dict(frost_threshold_m=forcing.water_frost_threshold_m,
            ice_albedo=forcing.water_ice_albedo,liquid_albedo=forcing.water_liquid_albedo,
            water_emissivity=forcing.water_emissivity,liquid_water=forcing.liquid_water_albedo)
            if forcing.water_frost_radiative_properties else None)
        column=column_physics(temperature,u,v,gas_tracers,soil,half,full,frost,heating,down,
            dt_seconds,body,soil_thickness=dz,soil_diffusivity=soil_diffusivity,
            soil_volumetric_heat_capacity=soil_volumetric_heat_capacity,
            emissivity=soil_forcing.emissivity,surface_exchange_start=exchange_start,
            latitude_rad=jnp.asarray(grid.latitudes)[None,:],rayleigh_options=rayleigh_options,
            conserve_drag_energy=conserve_drag_energy,
            suspended_co2_ice_ratio=cloud_ratio if suspended_co2_clouds else None,
            cloud_nuclei_per_kg=cloud_nuclei_per_kg,bulk_water=water_state,
            water_surface_options=water_surface_options,water_cloud_options=water_cloud_options,
            form_bulk_clouds=form_bulk_clouds,water_radiative_options=water_optics,
            terrain_drag_options=terrain_drag_options,
            moment_particles=(jnp.stack([grid.to_nodal(dyn.tracers[name]) for name in
                MOMENT_PARTICLE_TRACERS],axis=-1) if moment_water else None),
            moment_cloud_options=moment_cloud_options,moment_sedimentation=moment_sedimentation,
            dust_lifting_options=dust_lifting_options,
            dust_lifting_opacity=None if dust_depth is None else dust_depth[dust_opacity_channel-1],
            surface_dust_kg_m2=None if not moment_water else predicted.auxiliary_fields['dust_surface_kg_m2'][0],
            coagulation_options=coagulation_options,dust_background_options=background_options,
            nonorographic_drag_options=nonorographic_drag_options,nonorographic_state=gravity_wave_state)
        columns={name:column.gas_tracers[...,i] for i,name in enumerate(gas_names)}
        if cloud_ratio is not None:
            columns[CO2_CLOUD_TRACER]=(column.suspended_co2_ice_ratio
                if suspended_co2_clouds else cloud_ratio)
        if WATER_CLOUD_TRACER in names:
            columns[WATER_CLOUD_TRACER]=(column.bulk_water.ice_ratio if water_cycle else
                grid.to_nodal(dyn.tracers[WATER_CLOUD_TRACER]))
        if moment_water:
            columns.update({name:column.moment_particles[...,index]
                for index,name in enumerate(MOMENT_PARTICLE_TRACERS)})
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
        if WATER_CLOUD_TRACER in names and not water_cycle:
            unsupported_cloud=unsupported_cloud|jnp.any(grid.to_nodal(dyn.tracers[WATER_CLOUD_TRACER])!=0.)
        invalid=jnp.any(column.diagnostics['co2_mass_guard'])|unsupported_cloud
        new_temperature=dyn.temperature_variation+grid.to_modal(t-temperature)
        new_temperature=jnp.where(invalid,jnp.nan,new_temperature)
        dynamics=dataclasses.replace(dyn,temperature_variation=new_temperature,
            vorticity=dyn.vorticity+vor,divergence=dyn.divergence+div,
            log_surface_pressure=dyn.log_surface_pressure+grid.to_modal(jnp.log(new_pressure/pressure)[None]),
            tracers={name:dyn.tracers[name]+grid.to_modal(q[...,i]-tracers[...,i])
                for i,name in enumerate(names)})
        auxiliary=predicted.auxiliary_fields
        if local_time_fields is not None:
            auxiliary={**auxiliary,**local_time_fields}
        if radiation_fields is not None:
            auxiliary={**auxiliary,**radiation_fields}
        if water_cycle:
            auxiliary={**auxiliary,'water_surface_kg_m2':column.bulk_water.surface_water_kg_m2[None],
                'water_exchange_coefficient_kg_m2_s':column.bulk_water.previous_exchange_coefficient[None],
                'water_pbl_top_layer':column.bulk_water.previous_pbl_top_layer[None]}
        if moment_water:
            auxiliary={**auxiliary,'dust_surface_kg_m2':column.diagnostics['bulk_water']['dust_surface_kg_m2'][None]}
            if dust_depth is not None:
                auxiliary={**auxiliary,'dust_reference_opacity':dust_depth}
            if coagulation_options is not None:
                mass_change=column.diagnostics['bulk_water']['dust_coagulation']['dust_mass_change_kg_kg']
                column_change=jnp.sum(mass_change*jnp.diff(half,axis=0)/body.gravity_m_s2,axis=0)
                auxiliary={**auxiliary,'dust_coagulation_mass_change_kg_m2':
                    auxiliary['dust_coagulation_mass_change_kg_m2']+column_change[None]}
        if nonorographic_drag_options is not None:
            drag=column.nonorographic_state
            auxiliary={**({} if auxiliary is None else auxiliary),
                'nonorographic_u_tendency_m_s2':drag.u_tendency,'nonorographic_v_tendency_m_s2':drag.v_tendency,
                'nonorographic_alarm_seconds':drag.alarm_seconds,'nonorographic_frequency_seconds':drag.frequency_seconds}
        return predicted._replace(dynamics=dynamics,surface_temperature=column.soil[:1],
            ground_temperature=column.soil[1:],co2_ice=(column.co2_ice_kg_m2*body.gravity_m_s2/pressure_scale)[None],
            surface_momentum=predicted.surface_momentum+impulse,auxiliary_fields=auxiliary)

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
        if water_cycle:
            diagnostics=diagnostics._replace(raw_water_mass_kg=water_inventory(raw,coords,specs,body),
                corrected_water_mass_kg=water_inventory(final,coords,specs,body),
                target_water_mass_kg=water_inventory(state,coords,specs,body))
        return final,diagnostics
    return audited
