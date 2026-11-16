"""Ames-order column physics for the differentiable Dinosaur backend.

Radiation inputs are supplied explicitly. Gas mass losses are returned for
the dynamical core to transport; this column operator does not remap pressure.
"""
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.co2_condensation import simple_condensation
from src.framework.physics.dry_convection import pass2
from src.framework.physics.boundary_layer import column_step,inactive_diagnostics
from src.framework.physics.soil_frost import surface_step
from src.framework.physics.rayleigh_drag import rayleigh_tendencies
from src.framework.physics.terrain_drag import tendencies as terrain_tendencies,hydrostatic_heights
from src.framework.physics.co2_microphysics import cloud_step,phase_change,transport_concentration
from src.framework.physics.water_bulk import cloud_step as water_cloud_step
from src.framework.physics.water_moments import subcycled_cloud_step,EffectiveState
from src.framework.physics.particle_sedimentation import sedimentation_step
from src.framework.physics.dust_lifting import (interactive_step as dust_lifting_step,background_step,
    interactive_modes_step,background_modes_step)
from src.framework.physics.dust_coagulation import coagulation_step
from src.framework.physics.nonorographic_drag import driver_step as gravity_wave_step
from src.framework.physics.legacy_convection import adjust as legacy_convection
from src.framework.physics.moist_convection import adjust as moist_convection
from src.framework.physics.water_surface import (surface_exchange as water_surface_exchange,
    optical_properties as water_optical_properties)


class BulkWaterColumnState(NamedTuple):
    vapor_index: object
    ice_ratio: object
    surface_water_kg_m2: object
    previous_exchange_coefficient: object
    previous_pbl_top_layer: object


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
    bulk_water: object = None
    moment_particles: object = None
    nonorographic_state: object = None
    moment_history: object = None
    moist_surface_water_kg_m2: object = None


def column_physics(temperature,u,v,gas_tracers,soil,pressure_half,pressure_full,
                   co2_ice_kg_m2,radiative_heating_k_s,downward_surface_flux_w_m2,
                   dt_seconds,body,*,soil_thickness,soil_diffusivity,
                   soil_volumetric_heat_capacity,emissivity,latent_heat=5.9e5,
                   surface_exchange_start,latitude_rad=0.,rayleigh_options=None,
                   conserve_drag_energy=False,suspended_co2_ice_ratio=None,
                   cloud_nuclei_per_kg=1.e5,bulk_water=None,
                   water_surface_options=None,water_cloud_options=None,form_bulk_clouds=True,
                   water_radiative_options=None,terrain_drag_options=None,
                   moment_particles=None,moment_cloud_options=None,moment_sedimentation=True,
                   dust_lifting_options=None,dust_lifting_opacity=None,surface_dust_kg_m2=None,
                   coagulation_options=None,dust_background_options=None,
                   nonorographic_drag_options=None,nonorographic_state=None,moment_history=None,
                   convection_scheme='fv3',moist_convection_options=None,moist_surface_water_kg_m2=None,
                   co2_gas_index=None,do_co2_condensation=True,do_atmos_co2_loss=True,
                   do_co2_condensation_cycle=True,do_mars_surface=True,do_vert_diff=True,do_ames_pbl=True):
    """Rayleigh/PBL, soil/frost, CO2 deposition or suspended clouds, convection.

    This is the standard present-Mars gas-column process order. Tracers here
    are gas species only. Optional CO2 cloud ice is supplied separately and
    excluded from gas PBL mixing. Convection mixes it with the other tracers,
    as in the source. Match the surface exchange start
    index to the reference grid before applying the returned gas mass losses.
    """
    if not isinstance(surface_exchange_start,int) or not 0<=surface_exchange_start<temperature.shape[0]:
        raise ValueError("surface mass exchange start must be a valid layer index")
    if convection_scheme not in ('none','fv3','legacy'):
        raise ValueError('convection scheme must be none, fv3 or legacy')
    if co2_gas_index is not None and (not isinstance(co2_gas_index,int) or
        not 0<=co2_gas_index<gas_tracers.shape[-1]):
        raise ValueError('CO2 gas index must select a gas tracer')
    if moist_convection_options is not None and bulk_water is None:
        raise ValueError('moist convection requires a water vapor tracer and surface water reservoir')
    if moist_convection_options is not None and moment_particles is not None and moist_surface_water_kg_m2 is None:
        raise ValueError('moment-water moist convection requires its separate bulk precipitation reservoir')
    if not do_mars_surface:
        if bulk_water is not None or moment_particles is not None:
            raise ValueError('the released no-surface branch does not initialize water/dust surface reservoirs')
        soil=jnp.asarray(soil).at[0].set(170.)
        emissivity=1.
        do_co2_condensation=False;do_co2_condensation_cycle=False
    initial_wind=(u,v)
    if moment_particles is None and (dust_lifting_options is not None or coagulation_options is not None or dust_background_options is not None):
        raise ValueError('dust lifting and coagulation require physical moment particles')
    initial_moments=None
    if moment_particles is not None:
        if bulk_water is None:
            raise ValueError('moment particles require the water reservoir and vapor state')
        if (moment_particles.shape[:-1]!=temperature.shape or moment_particles.shape[-1]<4 or
            (moment_particles.shape[-1]-4)%3):
            raise ValueError('moment particles require four entries then extra dust mass/number/core triples')
        particle_count=moment_particles.shape[-1]
        dust_mass_indices=jnp.asarray([0]+list(range(4,particle_count,3)))
        dust_number_indices=dust_mass_indices+1
        initial_dust_mass=moment_particles[...,dust_mass_indices]
        initial_dust_number=moment_particles[...,dust_number_indices]
        initial_moments=jnp.stack([moment_particles[...,0],moment_particles[...,1],
            bulk_water.ice_ratio,moment_particles[...,2],moment_particles[...,3],
            gas_tracers[...,bulk_water.vapor_index]],axis=-1)
    predicted=jnp.asarray(temperature)+jnp.asarray(radiative_heating_k_s)*dt_seconds
    water_exchange=None
    if bulk_water is not None:
        gas_tracers=jnp.asarray(gas_tracers)
        if isinstance(bulk_water.vapor_index,int) and not 0<=bulk_water.vapor_index<gas_tracers.shape[-1]:
            raise ValueError('bulk water vapor index must select a gas tracer')
        water_exchange=water_surface_exchange(gas_tracers[...,bulk_water.vapor_index],
            pressure_half,soil[0],bulk_water.surface_water_kg_m2,
            bulk_water.previous_exchange_coefficient,bulk_water.previous_pbl_top_layer,
            dt_seconds,gravity=body.gravity_m_s2,
            **({} if water_surface_options is None else water_surface_options))
        gas_tracers=gas_tracers.at[...,bulk_water.vapor_index].set(water_exchange.vapor_ratio)
    du,dv=rayleigh_tendencies(u,v,pressure_full,pressure_half[-1],latitude_rad,
        **({} if rayleigh_options is None else rayleigh_options))
    drag_heat=-((u+.5*du*dt_seconds)*du+(v+.5*dv*dt_seconds)*dv)*dt_seconds/body.cp_j_kg_k
    if conserve_drag_energy:
        predicted=predicted+drag_heat
    if do_mars_surface and do_vert_diff and do_ames_pbl:
        t,u,v,r,pbl=column_step(predicted,u+du*dt_seconds,v+dv*dt_seconds,gas_tracers,
            pressure_half,pressure_full,soil[0],
            dt_seconds,body.gas_constant_j_kg_k,body.cp_j_kg_k,body.gravity_m_s2,
            surface_co2_ice=co2_ice_kg_m2,
            surface_water_ice=0. if bulk_water is None else bulk_water.surface_water_kg_m2)
    else:
        t,u,v,r=predicted,u+du*dt_seconds,v+dv*dt_seconds,gas_tracers
        pbl=inactive_diagnostics(t,u,v)
    pbl_wind=(u,v)
    if water_exchange is not None and water_radiative_options is not None:
        _,emissivity=water_optical_properties(0.,emissivity,soil[0],co2_ice_kg_m2,
            water_exchange.surface_water_kg_m2,**water_radiative_options)
    if do_mars_surface:
        ground,snow,surface=surface_step(soil,soil_thickness,soil_diffusivity,
            soil_volumetric_heat_capacity,pressure_half[-1],co2_ice_kg_m2,
            downward_surface_flux_w_m2,pbl['heat_flux_w_m2'],
            pbl['heat_flux_surface_derivative_w_m2_k'],emissivity,dt_seconds,latent_heat=latent_heat,
            bulk_latent_flux=0. if water_exchange is None else water_exchange.latent_flux_w_m2,
            bulk_latent_derivative=0. if water_exchange is None else water_exchange.latent_derivative_w_m2_k)
    else:
        ground=soil;snow=co2_ice_kg_m2
        surface=dict(co2_phase_change_kg_m2=jnp.zeros_like(soil[0]),
            frost_temperature_k=3182.48/(23.3494-jnp.log(pressure_half[-1]*.01)),
            frost_retry=jnp.zeros_like(soil[0],dtype=bool))
    surface['emissivity']=emissivity
    cloud=None
    has_co2_cloud=suspended_co2_ice_ratio is not None
    cloud_diagnostics={}
    if not do_co2_condensation:
        mass=jnp.zeros_like(t);precip=jnp.zeros_like(t[-1]);guard=jnp.zeros_like(t,dtype=bool)
    elif not has_co2_cloud:
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
    if not do_atmos_co2_loss:
        mass=jnp.zeros_like(mass);precip=jnp.zeros_like(precip)
    terrain=None
    if terrain_drag_options is not None:
        options=dict(terrain_drag_options)
        top_pressure=options.pop('top_pressure_pa')
        factor=options.pop('factor',1.)
        z,zh=hydrostatic_heights(temperature,pressure_half,pressure_full,
            gas_constant=body.gas_constant_j_kg_k,gravity=body.gravity_m_s2,top_pressure_pa=top_pressure)
        terrain=terrain_tendencies(u,v,t,pressure_full,pressure_half,z,zh,
            gas_constant=body.gas_constant_j_kg_k,cp=body.cp_j_kg_k,
            gravity=body.gravity_m_s2,**options)
        u=u+dt_seconds*factor*terrain.u_tendency
        v=v+dt_seconds*factor*terrain.v_tendency
    nonorographic=None
    if nonorographic_drag_options is not None:
        if nonorographic_state is None:
            raise ValueError('non-orographic drag requires initialized forcing and timer state')
        options=dict(nonorographic_drag_options)
        z,_=hydrostatic_heights(temperature,pressure_half,pressure_full,
            gas_constant=body.gas_constant_j_kg_k,gravity=body.gravity_m_s2,
            top_pressure_pa=options.pop('top_pressure_pa'))
        source_index=options.pop('source_index');source_amplitude=options.pop('source_amplitude')
        nonorographic,nonorographic_state=gravity_wave_step(t,pressure_full,z,u,v,
            source_index,source_amplitude,dt_seconds,nonorographic_state,
            gas_constant=body.gas_constant_j_kg_k,gravity=body.gravity_m_s2,cp=body.cp_j_kg_k,**options)
        u=u+dt_seconds*nonorographic.u_tendency
        v=v+dt_seconds*nonorographic.v_tendency
        t=t+dt_seconds*nonorographic.heating_k_s
    gas_count=r.shape[-1]
    particles=[]
    if has_co2_cloud:
        particles.append((cloud.ice_ratio if cloud is not None else jnp.asarray(suspended_co2_ice_ratio))[...,None])
    if bulk_water is not None: particles.append(jnp.asarray(bulk_water.ice_ratio)[...,None])
    moment_start=gas_count+len(particles)
    if moment_particles is not None: particles.append(moment_particles)
    if particles: r=jnp.concatenate([r,*particles],axis=-1)
    before_convection=(t,u,v,r)
    if convection_scheme=='fv3':
        t,u,v,r=pass2(t,u,v,r,pressure_half,pressure_full,body.kappa)
    elif convection_scheme=='legacy':
        t,u,v,r=legacy_convection(t,u,v,r,pressure_half,pressure_full,body.kappa,gravity=body.gravity_m_s2)
    carrier=jnp.diff(pressure_half,axis=0)/body.gravity_m_s2
    t_old,u_old,v_old,r_old=before_convection
    convection=dict(scheme_index=('none','fv3','legacy').index(convection_scheme),
        enthalpy_change_j_m2=jnp.sum(carrier*body.cp_j_kg_k*(t-t_old),axis=0),
        kinetic_energy_change_j_m2=.5*jnp.sum(carrier*(u*u+v*v-u_old*u_old-v_old*v_old),axis=0),
        u_momentum_change_kg_m_s=jnp.sum(carrier*(u-u_old),axis=0),
        v_momentum_change_kg_m_s=jnp.sum(carrier*(v-v_old),axis=0),
        tracer_column_change=jnp.sum(carrier[...,None]*(r-r_old),axis=0))
    moist=None;moist_precip=0.
    if moist_convection_options is not None:
        moist=moist_convection(t,r[...,bulk_water.vapor_index],pressure_half,pressure_full,dt_seconds,
            gas_constant=body.gas_constant_j_kg_k,cp=body.cp_j_kg_k,gravity=body.gravity_m_s2,
            **moist_convection_options)
        t=t+dt_seconds*moist.temperature_tendency
        r=r.at[...,bulk_water.vapor_index].add(dt_seconds*moist.vapor_tendency)
        moist_precip=dt_seconds*(moist.rain_kg_m2_s+moist.snow_kg_m2_s)
        if moment_particles is not None:
            bank=jnp.asarray(moist_surface_water_kg_m2)
            bank=jnp.where(jnp.isfinite(bank)&(bank>=0.),bank,jnp.nan)
            moist_surface_water_kg_m2=bank+moist_precip
    active_moist_precip=moist_precip if moment_particles is None else 0.
    average=.5*(pbl['heat_diffusivity_m2_s']+pbl['momentum_diffusivity_m2_s'])
    connected=jnp.cumprod((average[::-1]>.01).astype(int),axis=0)
    pbl_top=temperature.shape[0]-1-jnp.sum(connected,axis=0)
    suspended=None
    if has_co2_cloud:
        suspended=r[...,gas_count]
    water_diagnostics={};water_result=None
    if bulk_water is not None:
        ice=r[...,gas_count+int(has_co2_cloud)]
        water_deposit=0.
        if moment_particles is not None:
            moment_particles=r[...,moment_start:moment_start+particle_count]
            if dust_lifting_options is not None:
                if dust_lifting_opacity is None or surface_dust_kg_m2 is None:
                    raise ValueError('interactive dust requires reference opacity and a surface reservoir')
                rho=pressure_half[-1]/(body.gas_constant_j_kg_k*temperature[-1])
                drag=rho*pbl['surface_momentum_coefficient']*pbl['wind_speed_ms']
                speed_squared=u[-1]**2+v[-1]**2
                speed=jnp.where(speed_squared>0.,jnp.sqrt(jnp.where(speed_squared>0.,speed_squared,1.)),0.)
                stress=drag*speed
                p_pbl=jnp.take_along_axis(pressure_full,pbl_top[None],axis=0)[0]
                lifting_options=dict(dust_lifting_options)
                if dust_background_options is not None:
                    if lifting_options.get('dual_mode',0)<=0:
                        raise ValueError('combined interactive/background dust requires a positive dual mode')
                    lifting_options['scenario_opacity']=dust_background_options.get('dual_scenario_opacity',
                        dust_background_options['scenario_opacity'])
                lift=interactive_modes_step if particle_count>4 else dust_lifting_step
                lifted=lift(moment_particles[...,dust_mass_indices] if particle_count>4 else moment_particles[...,0],
                    moment_particles[...,dust_number_indices] if particle_count>4 else moment_particles[...,1],t,
                    pressure_half,pressure_full,ground[0],stress,pbl['heat_flux_w_m2'],pbl_top,p_pbl,
                    latitude_rad,snow,water_exchange.surface_water_kg_m2+active_moist_precip,
                    jnp.maximum(dust_lifting_opacity/pressure_full[-1]*610.,1e-10),
                    surface_dust_kg_m2=surface_dust_kg_m2,dt_seconds=dt_seconds,gravity=body.gravity_m_s2,
                    gas_constant=body.gas_constant_j_kg_k,cp=body.cp_j_kg_k,**lifting_options)
                moment_particles=moment_particles.at[...,dust_mass_indices if particle_count>4 else 0].set(lifted.mass_ratio)
                moment_particles=moment_particles.at[...,dust_number_indices if particle_count>4 else 1].set(lifted.number_ratio)
                surface_dust_kg_m2=lifted.surface_dust_kg_m2
            if dust_background_options is not None:
                if dust_lifting_opacity is None or surface_dust_kg_m2 is None:
                    raise ValueError('background dust requires reference opacity and a surface reservoir')
                background_lift=background_modes_step if particle_count>4 else background_step
                background=background_lift(moment_particles[...,dust_mass_indices] if particle_count>4 else moment_particles[...,0],
                    moment_particles[...,dust_number_indices] if particle_count>4 else moment_particles[...,1],pressure_half,
                    pbl_top,ground[0],snow,jnp.maximum(dust_lifting_opacity/pressure_full[-1]*610.,1e-10),
                    surface_dust_kg_m2=surface_dust_kg_m2,dt_seconds=dt_seconds,
                    gravity=body.gravity_m_s2,**{key:value for key,value in dust_background_options.items()
                        if key!='dual_scenario_opacity'})
                moment_particles=moment_particles.at[...,dust_mass_indices if particle_count>4 else 0].set(background.mass_ratio)
                moment_particles=moment_particles.at[...,dust_number_indices if particle_count>4 else 1].set(background.number_ratio)
                surface_dust_kg_m2=background.surface_dust_kg_m2
            if coagulation_options is not None:
                coagulated=coagulation_step(t,pressure_full,moment_particles[...,dust_mass_indices],
                    moment_particles[...,dust_number_indices],dt_seconds,gravity=body.gravity_m_s2,
                    gas_constant=body.gas_constant_j_kg_k,**coagulation_options)
                moment_particles=moment_particles.at[...,dust_mass_indices].add(dt_seconds*coagulated.mass_tendency)
                moment_particles=moment_particles.at[...,dust_number_indices].add(dt_seconds*coagulated.number_tendency)
            moments=jnp.stack([moment_particles[...,0],moment_particles[...,1],ice,
                moment_particles[...,2],moment_particles[...,3],r[...,bulk_water.vapor_index]],axis=-1)
            moment_result=subcycled_cloud_step(temperature,t,pressure_full,initial_moments,moments,
                dt_seconds,cp=body.cp_j_kg_k,previous_state=moment_history,
                initial_dust_mass=initial_dust_mass,initial_dust_number=initial_dust_number,
                dust_mass=moment_particles[...,dust_mass_indices],dust_number=moment_particles[...,dust_number_indices],
                **({} if moment_cloud_options is None else moment_cloud_options))
            if moment_cloud_options is not None and moment_cloud_options.get('effective_input_mode',0)==1:
                # Source saves post-cloud T but pre-cloud Q; its dust history is never filled.
                moment_history=EffectiveState(t+dt_seconds*moment_result.temperature_tendency,
                    moments,moment_particles[...,dust_mass_indices],moment_particles[...,dust_number_indices])
            moment_particles=moment_particles.at[...,dust_mass_indices].add(dt_seconds*moment_result.dust_mass_tendency)
            moment_particles=moment_particles.at[...,dust_number_indices].add(dt_seconds*moment_result.dust_number_tendency)
            moments=moments+dt_seconds*moment_result.tracer_tendency
            t=t+dt_seconds*moment_result.temperature_tendency
            moments=jnp.concatenate([moments,moment_particles[...,4:]],axis=-1)
            if moment_sedimentation:
                sediment=sedimentation_step(moments,pressure_half,pressure_full,t,ground[0],
                    pbl['heat_diffusivity_m2_s'],dt_seconds,
                    gas_constant=body.gas_constant_j_kg_k,gravity=body.gravity_m_s2)
                moments=sediment.tracers
                water_deposit=sediment.surface_deposit[...,5]
            moment_particles=jnp.concatenate([jnp.stack([moments[...,0],moments[...,1],moments[...,3],moments[...,4]],axis=-1),
                moments[...,6:]],axis=-1)
            ice=moments[...,2]
            r=r.at[...,bulk_water.vapor_index].set(moments[...,5])
        elif form_bulk_clouds:
            water_result=water_cloud_step(t,pressure_half,pressure_full,r[...,bulk_water.vapor_index],
                ice,ground[0],pbl['heat_diffusivity_m2_s'],dt_seconds,
                gas_constant=body.gas_constant_j_kg_k,gravity=body.gravity_m_s2,cp=body.cp_j_kg_k,
                **({} if water_cloud_options is None else water_cloud_options))
            r=r.at[...,bulk_water.vapor_index].set(water_result.vapor_ratio)
            t=water_result.temperature
            ice=water_result.ice_ratio
            water_deposit=water_result.deposition_kg_m2
        rho=pressure_half[-1]/(body.gas_constant_j_kg_k*temperature[-1])
        exchange_coefficient=rho*pbl['surface_heat_coefficient']*pbl['friction_velocity_ms']
        bulk_water=BulkWaterColumnState(bulk_water.vapor_index,ice,
            water_exchange.surface_water_kg_m2+water_deposit+active_moist_precip,
            exchange_coefficient,pbl_top.astype(t.dtype))
        water_diagnostics=dict(surface=water_exchange.diagnostics,
            cloud={} if water_result is None else water_result.diagnostics,
            upward_surface_flux_kg_m2_s=water_exchange.upward_flux_kg_m2_s)
        if moment_particles is not None:
            water_diagnostics['moment_cloud']=dict(moment_result._asdict())
            if dust_lifting_options is not None:
                water_diagnostics['dust_lifting']=lifted.diagnostics
            if dust_background_options is not None:
                water_diagnostics['dust_background']=background.diagnostics
            if coagulation_options is not None:
                water_diagnostics['dust_coagulation']=coagulated.diagnostics
            if moment_sedimentation:
                dust_deposit=sediment.surface_deposit[...,jnp.asarray([0]+list(range(6,moments.shape[-1],3)))]
                if particle_count==4: dust_deposit=dust_deposit[...,0]
                water_diagnostics['particle_sedimentation']=dict(sediment.diagnostics,
                    dust_deposit_kg_m2=dust_deposit)
                if surface_dust_kg_m2 is not None:
                    surface_dust_kg_m2=surface_dust_kg_m2+dust_deposit
            if surface_dust_kg_m2 is not None:
                water_diagnostics['dust_surface_kg_m2']=surface_dust_kg_m2
    r=r[...,:gas_count]
    dp=jnp.diff(pressure_half,axis=0)
    if do_co2_condensation_cycle:
        selected=jnp.asarray(np.arange(temperature.shape[0])>=surface_exchange_start)
        selected=selected.reshape((temperature.shape[0],)+(1,)*(dp.ndim-1))
        weights=jnp.where(selected,dp,0.)
        if co2_gas_index is not None:
            weights=weights*gas_tracers[...,co2_gas_index]
        weights=weights/jnp.sum(weights,axis=0,keepdims=True)
        surface_mass=surface['co2_phase_change_kg_m2']*weights
        mass=mass+surface_mass
    else:
        surface_mass=jnp.zeros_like(mass);mass=jnp.zeros_like(mass)
    if has_co2_cloud:
        gas=dp/body.gravity_m_s2
        # Only drag, surface phase exchange and deposited ice torque the ground.
        ground_impulse=jnp.stack([jnp.sum(gas*(before-after)+surface_mass*after,axis=0)
            for before,after in zip(initial_wind,pbl_wind,strict=True)])
        ground_impulse=ground_impulse+cloud_diagnostics.get('deposition_momentum_kg_m_s',0.)
        cloud_diagnostics['ground_momentum_exchange_kg_m_s']=ground_impulse
        old_mass=gas*(1.+jnp.asarray(suspended_co2_ice_ratio))
        new_mass=gas-mass+gas*suspended
        cloud_diagnostics['unresolved_momentum_source_kg_m_s']=ground_impulse+jnp.stack([
            jnp.sum(new_mass*after-old_mass*before,axis=0)
            for before,after in zip(initial_wind,(u,v),strict=True)])
    diagnostics=dict(pbl=pbl,surface=surface,co2_mass_guard=guard,
        atmospheric_co2_precipitation_kg_m2=precip,
        co2_cloud=cloud_diagnostics,
        bulk_water=water_diagnostics,
        terrain_drag={} if terrain is None else dict(terrain._asdict()),
        nonorographic_drag={} if nonorographic is None else nonorographic.diagnostics,
        convection=convection,
        moist_convection={} if moist is None else moist._asdict(),
        rayleigh=dict(u_tendency_m_s2=du,v_tendency_m_s2=dv,dissipated_heat_k=drag_heat,
            reheating_enabled=conserve_drag_energy),
        co2_controls=dict(condensation=do_co2_condensation,atmospheric_loss=do_atmos_co2_loss,
            mass_feedback=do_co2_condensation_cycle))
    return ColumnResult(t,u,v,r,ground,snow+precip,mass,diagnostics,suspended,bulk_water,
        moment_particles,nonorographic_state,moment_history,moist_surface_water_kg_m2)
