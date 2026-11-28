"""Restartable step means and three-hour local-solar-time bins.

The counters and sums live in the ordinary restart auxiliary fields. Samples
are end-of-step values with equal duration; model steps must divide a sol.
"""
import numpy as np
from src.framework.gcm._dinosaur import jnp,scales,spherical_harmonic
from src.framework.gcm.dynamics import reference_temperature
from src.framework.physics.tracer_transport import tracer_to_nodal
from src.framework.physics.water_bulk import WATER_VAPOR_TRACER,WATER_CLOUD_TRACER


def fields(state,coords,specs,body):
    grid=coords.horizontal
    velocity=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    pressure=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,state.dynamics.vorticity,state.dynamics.divergence)
    result=dict(surface_pressure=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*pressure,
        surface_temperature=state.surface_temperature[0],co2_frost=state.co2_ice[0]*pressure,
        air_temperature=grid.to_nodal(state.dynamics.temperature_variation)+jnp.asarray(reference_temperature(coords,body))[:,None,None],
        eastward_wind=u*velocity,northward_wind=v*velocity,wind_speed=jnp.hypot(u,v)*velocity)
    result.update({'tracer_'+name:tracer_to_nodal(grid,value) for name,value in state.dynamics.tracers.items()})
    water=[result['tracer_'+name] for name in (WATER_VAPOR_TRACER,WATER_CLOUD_TRACER)
        if 'tracer_'+name in result]
    if water:
        layer_mass=jnp.diff(jnp.asarray(coords.vertical.boundaries))[:,None,None]*result['surface_pressure']/body.gravity_m_s2
        result['water_atmospheric_column']=jnp.sum(sum(water)*layer_mass,axis=0)
    return result


def averaging_step(step,state,coords,specs,body,window_steps,dt_seconds):
    if not isinstance(window_steps,int) or window_steps<1:
        raise ValueError('comparison window must contain a positive integer number of steps')
    if not np.isclose(body.solar_day_seconds/dt_seconds,round(body.solar_day_seconds/dt_seconds)):
        raise ValueError('comparison averaging requires whole steps per sol')
    values=fields(state,coords,specs,body)
    auxiliary=dict(state.auxiliary_fields or {})
    prefix='comparison_'
    if prefix+'count' not in auxiliary:
        auxiliary[prefix+'count']=jnp.asarray(0)
        auxiliary[prefix+'bin_count']=jnp.zeros((8,coords.horizontal.nodal_shape[0]))
    for name,value in values.items():
        if prefix+'sum_'+name in auxiliary: continue
        if int(auxiliary[prefix+'count']) not in (0,window_steps):
            raise ValueError(f'comparison restart lacks {name} samples in its partial averaging window')
        auxiliary[prefix+'sum_'+name]=jnp.zeros_like(value)
        # Diurnal climate diagnostics use the lowest atmospheric layer.
        surface=value[-1] if value.ndim==3 else value
        auxiliary[prefix+'bin_'+name]=jnp.zeros((8,*surface.shape))
    saved=auxiliary.get(prefix+'window_steps')
    if saved is not None and int(saved)!=window_steps:
        raise ValueError('comparison averaging window differs from the restart')
    auxiliary[prefix+'window_steps']=jnp.asarray(window_steps)
    state=state._replace(auxiliary_fields=auxiliary)
    time_scale=float(specs.dimensionalize(1.,scales.units.second).magnitude)

    def advance(state):
        final=step(state)
        old=state.auxiliary_fields
        reset=old[prefix+'count']==window_steps
        result=dict(final.auxiliary_fields)
        result[prefix+'count']=jnp.where(reset,0,old[prefix+'count'])+1
        hours=jnp.mod(12.+12.*jnp.asarray(coords.horizontal.longitudes)/jnp.pi+
            24.*final.dynamics.sim_time*time_scale/body.solar_day_seconds,24.)
        selected=jnp.arange(8)[:,None]==jnp.floor(hours/3.).astype(int)[None,:]
        result[prefix+'bin_count']=jnp.where(reset,0.,old[prefix+'bin_count'])+selected
        for name,value in fields(final,coords,specs,body).items():
            result[prefix+'sum_'+name]=jnp.where(reset,0.,old[prefix+'sum_'+name])+value
            surface=value[-1] if value.ndim==3 else value
            result[prefix+'bin_'+name]=jnp.where(reset,0.,old[prefix+'bin_'+name])+selected[...,None]*surface
        return final._replace(auxiliary_fields=result)
    return state,advance


def apply_means(dataset,state,body,dt_seconds):
    aux=state.auxiliary_fields
    count=int(aux['comparison_count'])
    if count<1: raise ValueError('comparison average contains no samples')
    dataset=dataset.drop_vars([name for name in dataset.data_vars
        if "comparison_sum_"+name not in aux and name!="solar_longitude"]).copy()
    dataset.solar_longitude.attrs["sampling"]="end of averaging window"
    for name in list(dataset.data_vars):
        key='comparison_sum_'+name
        if key not in aux: continue
        value=np.asarray(aux[key])/count
        value=value.transpose(0,2,1) if value.ndim==3 else value.T
        dataset[name].data=value[None]
        dataset[name].attrs['cell_methods']='time: mean'
        bins=np.asarray(aux['comparison_bin_'+name])
        counts=np.asarray(aux['comparison_bin_count'])
        mean=np.divide(bins,counts[...,None],out=np.full_like(bins,np.nan),where=counts[...,None]>0.)
        dataset['diurnal_'+name]=(('local_time','lat','lon'),mean.transpose(0,2,1),
            dict(dataset[name].attrs,description='Lowest atmospheric layer for 3-D fields'))
    dataset=dataset.assign_coords(local_time=('local_time',np.arange(1.5,24.,3.),{'units':'hour'}))
    dataset['local_time_sample_count']=(('local_time','lon'),np.asarray(aux['comparison_bin_count']))
    end=float(dataset.time.values[0])
    dataset['time_bounds']=(('time','bounds'),[[end-count*dt_seconds/body.solar_day_seconds,end]])
    dataset.time.attrs['bounds']='time_bounds'
    dataset.attrs.update(temporal_sampling='step mean; three-hour local-solar-time surface bins',
        averaging_steps=count,complete_window=int(count==int(aux['comparison_window_steps'])))
    return dataset
