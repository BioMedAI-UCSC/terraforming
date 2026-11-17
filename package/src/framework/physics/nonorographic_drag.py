"""Alexander–Dunkerton non-orographic gravity-wave forcing in JAX.

SPDX-License-Identifier: LGPL-3.0-only
Adapted from NOAA-GFDL atmos_phys cg_drag, revision 46817a1. Mars parameters
follow NASA's Apache-2.0 patch. See licenses/NONOROGRAPHIC-DRAG-NOTICE.txt.
"""
from functools import lru_cache
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp


class WaveResult(NamedTuple):
    wind_tendency: object
    eddy_diffusivity: object
    diagnostics: object


class DragResult(NamedTuple):
    u_tendency: object
    v_tendency: object
    heating_k_s: object
    diagnostics: object


class DragState(NamedTuple):
    u_tendency: object
    v_tendency: object
    alarm_seconds: object
    frequency_seconds: object


def schedule(update_seconds=0,offset_seconds=0):
    """Validate the source integer frequency and offset without a time cutoff."""
    for value in (update_seconds,offset_seconds):
        if not np.isfinite(value) or value<0 or value!=int(value):
            raise ValueError('gravity-wave frequency and offset must be nonnegative whole seconds')
    return int(update_seconds),int(offset_seconds)


def initial_drag_state(template,*,update_seconds=0,offset_seconds=0):
    frequency,offset=schedule(update_seconds,offset_seconds)
    zero=jnp.zeros_like(template)
    return DragState(zero,zero,jnp.asarray(float(offset if offset>0 else frequency)),
        jnp.asarray(float(frequency)))


@lru_cache(maxsize=8)
def spectrum(maximum_speed=72.,speed_interval=1.2,wavelength_count=1):
    """Original spectrum with separate IEEE64 multiply/subtract operations.

    Materialize constants on the host. A contracted multiply/add moves the
    nominal zero phase speed and changes the source's exact critical-level test.
    """
    if not np.isfinite(maximum_speed) or maximum_speed<=0. or not np.isfinite(speed_interval) or speed_interval<=0.:
        raise ValueError('wave speed bound and interval must be finite and positive')
    if not isinstance(wavelength_count,int) or isinstance(wavelength_count,bool) or wavelength_count<1:
        raise ValueError('wave spectrum requires a positive integer wavelength count')
    count=int(2.*maximum_speed/speed_interval+1.)
    speeds=np.arange(count,dtype=np.float64)*speed_interval-maximum_speed
    wavenumbers=2.*np.pi/((30.*10.**np.arange(1,wavelength_count+1))*1e3)
    return speeds,wavenumbers


def source_parameters(reference_pressure_half_pa,latitude_rad,*,source_pressure_pa=150.,
                      base_amplitude=0.,equatorial_amplitude=.08,equatorial_width=45.):
    """Patched source level and amplitude on the reference interface grid.

    The returned source index includes the extra top ghost level. The patched
    latitude formula divides radians by the literal width 45; retain that rule.
    """
    pressure=np.asarray(reference_pressure_half_pa,dtype=float)
    if pressure.ndim!=1 or len(pressure)<3 or not np.all(np.isfinite(pressure)) or \
            pressure[0]<0. or np.any(pressure[1:]<=0.) or np.any(np.diff(pressure)<=0.):
        raise ValueError('gravity-wave source requires increasing reference pressure interfaces')
    selected=np.flatnonzero(pressure[:-1]>source_pressure_pa)
    if not len(selected): raise ValueError('source pressure must be below a reference model level')
    level=min(int(selected[0])+1,len(pressure)-2)
    lat=jnp.asarray(latitude_rad)
    amplitude=base_amplitude+equatorial_amplitude*(1.-jnp.tanh(lat/equatorial_width)**2)
    return level,amplitude


def wave_forcing(density,wind,buoyancy_frequency,height,source_index,source_amplitude,*,
                 maximum_speed=72.,speed_interval=1.2,wavelength_count=1,
                 spectrum_width=25.,wide_amplitude=.4,narrow_amplitude=0.,
                 stationary_spectrum=True,dump_remaining_flux=False):
    """Original gwfc on extended level-first arrays, including top ghost index 0.

    Source indices select these extended arrays. Model levels start at index 1.
    Breaking and reflection retain the source's discrete phase-speed masks.
    With top dumping, its last relative phase speed is used for every remaining
    wave's diffusion term, exactly as in the original routine.
    """
    rho,u,bf,z=map(jnp.asarray,(density,wind,buoyancy_frequency,height))
    if rho.shape!=u.shape or bf.shape!=u.shape or z.shape!=u.shape or u.shape[0]<3:
        raise ValueError('wave forcing requires matching profiles with an extra top ghost level')
    source=jnp.broadcast_to(jnp.asarray(source_index,dtype=int),u.shape[1:])
    valid=(source>=1)&(source<u.shape[0]-1)
    safe_source=jnp.clip(source,1,u.shape[0]-2)
    source_u=jnp.take_along_axis(u,safe_source[None],axis=0)[0]
    source_rho=jnp.take_along_axis(rho,safe_source[None],axis=0)[0]
    speeds,wavenumbers=spectrum(maximum_speed,speed_interval,wavelength_count)
    speeds=jnp.asarray(speeds)
    relative_source=speeds-source_u[...,None]
    phase=speeds if stationary_spectrum else relative_source
    magnitude=(wide_amplitude+narrow_amplitude)*jnp.exp(-jnp.log(2.)*(phase/spectrum_width)**2)
    flux=jnp.where(relative_source==0.,0.,jnp.where(relative_source<0.,-magnitude,magnitude))
    normalization=jnp.sum(jnp.abs(flux),axis=-1)
    intermittency=jnp.asarray(source_amplitude)*1.5/wavelength_count/jnp.where(normalization>0.,normalization,1.)
    valid=valid&(normalization>0.)
    total_forcing=jnp.zeros_like(u);total_diffusion=jnp.zeros_like(u)
    for wavenumber in wavenumbers:
        def level(state,k):
            mask,last_relative=state
            inside=k<=source
            relative=speeds-u[k,...,None]
            safe_relative=jnp.where(relative!=0.,relative,1.)
            dz=z[k]-z[k+1]
            alpha_squared=(jnp.log(rho[k]/rho[k+1])/(2.*dz))**2
            critical=jnp.sqrt(bf[k]**2*wavenumber**2/(wavenumber**2+alpha_squared))
            reflected=jnp.abs(relative)*wavenumber>=critical[...,None]
            threshold=.5*(rho[k]/source_rho)*wavenumber/bf[k]
            breaks=(relative_source*relative<=0.)|(flux/safe_relative**3>=threshold[...,None])
            removed=(relative==0.)|reflected|breaks
            deposit=mask&(relative!=0.)&~reflected&breaks&inside[...,None]&(k<source)[...,None]
            momentum=jnp.sum(jnp.where(deposit,flux,0.),axis=-1)
            energy=jnp.sum(jnp.where(deposit,relative*flux,0.),axis=-1)
            indices=jnp.arange(len(speeds))
            last=jnp.max(jnp.where(mask,indices,-1),axis=-1)
            candidate=jnp.take_along_axis(relative,jnp.maximum(last,0)[...,None],axis=-1)[...,0]
            last_relative=jnp.where(inside&(last>=0),candidate,last_relative)
            mask=jnp.where(inside[...,None],mask&~removed,mask)
            if dump_remaining_flux:
                remaining=jnp.sum(jnp.where(mask,flux,0.),axis=-1)
                momentum=momentum+jnp.where((k==0)&(k<source),remaining,0.)
                energy=energy+jnp.where((k==0)&(k<source),last_relative*remaining,0.)
            factor=source_rho/jnp.sqrt(rho[k]*rho[k+1])*intermittency/dz
            force=jnp.where(k<source,factor*momentum,0.)
            diffusion=jnp.where(k<source,factor*energy/bf[k]**2,0.)
            return (mask,last_relative),(force,diffusion)
        initial=(jnp.ones((*u.shape[1:],len(speeds)),dtype=bool),jnp.zeros(u.shape[1:],dtype=u.dtype))
        _,(force,diffusion)=jax.lax.scan(level,initial,jnp.arange(u.shape[0]-2,-1,-1))
        def average(values):
            raw=jnp.concatenate([values[::-1],jnp.zeros_like(values[:1])],axis=0)
            return jnp.concatenate([raw[:1],.5*(raw[1:]+raw[:-1])],axis=0)
        total_forcing=total_forcing+average(force)
        total_diffusion=total_diffusion+average(diffusion)
    return WaveResult(jnp.where(valid[None],total_forcing,jnp.nan),
        jnp.where(valid[None],total_diffusion,jnp.nan),dict(source_index=source,
            source_flux=flux,intermittency=intermittency,valid=valid))


def tendencies(temperature,pressure_full,height,u,v,source_index,source_amplitude,dt_seconds,*,
               gas_constant=189.02,gravity=3.72,cp=735.94,conserve_energy=False,**wave_options):
    """Original cg_drag_calc profile preparation, forcing and optional reheating.

    All supplied model winds are copied before extrapolating the top ghost.
    This gives a finite meaning to source index 1, where the original auxiliary
    wind at index 2 is uninitialized. Native driver checks use indices >=2.
    """
    t,p,z,u,v=map(jnp.asarray,(temperature,pressure_full,height,u,v))
    if t.shape!=p.shape or t.shape!=z.shape or t.shape!=u.shape or t.shape!=v.shape or len(t)<3:
        raise ValueError('gravity-wave driver requires matching profiles with at least three levels')
    lapse_top=(t[0]-t[1])/(z[0]-z[1])
    lapse_middle=(t[:-2]-t[2:])/(z[:-2]-z[2:])
    lapse=jnp.concatenate([lapse_top[None],lapse_middle,jnp.zeros_like(t[:1])],axis=0)
    bf=jnp.sqrt(jnp.maximum(gravity/t*(lapse+gravity/cp),2.5e-5))
    rho=p/(gas_constant*t)
    rho=jnp.concatenate([(rho[0]**2/rho[1])[None],rho],axis=0)
    bf=jnp.concatenate([bf[:1],bf],axis=0)
    extended_z=jnp.concatenate([(2.*z[0]-z[1])[None],z],axis=0)
    def forcing(wind):
        extended=jnp.concatenate([(2.*wind[0]-wind[1])[None],wind],axis=0)
        return wave_forcing(rho,extended,bf,extended_z,source_index,source_amplitude,**wave_options)
    zonal,meridional=forcing(u),forcing(v)
    du,dv=zonal.wind_tendency[1:],meridional.wind_tendency[1:]
    heat=-((u+.5*dt_seconds*du)*du+(v+.5*dt_seconds*dv)*dv)/cp if conserve_energy else jnp.zeros_like(t)
    index=jnp.arange(1,len(t)+1).reshape((len(t),)+(1,)*(t.ndim-1))
    diagnostic_bf=jnp.where(index<=jnp.asarray(source_index),bf[1:],0.)
    return DragResult(du,dv,heat,dict(buoyancy_frequency_s=diagnostic_bf,
        zonal_diffusivity_m2_s=zonal.eddy_diffusivity[1:],meridional_diffusivity_m2_s=meridional.eddy_diffusivity[1:],
        zonal=zonal.diagnostics,meridional=meridional.diagnostics))


def driver_step(temperature,pressure_full,height,u,v,source_index,source_amplitude,dt_seconds,
                state,*,update_seconds=0,offset_seconds=0,cp=735.94,conserve_energy=False,**options):
    """Original component timer, held forcing and current-wind reheating.

    Apply time_vary, calc and endts in that order. The released Mars caller does
    not call the timer helpers; its default zero frequency is unaffected. A
    nonzero frequency here uses the complete GFDL component sequence.
    """
    frequency,offset=schedule(update_seconds,offset_seconds)
    if state.u_tendency.shape!=temperature.shape or state.v_tendency.shape!=temperature.shape:
        raise ValueError('saved gravity-wave tendencies must match the atmospheric grid')
    changed=state.frequency_seconds!=frequency
    alarm=state.alarm_seconds+jnp.where(changed,frequency-state.frequency_seconds,0.)
    if offset>0:
        alarm=jnp.where(changed,jnp.minimum(alarm,offset),alarm)
    # cgdrag_alarm is an integer in the source, including assignment from delt.
    alarm=jnp.trunc(alarm-dt_seconds)
    refresh=alarm<=0.
    def calculate(_):
        result=tendencies(temperature,pressure_full,height,u,v,source_index,source_amplitude,
            dt_seconds,cp=cp,conserve_energy=False,**options)
        return result.u_tendency,result.v_tendency
    du,dv=jax.lax.cond(refresh,calculate,lambda _:(state.u_tendency,state.v_tendency),None)
    heat=-((u+.5*dt_seconds*du)*du+(v+.5*dt_seconds*dv)*dv)/cp if conserve_energy else jnp.zeros_like(temperature)
    next_state=DragState(du,dv,jnp.where(refresh,alarm+frequency,alarm),jnp.asarray(float(frequency)))
    return DragResult(du,dv,heat,dict(refreshed=refresh,alarm_seconds=next_state.alarm_seconds)),next_state
