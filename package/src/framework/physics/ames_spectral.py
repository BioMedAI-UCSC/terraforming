"""Ames sfluxv/sfluxi aggregation on supplied radiation-grid optics.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
Arrays use (radiation layer, spectral band, Gaussian channel, ...).
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.ames_radiation import load_ames_co2_tables_jax,planck_band_intensities
from src.framework.physics.ames_twostream import visible_fluxes,infrared_fluxes


class SpectralFlux(NamedTuple):
    up: object
    down: object
    net_top: object


def _channel_selection(gas_column,clear,gaussian_weights,threshold):
    gas=jnp.asarray(gas_column)
    clear=jnp.reshape(clear,(len(clear),)+(1,)*(gas.ndim-1))
    active=(clear<.99)&(gas>=threshold)
    weights=(1.-clear)*jnp.reshape(gaussian_weights,(1,len(gaussian_weights))+(1,)*(gas.ndim-2))
    resolved=jnp.where(active,weights,0.)
    merged=(clear<.99)&(gas<threshold)
    continuum=clear[:,0]+jnp.sum(jnp.where(merged,weights,0.),axis=1)
    return active,jnp.concatenate([resolved,continuum[:,None]],axis=1)


def _mask_unused(values,active):
    # Evaluate unused channels with clear-channel inputs to keep reverse mode finite.
    return jnp.concatenate([jnp.where(active[None],values[:,:,:-1],values[:,:,-1:]),
        values[:,:,-1:]],axis=2)


def visible_spectrum(dtau,scattering_albedo,asymmetry,cumulative,gas_column,
                     mu0,solar_intensity,reflectivity):
    """Original sfluxv channel merging and surface reflection, for mu0 > 0.

    gas_column excludes the clear channel and contains gas opacity only.
    Night is handled by the outer radiation driver, as in Ames.
    """
    data=load_ames_co2_tables_jax()
    active,weights=_channel_selection(gas_column,data['clear_fraction_sw'],data['gauss_weights'],1.e-3)
    dtau,w,g,cumulative=(_mask_unused(jnp.asarray(x),active)
        for x in (dtau,scattering_albedo,asymmetry,cumulative))
    solar=jnp.asarray(solar_intensity)
    solar=solar.reshape((solar.shape[0],1)+(1,)*(dtau.ndim-3))
    delta=jnp.sum(dtau*(1.-w*g*g),axis=0)
    surface=reflectivity*mu0*solar*jnp.exp(-jnp.minimum(delta/mu0,35.))
    up,down,topup,topdown,_,_=visible_fluxes(dtau,w,g,cumulative,mu0,solar,reflectivity,0.,surface)
    return SpectralFlux(jnp.sum(up*weights[None],axis=(1,2)),
        jnp.sum(down*weights[None],axis=(1,2)),jnp.sum((topup-topdown)*weights,axis=(0,1)))


def infrared_spectrum(dtau,scattering_albedo,asymmetry,cumulative,gas_column,
                      pressure_levels,temperature_levels,reflectivity,*,scale_surface_flux=False):
    """Original sfluxi channel merging, top emission and spectral integration.

    Continuous Planck interpolation is the explicit adaptation for temperature
    gradients. The default leaves surface emission unscaled, as in Ames.
    """
    data=load_ames_co2_tables_jax()
    active,weights=_channel_selection(gas_column,data['clear_fraction_ir'],data['gauss_weights'],5.e-3)
    dtau,w,g,cumulative=(_mask_unused(jnp.asarray(x),active)
        for x in (dtau,scattering_albedo,asymmetry,cumulative))
    pressure,temperature=map(jnp.asarray,(pressure_levels,temperature_levels))
    if pressure.shape!=temperature.shape or pressure.shape[0]!=2*dtau.shape[0]+1:
        raise ValueError("infrared spectral transfer requires the Ames radiation subgrid")
    boundary_t=jnp.concatenate([temperature[1::2],temperature[-1:]],axis=0)
    planck=jnp.moveaxis(planck_band_intensities(boundary_t),0,1)[:,:,None]
    planck=jnp.broadcast_to(planck,(dtau.shape[0]+1,)+dtau.shape[1:])
    width=jnp.diff(data['ir_bounds_cm1']).reshape((5,)+(1,)*(dtau.ndim-2))
    surface=(1.-reflectivity)*planck[-1]
    if scale_surface_flux:
        integral=3.14159*jnp.sum(width[:,0]*surface[:,0],axis=0)
        surface=surface*(5.67e-8*temperature[-1]**4/integral)
    top_tau=dtau[0]*pressure[1]/(pressure[3]-pressure[1])
    top=(1.-jnp.exp(-top_tau/.5))*planck[0]
    offset=cumulative[2::2]-cumulative[1::2]
    up,down,topnet=infrared_fluxes(dtau,w,g,planck,offset,reflectivity,top,surface)
    weighted=weights*width
    return SpectralFlux(jnp.sum(up*weighted[None],axis=(1,2)),
        jnp.sum(down*weighted[None],axis=(1,2)),jnp.sum(topnet*weighted,axis=(0,1)))
