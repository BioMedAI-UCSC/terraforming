"""Differentiable access to the staged NASA Ames 12-band CO2 k tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.framework.gcm._dinosaur import jnp

CO2_COLUMN_FACTOR_PER_MBAR = 3.51e22
_ASSET = Path(__file__).with_name("co2_12band_tables.npz")

# Ames fixed-dust optical properties for Reff=1.5 um, Veff=0.5, ordered on the
# same seven solar and five infrared intervals as the gas tables.
DUST_SW_EXTINCTION = jnp.array([1.834, 2.296, 2.672, 2.829, 2.698, 2.452, 2.261])
DUST_SW_SCATTERING = jnp.array([1.695, 2.031, 2.583, 2.744, 2.626, 2.225, 1.525])
DUST_SW_ASYMMETRY = jnp.array([0.551, 0.640, 0.661, 0.678, 0.690, 0.743, 0.868])
DUST_IR_EXTINCTION = jnp.array([0.008, 0.262, 0.491, 1.017, 0.444])
DUST_IR_SCATTERING = jnp.array([0.001, 0.037, 0.122, 0.351, 0.336])
DUST_IR_ASYMMETRY = jnp.array([0.004, 0.030, 0.095, 0.214, 0.316])


@lru_cache(maxsize=1)
def load_ames_co2_tables() -> dict[str, np.ndarray]:
    """Load the compact, reproducibly staged dry-CO2 table asset once."""
    if not _ASSET.exists():
        raise FileNotFoundError(
            f"missing {_ASSET}; run scripts/stage_co2_radiation_tables.py"
        )
    with np.load(_ASSET, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def load_ames_co2_tables_jax() -> dict[str, object]:
    """Trace-safe numeric table fields as device-compatible JAX arrays.

    The underlying NumPy archive is cached. Do not cache this returned mapping:
    the first call can occur inside ``lax.scan`` tracing, in which case caching
    would retain escaped tracers and break the next integration chunk.
    """
    return {
        name: jnp.asarray(value)
        for name, value in load_ames_co2_tables().items()
        if np.issubdtype(value.dtype, np.number) and "mixtures" not in name
    }


@lru_cache(maxsize=3)
def load_radiation_tables(mode='12band'):
    """Numeric reference tables for the three released radiation configurations."""
    if mode=='12band': return load_ames_co2_tables()
    if mode not in ('15band','extended'):
        raise ValueError('radiation mode must be 12band, 15band or extended')
    with np.load(_ASSET.with_name(f'co2_{mode}_tables.npz'),allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def radiation_tables_jax(mode='12band'):
    """Convert cached host tables while avoiding escaped tracers across rollouts."""
    return {name:jnp.asarray(value) for name,value in load_radiation_tables(mode).items()
        if np.issubdtype(value.dtype,np.number) and 'mixtures' not in name}


def _interp2_log_table(table, temperature_axis, pressure_axis, temperature_k,
                       pressure_mbar,*,mixture_index=None):
    """Bilinear interpolation of log10(k) on T and log10-pressure axes."""
    t_axis = jnp.asarray(temperature_axis)
    p_axis = jnp.log10(jnp.asarray(pressure_axis))
    t = jnp.clip(temperature_k, t_axis[0], t_axis[-1])
    p = jnp.clip(jnp.log10(jnp.clip(pressure_mbar, 1.0e-12, None)),
                 p_axis[0], p_axis[-1])
    it = jnp.clip(jnp.searchsorted(t_axis, t, side="right") - 1, 0, t_axis.size - 2)
    ip = jnp.clip(jnp.searchsorted(p_axis, p, side="right") - 1, 0, p_axis.size - 2)
    ft = (t - t_axis[it]) / (t_axis[it + 1] - t_axis[it])
    fp = (p - p_axis[ip]) / (p_axis[ip + 1] - p_axis[ip])
    values = jnp.asarray(table)
    def lookup(ti,pi):
        return values[ti,pi] if mixture_index is None else values[ti,pi,mixture_index]
    v00, v10 = lookup(it,ip),lookup(it+1,ip)
    v01, v11 = lookup(it,ip+1),lookup(it+1,ip+1)
    return ((1.0 - ft)[..., None, None] *
            ((1.0 - fp)[..., None, None] * v00 + fp[..., None, None] * v01)
            + ft[..., None, None] *
            ((1.0 - fp)[..., None, None] * v10 + fp[..., None, None] * v11))


def correlated_k_optical_depths(temperature_k, pressure_mid_pa, delta_pressure_pa,*,
                               mode='12band',water_mass_fraction=None):
    """Return SW/LW gas optical depths as ``(band, g, layer, lon, lat)``."""
    data = radiation_tables_jax(mode)
    pressure_mbar = pressure_mid_pa / 100.0
    delta_mbar = delta_pressure_pa / 100.0
    common = (data["temperature_k"], data["pressure_mbar_ames"], temperature_k,
              pressure_mbar)
    if mode!='12band':
        tables=load_radiation_tables(mode)
        water=0. if water_mass_fraction is None else water_mass_fraction
        mixture=jnp.clip(jnp.searchsorted(data['water_mass_fractions'],jnp.asarray(water),side='right')-1,0,9)
        log_sw=_box_lookup(tables['log10_k_sw_mixtures_ames'],*common,mixture)
        log_ir=_box_lookup(tables['log10_k_ir_mixtures_ames'],*common,mixture)
    elif water_mass_fraction is None:
        log_sw = _interp2_log_table(data["log10_k_sw_ames"], *common)
        log_ir = _interp2_log_table(data["log10_k_ir_ames"], *common)
    else:
        # tpindex15 selects a mixture bin; it does not interpolate water amount.
        mixture=jnp.clip(jnp.searchsorted(data['water_mass_fractions'],
            jnp.asarray(water_mass_fraction),side='right')-1,0,9)
        tables=load_radiation_tables(mode)
        log_sw=_interp2_log_table(tables['log10_k_sw_mixtures_ames'],*common,mixture_index=mixture)
        log_ir=_interp2_log_table(tables['log10_k_ir_mixtures_ames'],*common,mixture_index=mixture)
    scale = CO2_COLUMN_FACTOR_PER_MBAR * delta_mbar[..., None, None]
    tau_sw = scale * 10.0 ** log_sw
    tau_ir = scale * 10.0 ** log_ir
    # The last channel is the Ames clear fraction and has zero gas opacity.
    tau_sw = jnp.concatenate([tau_sw, jnp.zeros_like(tau_sw[..., :1])], axis=-1)
    tau_ir = jnp.concatenate([tau_ir, jnp.zeros_like(tau_ir[..., :1])], axis=-1)
    return jnp.moveaxis(tau_sw, (-2, -1), (0, 1)), jnp.moveaxis(
        tau_ir, (-2, -1), (0, 1)
    )


def _box_lookup(table,taxis,paxis,temperature,pressure,mixture):
    """tpindex15/boxinterp branch arithmetic, including nonfinite source entries."""
    temperature,pressure=map(jnp.asarray,(temperature,pressure))
    logp=jnp.log10(jnp.maximum(pressure,1e-12));logaxis=jnp.log10(paxis)
    ti=(temperature>taxis[0])&(temperature<taxis[-1])
    pi=(logp>logaxis[0])&(logp<logaxis[-1])
    it=jnp.clip(jnp.searchsorted(taxis,temperature,side='right')-1,0,len(taxis)-1)
    ip=jnp.clip(jnp.searchsorted(logaxis,logp,side='right')-1,0,len(logaxis)-1)
    jt=jnp.minimum(it+ti.astype(int),len(taxis)-1)
    jp=jnp.minimum(ip+pi.astype(int),len(logaxis)-1)
    ft=(temperature-taxis[it])/jnp.where(ti,taxis[jt]-taxis[it],1.)
    fp=(logp-logaxis[ip])/jnp.where(pi,logaxis[jp]-logaxis[ip],1.)
    values=jnp.asarray(table)
    a,b=values[it,ip,mixture],values[jt,ip,mixture]
    c,d=values[it,jp,mixture],values[jt,jp,mixture]
    row0=a+jnp.where(ti[...,None,None],(b-a)*ft[...,None,None],0.)
    row1=c+jnp.where(ti[...,None,None],(d-c)*ft[...,None,None],0.)
    return row0+jnp.where(pi[...,None,None],(row1-row0)*fp[...,None,None],0.)


def channel_weights(clear_fraction):
    """Ames split-Gaussian weights plus the clear-spectrum channel."""
    data = load_ames_co2_tables_jax()
    clear = jnp.asarray(clear_fraction)
    absorbing = ((1.0 - clear)[:, None]
                 * data["gauss_weights"][None, :])
    return jnp.concatenate([absorbing, clear[:, None]], axis=1)


def planck_band_fractions(temperature_k):
    """Linearly interpolate normalized five-band Planck flux fractions."""
    data = load_ames_co2_tables_jax()
    axis = data["planck_temperature_k"]
    values = data["planck_fraction_ir"]
    t = jnp.clip(temperature_k, axis[0], axis[-1])
    index = jnp.clip(jnp.searchsorted(axis, t, side="right") - 1, 0, axis.size - 2)
    fraction = (t - axis[index]) / (axis[index + 1] - axis[index])
    result = values[index] + fraction[..., None] * (values[index + 1] - values[index])
    return jnp.moveaxis(result, -1, 0)


def planck_band_intensities(temperature_k,*,mode='12band'):
    """Ames absolute W/m2/sr/(cm^-1) means, continuously interpolated in T.

    At the 0.1 K reference nodes this uses setspi's unnormalized integrals.
    Continuous interpolation replaces the original integer temperature index
    so gradients reach temperature. Native validation covers 50--900 K.
    """
    data=radiation_tables_jax(mode)
    axis=data['ames_planck_temperature_k']
    values=data['ames_planck_intensity_ir']
    temperature=jnp.asarray(temperature_k)
    index=jnp.clip(jnp.searchsorted(axis,temperature,side='right')-1,0,axis.size-2)
    fraction=(temperature-axis[index])/(axis[index+1]-axis[index])
    intensity=values[index]+fraction[...,None]*(values[index+1]-values[index])
    return jnp.moveaxis(intensity,-1,0)
