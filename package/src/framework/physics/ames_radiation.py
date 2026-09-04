"""Differentiable access to the staged NASA Ames 12-band CO2 k tables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.framework.gcm._dinosaur import jnp

CO2_COLUMN_FACTOR_PER_MBAR = 3.51e22
_ASSET = Path(__file__).with_name("ames_co2_12band.npz")

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
            f"missing {_ASSET}; run scripts/stage_ames_co2_radiation.py"
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
        if np.issubdtype(value.dtype, np.number)
    }


def _interp2_log_table(table, temperature_axis, pressure_axis, temperature_k,
                       pressure_mbar):
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
    v00, v10 = values[it, ip], values[it + 1, ip]
    v01, v11 = values[it, ip + 1], values[it + 1, ip + 1]
    return ((1.0 - ft)[..., None, None] *
            ((1.0 - fp)[..., None, None] * v00 + fp[..., None, None] * v01)
            + ft[..., None, None] *
            ((1.0 - fp)[..., None, None] * v10 + fp[..., None, None] * v11))


def correlated_k_optical_depths(temperature_k, pressure_mid_pa, delta_pressure_pa):
    """Return SW/LW gas optical depths as ``(band, g, layer, lon, lat)``."""
    data = load_ames_co2_tables_jax()
    pressure_mbar = pressure_mid_pa / 100.0
    delta_mbar = delta_pressure_pa / 100.0
    common = (data["temperature_k"], data["pressure_mbar"], temperature_k,
              pressure_mbar)
    log_sw = _interp2_log_table(data["log10_k_sw"], *common)
    log_ir = _interp2_log_table(data["log10_k_ir"], *common)
    scale = CO2_COLUMN_FACTOR_PER_MBAR * delta_mbar[..., None, None]
    tau_sw = scale * 10.0 ** log_sw
    tau_ir = scale * 10.0 ** log_ir
    # The last channel is the Ames clear fraction and has zero gas opacity.
    tau_sw = jnp.concatenate([tau_sw, jnp.zeros_like(tau_sw[..., :1])], axis=-1)
    tau_ir = jnp.concatenate([tau_ir, jnp.zeros_like(tau_ir[..., :1])], axis=-1)
    return jnp.moveaxis(tau_sw, (-2, -1), (0, 1)), jnp.moveaxis(
        tau_ir, (-2, -1), (0, 1)
    )


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
