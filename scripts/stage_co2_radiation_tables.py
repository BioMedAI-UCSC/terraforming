#!/usr/bin/env python3
"""Decode the bundled Ames 12-band Fortran tables into a compact JAX asset.

The source records and dimensions follow ``set_bands``,
``setrad``, ``laginterp``, ``optcv``, and ``optci`` in AmesGCM's
``atmos_param_mars/Rad/rtmod_mgcm.F90``.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import struct

import numpy as np


T_REF_K = np.arange(50.0, 351.0, 50.0)
P_REF_MBAR = 10.0 ** np.arange(-6.0, 5.0)
P_FINE_LOG_MBAR = np.round(np.linspace(-6.,4.,51),1)
WATER_MASS_FRACTIONS = np.array([1e-7,1e-6,1e-5,1e-4,1e-3,.01,.1,.2,.3,.4])
GAUSS_WEIGHTS = np.array([
    4.8083554740e-2, 1.0563099137e-1, 1.4901065679e-1,
    1.7227479710e-1, 1.7227479710e-1, 1.4901065679e-1,
    1.0563099137e-1, 4.8083554740e-2, 2.5307134073e-3,
    5.5595258613e-3, 7.8426661469e-3, 9.0670945845e-3,
    9.0670945845e-3, 7.8426661469e-3, 5.5595258613e-3,
    2.5307134073e-3,
])
_SOLAR_BAND_FLUX = np.array([12.7, 24.2, 54.6, 145.9, 354.9, 657.5, 106.3])
SOLAR_WEIGHTS = _SOLAR_BAND_FLUX / _SOLAR_BAND_FLUX.sum()
IR_BOUNDS_CM1 = np.array([10.0, 166.667, 416.667, 833.333, 1250.0, 2222.222])


def _record(handle) -> bytes:
    marker = handle.read(4)
    if len(marker) != 4:
        raise ValueError("missing Fortran record marker")
    size = struct.unpack("<I", marker)[0]
    payload = handle.read(size)
    trailer = handle.read(4)
    if len(payload) != size or trailer != marker:
        raise ValueError("invalid little-endian sequential Fortran record")
    return payload


def decode(path: Path, bands: int) -> tuple[np.ndarray, np.ndarray]:
    shape = (7, 11, 10, bands, 17)
    with path.open("rb") as handle:
        coefficients = np.frombuffer(_record(handle), dtype="<f8")
        clear_fraction = np.frombuffer(_record(handle), dtype="<f8")
        if handle.read(1):
            raise ValueError(f"unexpected trailing bytes in {path}")
    if coefficients.size != np.prod(shape) or clear_fraction.size != bands:
        raise ValueError(f"unexpected Ames table dimensions in {path}")
    # Fortran's first index is contiguous. Omit the 17th clear channel,
    # whose gas coefficient is defined as zero.
    table = coefficients.reshape(shape, order="F")[..., :16]
    return np.log10(np.maximum(table, 1.0e-200)), clear_fraction.copy()


def ames_pressure_interpolation(log_table):
    """Ames laginterp: four-point cubic log(k), then the 51-point RT grid."""
    log_table = np.asarray(log_table)
    coarse = np.log10(P_REF_MBAR)
    rows = []
    for p in P_FINE_LOG_MBAR[:-1]:
        interval = min(max(int(np.floor(p-coarse[0])),0),len(coarse)-2)
        first = min(max(interval-1,0),len(coarse)-4)
        x = coarse[first:first+4]
        y = log_table[:,first:first+4]
        value = np.zeros_like(y[:,0])
        for k in range(4):
            others = [i for i in range(4) if i!=k]
            numerator = (p-x[others[0]])*(p-x[others[1]])*(p-x[others[2]])
            denominator = (x[k]-x[others[0]])*(x[k]-x[others[1]])*(x[k]-x[others[2]])
            value += numerator*y[:,k]/denominator
        # Preserve laginterp's conversion back to k and then to log(k).
        rows.append(np.log10(10.**value))
    rows.append(np.log10(10.**log_table[:,-1]))
    return np.stack(rows,axis=1)


def _planck_fractions() -> tuple[np.ndarray, np.ndarray]:
    """Planck flux fractions for the five Ames IR intervals (50--400 K)."""
    temperatures = np.arange(50.0, 401.0, 1.0)
    nodes, weights = np.polynomial.legendre.leggauss(64)
    h, c, kb = 6.62607015e-34, 299792458.0, 1.380649e-23
    values = np.empty((temperatures.size, 5), dtype=np.float64)
    for band, (lo, hi) in enumerate(zip(IR_BOUNDS_CM1[:-1], IR_BOUNDS_CM1[1:])):
        lo_m, hi_m = lo * 100.0, hi * 100.0
        wn = 0.5 * (hi_m - lo_m) * nodes + 0.5 * (hi_m + lo_m)
        for ti, temperature in enumerate(temperatures):
            spectral = 2.0 * h * c**2 * wn**3 / np.expm1(h * c * wn / (kb * temperature))
            values[ti, band] = 0.5 * (hi_m - lo_m) * np.sum(weights * spectral)
    values /= values.sum(axis=1, keepdims=True)
    return temperatures, values


def ames_planck_intensities():
    """Absolute Planck band means from Ames setspi's 12-point quadrature."""
    temperatures=np.arange(500,9001,dtype=np.float64)/10.
    nodes=np.array([-.981560634246719,-.904117256370475,-.769902674194305,
        -.587317954286617,-.367831498998180,-.125233408511469,
        .125233408511469,.367831498998180,.587317954286617,
        .769902674194305,.904117256370475,.981560634246719])
    weights=np.array([.047175336386512,.106939325995318,.160078328543346,
        .203167426723066,.233492536538355,.249147045813403,
        .249147045813403,.233492536538355,.203167426723066,
        .160078328543346,.106939325995318,.047175336386512])
    values=np.empty((len(temperatures),5))
    for band,(lo,hi) in enumerate(zip(IR_BOUNDS_CM1[:-1],IR_BOUNDS_CM1[1:],strict=True)):
        a,b=1.e-2/hi,1.e-2/lo
        bpa,bma=(b+a)/2.,(b-a)/2.
        ans=np.zeros_like(temperatures)
        for node,weight in zip(nodes,weights,strict=True):
            y=bma*node+bpa
            ans+=weight*3.741832e-16/(y**5*(np.exp(1.438786e-2/(y*temperatures))-1.))
        values[:,band]=ans*bma/(np.pi*(hi-lo))
    return temperatures,values


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ames-data", type=Path, default=root / "AmesGCM/data")
    parser.add_argument(
        "--output", type=Path,
        default=root / "package/src/framework/physics/co2_12band_tables.npz",
    )
    args = parser.parse_args()
    ir_path = args.ames_data / "CO2H2O_IR_12_95_INTEL"
    sw_path = args.ames_data / "CO2H2O_V_12_95_INTEL"
    ir, ir_clear = decode(ir_path, 5)
    sw, sw_clear = decode(sw_path, 7)
    planck_t, planck_fraction = _planck_fractions()
    ames_planck_t,ames_planck=ames_planck_intensities()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        log10_k_ir=ir[:,:,0],
        log10_k_sw=sw[:,:,0],
        log10_k_ir_ames=ames_pressure_interpolation(ir)[:,:,0],
        log10_k_sw_ames=ames_pressure_interpolation(sw)[:,:,0],
        pressure_mbar_ames=10.**P_FINE_LOG_MBAR,
        water_mass_fractions=WATER_MASS_FRACTIONS,
        log10_k_ir_mixtures_ames=ames_pressure_interpolation(ir),
        log10_k_sw_mixtures_ames=ames_pressure_interpolation(sw),
        clear_fraction_ir=ir_clear,
        clear_fraction_sw=sw_clear,
        gauss_weights=GAUSS_WEIGHTS,
        solar_weights=SOLAR_WEIGHTS,
        temperature_k=T_REF_K,
        pressure_mbar=P_REF_MBAR,
        planck_temperature_k=planck_t,
        planck_fraction_ir=planck_fraction,
        ames_planck_temperature_k=ames_planck_t,
        ames_planck_intensity_ir=ames_planck,
        ir_bounds_cm1=IR_BOUNDS_CM1,
        source_ir_sha256=hashlib.sha256(ir_path.read_bytes()).hexdigest(),
        source_sw_sha256=hashlib.sha256(sw_path.read_bytes()).hexdigest(),
        source="NASA Ames Mars GCM v3.2.1 12-band CO2/H2O tables; full mixtures and laginterp pressure grid",
    )
    print(f"staged {args.output}")
    print(f"IR log10(k): {ir.min():.3f} .. {ir.max():.3f}")
    print(f"SW log10(k): {sw.min():.3f} .. {sw.max():.3f}")


if __name__ == "__main__":
    main()
