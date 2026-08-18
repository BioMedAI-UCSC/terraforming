#!/usr/bin/env python3
"""Decode the bundled Ames 12-band Fortran tables into a compact JAX asset.

Only the nearly pure-CO2 mixture (H2O mass fraction 1e-7) is staged for the
present dry Mars GCM. The source records and dimensions follow ``set_bands``,
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
    # Fortran's first index is contiguous. Select the driest mixture and omit
    # the 17th clear channel, whose gas coefficient is defined as zero.
    table = coefficients.reshape(shape, order="F")[:, :, 0, :, :16]
    return np.log10(np.maximum(table, 1.0e-200)), clear_fraction.copy()


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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--ames-data", type=Path, default=root / "AmesGCM/data")
    parser.add_argument(
        "--output", type=Path,
        default=root / "package/src/gcm3d/ames_co2_12band.npz",
    )
    args = parser.parse_args()
    ir_path = args.ames_data / "CO2H2O_IR_12_95_INTEL"
    sw_path = args.ames_data / "CO2H2O_V_12_95_INTEL"
    ir, ir_clear = decode(ir_path, 5)
    sw, sw_clear = decode(sw_path, 7)
    planck_t, planck_fraction = _planck_fractions()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        log10_k_ir=ir,
        log10_k_sw=sw,
        clear_fraction_ir=ir_clear,
        clear_fraction_sw=sw_clear,
        gauss_weights=GAUSS_WEIGHTS,
        solar_weights=SOLAR_WEIGHTS,
        temperature_k=T_REF_K,
        pressure_mbar=P_REF_MBAR,
        planck_temperature_k=planck_t,
        planck_fraction_ir=planck_fraction,
        ir_bounds_cm1=IR_BOUNDS_CM1,
        source_ir_sha256=hashlib.sha256(ir_path.read_bytes()).hexdigest(),
        source_sw_sha256=hashlib.sha256(sw_path.read_bytes()).hexdigest(),
        source="NASA Ames Mars GCM v3.2.1 12-band CO2/H2O tables; dry-mixture slice",
    )
    print(f"staged {args.output}")
    print(f"IR log10(k): {ir.min():.3f} .. {ir.max():.3f}")
    print(f"SW log10(k): {sw.min():.3f} .. {sw.max():.3f}")


if __name__ == "__main__":
    main()
