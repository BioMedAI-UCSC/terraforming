"""Hash-checked primary optical tables. I/O is outside every JAX kernel."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ASSETS=Path(__file__).with_name("assets")


@dataclass(frozen=True)
class SpectralTable:
    wavelength_m: np.ndarray
    absorption_m2: np.ndarray
    scattering_m2: np.ndarray
    asymmetry: np.ndarray
    provenance: dict

    def __post_init__(self):
        w,a,s,g=self.wavelength_m,self.absorption_m2,self.scattering_m2,self.asymmetry
        if (w.ndim!=1 or len(w)<2 or any(v.shape!=w.shape for v in (a,s,g))
                or not all(np.isfinite(v).all() for v in (w,a,s,g))
                or np.any(np.diff(w)<=0) or np.any(w<=0) or np.any(a<0) or np.any(s<0)
                or np.any(np.abs(g)>1)):
            raise ValueError("invalid spectral units, shapes, ordering or physical ranges")
        if not self.provenance.get("source") or not self.provenance.get("license"):
            raise ValueError("optics source and reuse terms are required")


def rod_table(particle):
    if particle.optics_key not in ("al","fe") or not np.allclose(
        [particle.length_m,particle.width_m,particle.height_m],[9e-6,.16e-6,.16e-6],rtol=0,atol=1e-15
    ):
        raise ValueError("no source-backed optical table for this geometry/material; supply an explicit table")
    manifest=json.loads((ASSETS/'manifest.json').read_text())
    path=ASSETS/'ansari2024_rods.csv'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['data_sha256']:
        raise ValueError("primary optical table checksum mismatch")
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    key=particle.optics_key
    return SpectralTable(np.array([float(r['wavelength_m']) for r in rows]),
        np.array([float(r[key+'_absorption_m2']) for r in rows]),
        np.array([float(r[key+'_scattering_m2']) for r in rows]),
        np.array([float(r[key+'_asymmetry']) for r in rows]),
        dict(source=manifest['source_doi']+' '+manifest['source_table'],license=manifest['license'],
             asset_sha256=manifest['data_sha256'],source_pdf_sha256=manifest['source_pdf_sha256'],
             classification='simulated monomer spectra; not aggregate or graphene optics'))


def load_external_table(path,*,expected_sha256,source,license):
    """External materials must provide actual spectra, not a dust fallback."""
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_sha256:
        raise ValueError("external optics checksum mismatch")
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    columns=['wavelength_m','absorption_m2','scattering_m2','asymmetry']
    arrays=[np.array([float(r[k]) for r in rows]) for k in columns]
    return SpectralTable(*arrays,dict(source=source,license=license,asset_sha256=expected_sha256,
                                     classification='user-supplied; validate geometry, orientation and spectral domain'))
