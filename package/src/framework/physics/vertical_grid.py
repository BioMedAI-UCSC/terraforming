"""Released hybrid pressure coefficients and their Dinosaur sigma adaptation.

NASA-derived data: Copyright 2023 United States Government as represented by
NASA. Apache License 2.0; see licenses/VERTICAL-GRID-NOTICE.txt.
"""
from functools import lru_cache
from pathlib import Path
import numpy as np
from src.framework.gcm._dinosaur import jnp


@lru_cache(maxsize=1)
def load_grids():
    with np.load(Path(__file__).with_name('vertical_grid_coefficients.npz'),allow_pickle=False) as opened:
        return {name:opened[name].copy() for name in opened.files}


def supported_layers():
    return tuple(sorted(int(name[1:]) for name in load_grids() if name.startswith('a')))


def coefficients(layers,reference_pressure_pa):
    """Original set_eta scales the released A coefficients by p_ref/725."""
    if not isinstance(layers,int) or isinstance(layers,bool) or layers not in supported_layers():
        raise ValueError(f'released vertical-grid layers must be one of {supported_layers()}')
    data=load_grids();reference=jnp.asarray(reference_pressure_pa)
    if reference.ndim!=0: raise ValueError('reference pressure must be scalar')
    reference=jnp.where(jnp.isfinite(reference)&(reference>0.),reference,jnp.nan)
    return jnp.asarray(data[f'a{layers}'])*reference/725.,jnp.asarray(data[f'b{layers}'])


def pressure_interfaces(layers,reference_pressure_pa,surface_pressure_pa):
    """Original hybrid interfaces in Pa, with levels first and arbitrary grid axes."""
    a,b=coefficients(layers,reference_pressure_pa)
    surface=jnp.asarray(surface_pressure_pa)
    shape=(layers+1,)+(1,)*surface.ndim
    pressure=a.reshape(shape)+b.reshape(shape)*surface
    return jnp.where(jnp.isfinite(surface)&(surface>0.),pressure,jnp.nan)


def sigma_boundaries(layers):
    """Reference-column hybrid grid at p_s=p_ref, with Dinosaur's zero top.

    A sigma core cannot retain fixed-pressure hybrid interfaces over varying
    terrain. Freeze their reference-column pressure fractions, then set only
    the first boundary to zero. Arithmetic full-level midpoints are Dinosaur's
    discretization, not FV3's logarithmic pressure means.
    """
    if not isinstance(layers,int) or isinstance(layers,bool) or layers not in supported_layers():
        raise ValueError(f'released vertical-grid layers must be one of {supported_layers()}')
    data=load_grids()
    boundaries=data[f'a{layers}']/725.+data[f'b{layers}']
    boundaries=boundaries.copy();boundaries[0]=0.
    if not np.all(np.isfinite(boundaries)) or not np.all(np.diff(boundaries)>0.) or boundaries[-1]!=1.:
        raise ValueError('released vertical grid cannot define increasing sigma boundaries')
    return boundaries


def initialization_reference_interfaces(sigma,reference_pressure_pa):
    """Static source selectors retain the released positive top interface."""
    sigma=np.asarray(sigma,dtype=float)
    reference=float(reference_pressure_pa)
    if sigma.ndim!=1 or len(sigma)<2 or not np.isfinite(sigma).all() or \
            sigma[0]<0. or sigma[-1]!=1. or np.any(np.diff(sigma)<=0.):
        raise ValueError('initialization sigma interfaces must increase to one')
    if not np.isfinite(reference) or reference<=0.:
        raise ValueError('initialization reference pressure must be finite and positive')
    layers=len(sigma)-1
    if layers in supported_layers() and np.array_equal(sigma,sigma_boundaries(layers)):
        data=load_grids()
        return data[f'a{layers}']*reference/725.+data[f'b{layers}']*reference
    return sigma*reference
