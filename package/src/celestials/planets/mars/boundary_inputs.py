"""Pinned polar-water and terrain-variance inputs for reference physics."""
import hashlib
from pathlib import Path
import numpy as np
from src.celestials.planets.mars.surface import conservative_surface_field


INPUT_BLOBS={
    'npcflag8.nc':'750894161ea187f4934301409974696c94aa3be3',
    'palmer_drag_input_c24_c48_c96.nc':'0f77f437e7ce0742952f2b6dc5057299d4d4e474',
}


def _read_field(grid,directory,filename,variable):
    import xarray as xr
    path=Path(directory)/filename
    raw=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=INPUT_BLOBS[filename]:
        raise ValueError(f'{filename} differs from the pinned reference input')
    with xr.open_dataset(path) as source:
        if variable not in source or set(source[variable].dims)!={'lat','lon'}:
            raise ValueError(f'{filename} lacks a latitude/longitude {variable} field')
        return conservative_surface_field(grid,source[variable],
            source['lonb'] if 'lonb' in source else None,
            source['latb'] if 'latb' in source else None)


def polar_water_on_grid(grid,data_directory,initial_frost_kg_m2=500.):
    """Original cold-start polar cap: remap the flag, then apply strict >0.5."""
    if not np.isfinite(initial_frost_kg_m2) or initial_frost_kg_m2<0.:
        raise ValueError('initial surface frost must be finite and nonnegative')
    flag=_read_field(grid,data_directory,'npcflag8.nc','npcflag')
    if np.any((flag < -1e-12)|(flag > 1.+1e-12)):
        raise ValueError('polar water flag must lie in [0,1]')
    return np.where(flag>.5,initial_frost_kg_m2,0.)


def terrain_standard_deviation_on_grid(grid,data_directory,grid_resolution=24):
    """Original nearest SD24/SD48/SD96 map; lower resolution wins a tie.

    Preserve signed map values: the original drag squares them after remapping.
    Squaring or clipping the input before remapping changes the source physics.
    """
    if not isinstance(grid_resolution,int) or isinstance(grid_resolution,bool) or grid_resolution<1:
        raise ValueError('terrain map resolution must be a positive integer')
    available=np.array([24,48,96])
    selected=int(available[np.argmin(np.abs(available-grid_resolution))])
    return _read_field(grid,data_directory,'palmer_drag_input_c24_c48_c96.nc',f'SD{selected}')
