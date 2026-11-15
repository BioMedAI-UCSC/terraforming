"""Spatial Mars surface boundary fields used by deterministic column physics."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def interpolate_surface_fields(grid,ds):
    """Sample latitude/longitude surface fields on the Dinosaur nodal grid."""
    import xarray as xr

    ds=ds.sortby('lat').sortby('lon')
    ds=ds.interpolate_na(dim='lon',method='nearest',fill_value='extrapolate')
    ds=ds.interpolate_na(dim='lat',method='nearest',fill_value='extrapolate')
    ds=ds.assign_coords(lon=np.mod(ds.lon,360.)).sortby('lon')
    _,unique=np.unique(np.asarray(ds.lon),return_index=True)
    ds=ds.isel(lon=np.sort(unique))
    left=ds.isel(lon=-1).assign_coords(lon=float(ds.lon[-1])-360.)
    right=ds.isel(lon=0).assign_coords(lon=float(ds.lon[0])+360.)
    ds=xr.concat([left,ds,right],dim='lon')
    result=ds.interp(lon=np.degrees(np.asarray(grid.longitudes))%360.,
        lat=np.degrees(np.asarray(grid.latitudes)))
    return {name:np.asarray(value.transpose('lon','lat')) for name,value in result.data_vars.items()}


def _overlap_weights(source_bounds,target_bounds,period=None):
    from scipy.sparse import csr_matrix

    source=np.asarray(source_bounds,dtype=float)
    target=np.asarray(target_bounds,dtype=float)
    if np.any(np.diff(source)<=0.) or np.any(np.diff(target)<=0.):
        raise ValueError('surface cell boundaries must increase')
    overlap=np.zeros((target.size-1,source.size-1))
    for shift in ((-period,0.,period) if period is not None else (0.,)):
        overlap+=np.maximum(0.,np.minimum(target[1:,None],source[None,1:]+shift)-
            np.maximum(target[:-1,None],source[None,:-1]+shift))
    weights=overlap/np.diff(target)[:,None]
    if not np.allclose(weights.sum(axis=1),1.,rtol=1e-12,atol=1e-12):
        raise ValueError('surface source cells do not cover the target grid')
    return csr_matrix(weights)


def conservative_surface_field(grid,field,longitude_bounds=None,latitude_bounds=None):
    """Reference conservative lat/lon remap adapted to Gaussian cell areas.

    As in read_sfc_field, absent source bounds mean uniform global cells.
    Target latitude edges integrate Dinosaur's Gaussian quadrature weights.
    """
    source=np.asarray(field.transpose('lat','lon'),dtype=float)
    if not np.isfinite(source).all():
        raise ValueError('reference surface map contains missing values')
    nlat,nlon=source.shape
    lon_source=(np.linspace(0.,2.*np.pi,nlon+1) if longitude_bounds is None else
        np.deg2rad(np.asarray(longitude_bounds)))
    lat_source=(np.linspace(-.5*np.pi,.5*np.pi,nlat+1) if latitude_bounds is None else
        np.deg2rad(np.asarray(latitude_bounds)))
    if lon_source.size!=nlon+1 or lat_source.size!=nlat+1:
        raise ValueError('surface cell bounds do not match the raster')
    longitude=np.asarray(grid.longitudes)
    spacing=2.*np.pi/longitude.size
    if not np.allclose(np.diff(longitude),spacing,rtol=1e-12,atol=1e-12):
        raise ValueError('conservative surface remap requires a uniform periodic longitude grid')
    lon_target=np.concatenate([longitude-.5*spacing,longitude[-1:]+.5*spacing])
    quadrature=np.asarray(grid.quadrature_weights).reshape(-1,len(grid.latitudes)).sum(axis=0)
    lat_target=np.concatenate([[-1.],-1.+2.*np.cumsum(quadrature)/quadrature.sum()])
    lat_target[-1]=1.
    lat_weights=_overlap_weights(np.sin(lat_source),lat_target)
    lon_weights=_overlap_weights(lon_source,lon_target,2.*np.pi)
    return np.asarray(lon_weights@(lat_weights@source).T)


def surface_boundary_fields_on_grid(grid, path: str | Path):
    """Interpolate an explicitly supplied surface dataset to nodal arrays.

    The file must contain ``albedo`` and ``thermal_inertia`` on ``lat``/``lon``.
    No reference-climatology dataset is imported or selected implicitly.
    """
    import xarray as xr

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"surface boundary file not found: {path}"
        )
    with xr.open_dataset(path) as opened:
        names = [name for name in ("albedo", "thermal_inertia", "emissivity", "roughness")
                 if name in opened]
        if not {"albedo", "thermal_inertia"}.issubset(names):
            raise ValueError(f"surface dataset {path} lacks albedo or thermal_inertia")
        ds = opened[names].load()
    result=interpolate_surface_fields(grid,ds)
    albedo=result['albedo']
    inertia=result['thermal_inertia']
    if not np.isfinite(albedo).all() or not np.isfinite(inertia).all():
        raise ValueError(f"surface dataset {path} contains unfillable missing values")
    if not np.all((albedo >= 0.0) & (albedo <= 1.0)) or not np.all(inertia > 0.0):
        raise ValueError(f"surface dataset {path} contains nonphysical values")
    fields = {"albedo": albedo, "thermal_inertia": inertia}
    for name in ("emissivity", "roughness"):
        if name in result:
            value = result[name]
            if not np.isfinite(value).all() or np.any(value <= 0.0):
                raise ValueError(f"surface dataset {path} has invalid {name}")
            fields[name] = value
    return fields


def surface_fields_on_grid(grid, path: str | Path):
    """Backward-compatible albedo/thermal-inertia pair."""
    fields = surface_boundary_fields_on_grid(grid, path)
    return fields["albedo"], fields["thermal_inertia"]


def reference_surface_fields_on_grid(grid,data_directory):
    """Load the supplied reference TES/GRS maps without substituting climatology.

    The latitude/longitude interpolation adapts the reference input maps to
    Dinosaur. Retain the source's 800-TIU maximum north of 65 degrees.
    """
    import xarray as xr

    source=Path(data_directory)
    fields={}
    inputs={
        'thermal_inertia':('mars_thermal_inertia8_2011.nc','thin'),
        'albedo':('mars_TES_albedo8_new.nc','albedo'),
        'emissivity':('mars_sfc_emissivity8.nc','emis'),
        'grs_ice_latitude_deg':('GRS_interp16.nc','grs_indices'),
    }
    for name,(filename,variable) in inputs.items():
        path=source/filename
        with xr.open_dataset(path) as opened:
            if variable not in opened or set(opened[variable].dims)!={'lat','lon'}:
                raise ValueError(f'{path} lacks a latitude/longitude {variable} field')
            value=conservative_surface_field(grid,opened[variable],
                opened['lonb'] if 'lonb' in opened else None,
                opened['latb'] if 'latb' in opened else None)
        if not np.isfinite(value).all():
            raise ValueError(f'{path} has unfillable missing values')
        fields[name]=value
    if np.any(fields['thermal_inertia']<=0.):
        raise ValueError('thermal inertia must be positive')
    if np.any((fields['albedo']<0.)|(fields['albedo']>1.)):
        raise ValueError('albedo must be in [0, 1]')
    # The reference TES map includes values up to 1.0093; Ames does not clip them.
    if np.any(fields['emissivity']<=0.) or np.any(np.abs(fields['grs_ice_latitude_deg'])>90.):
        raise ValueError('reference emissivity or GRS ice-boundary latitude is invalid')
    north=np.degrees(np.asarray(grid.latitudes))[None,:]>65.
    fields['thermal_inertia']=np.where(north,np.minimum(fields['thermal_inertia'],800.),
        fields['thermal_inertia'])
    return fields
