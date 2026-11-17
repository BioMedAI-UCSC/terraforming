"""Pinned dust scenarios with FMS bilinear sampling on Dinosaur nodes.

Interpolation follows GFDL FMS (Apache License 2.0), including the periodic
seam epsilon and nearest-row polar extension. Scenario conversion follows the
NASA Mars GCM; see its source license and the recorded input provenance.
"""
import hashlib
from pathlib import Path
import numpy as np


INPUT_BLOBS={
    'DustScenario_Background.nc':'ad086574f8def0d45247409811b02bf76afacbbf',
    'DustScenario_MGS.nc':'27b7d7387e3170c63aac609a930d89cef0ff6e88',
    'DustScenario_MY24.nc':'5cdc13d5cd83deedca7f12dcb411a54e5a009c03',
    'DustScenario_MY30.nc':'da305d4b2c1f678abc36a1a82f78ab52d636a851',
    'DustScenario_MY31.nc':'578dc00bb774edd86b3fa984d46e33bb7b31c75e',
    'DustScenario_MY34.nc':'50e8f4288f111c612539d17c2b1b7c505aa0013a',
}


def _bilinear_weights(grid,longitude_bounds_deg,latitude_bounds_deg):
    xb,yb=(np.deg2rad(np.asarray(v,dtype=float)) for v in (longitude_bounds_deg,latitude_bounds_deg))
    if xb.ndim!=1 or yb.ndim!=1 or len(xb)<3 or len(yb)<3 or \
            not np.isfinite(xb).all() or not np.isfinite(yb).all() or \
            np.any(np.diff(xb)<=0.) or np.any(np.diff(yb)<=0.):
        raise ValueError('dust-map bounds must be finite and increase')
    x=.5*(xb[:-1]+xb[1:]);y=.5*(yb[:-1]+yb[1:])
    lon=np.asarray(grid.longitudes,dtype=float)[:,None]
    lon=np.where(lon<0.,lon+2.*np.pi,np.where(lon>2.*np.pi,lon-2.*np.pi,lon))
    lat=np.asarray(grid.latitudes,dtype=float)[None,:]
    upper=np.searchsorted(x,lon,side='right')
    east=np.mod(upper,len(x));west=np.mod(upper-1,len(x))
    seam=(upper==0)|(upper==len(x))
    gap=x[0]-x[-1]+2.*np.pi+1e-10
    seam_weight=np.where(lon<=x[0],(x[0]-lon)/gap,(x[0]-lon+2.*np.pi+1e-10)/gap)
    ww=np.where(seam,seam_weight,(x[east]-lon)/(x[east]-x[west]))
    upper_y=np.searchsorted(y,lat,side='right')
    north=np.clip(upper_y,0,len(y)-1);south=np.clip(upper_y-1,0,len(y)-1)
    outside=(lat<y[0])|(lat>=y[-1])
    span=np.where(outside,1.,y[north]-y[south])
    ws=np.where(outside,1.,(y[north]-lat)/span)
    shape=(len(grid.longitudes),len(grid.latitudes))
    return tuple(np.broadcast_to(a,shape) for a in (west,east,south,north,ww,ws))


def bilinear_map(grid,values,longitude_bounds_deg,latitude_bounds_deg):
    """Sample latitude/longitude cell values; leading dimensions are retained."""
    values=np.asarray(values,dtype=float)
    if values.shape[-2:]!=(len(latitude_bounds_deg)-1,len(longitude_bounds_deg)-1):
        raise ValueError('dust-map data and cell boundaries must match')
    if not np.isfinite(values).all():
        raise ValueError('dust-map fields must be finite')
    west,east,south,north,ww,ws=_bilinear_weights(grid,longitude_bounds_deg,latitude_bounds_deg)
    return (values[...,south,west]*ww*ws+values[...,south,east]*(1.-ww)*ws+
        values[...,north,east]*(1.-ww)*(1.-ws)+values[...,north,west]*ww*(1.-ws))


def load_scenario_on_grid(grid,path,*,map_scale=1.):
    """Load original taufill/2.75 and zmax; retain samples and close missing periodic endpoints."""
    import xarray as xr
    path=Path(path);raw=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if INPUT_BLOBS.get(path.name)!=blob:
        raise ValueError('dust scenario differs from the pinned reference input')
    if not np.isfinite(map_scale) or map_scale<0.:
        raise ValueError('dust scenario scale must be finite and nonnegative')
    with xr.open_dataset(path,decode_times=False) as source:
        seasons=np.asarray(source.areo,dtype=float)
        if not np.isfinite(seasons).all() or np.any(np.diff(seasons)<=0.) or seasons[0]!=0. or seasons[-1]>360.:
            raise ValueError('dust seasons must increase from zero within one year')
        opacity=bilinear_map(grid,source.taufill.transpose('time','lat','lon'),source.lonb,source.latb)*map_scale/2.75
        height=bilinear_map(grid,source.zmax.transpose('time','lat','lon'),source.lonb,source.latb)
    appended=seasons[-1]<360.
    if appended:
        # Four published scenarios omit the final endpoint; the native lookup otherwise reads past its array.
        seasons=np.concatenate([seasons,[360.]])
        opacity=np.concatenate([opacity,opacity[:1]],axis=0)
        height=np.concatenate([height,height[:1]],axis=0)
    if np.any(opacity<0.) or np.any(height<=0.):
        raise ValueError('dust scenario requires nonnegative opacity and positive top height')
    return dict(seasons=seasons,opacity=opacity,top_height_km=height,source_git_blob=blob,periodic_endpoint_appended=bool(appended))


def _load_fraction_fields(grid,path,prefix,mode_count):
    import xarray as xr
    with xr.open_dataset(path,decode_times=False) as source:
        seasons=np.asarray(source.areo,dtype=float)
        bins=np.asarray(source.bins,dtype=float)
        if (seasons.ndim!=1 or len(seasons)<2 or not np.isfinite(seasons).all() or
            np.any(np.diff(seasons)<=0.) or seasons[0]<0. or seasons[-1]!=360.):
            raise ValueError('dust fraction seasons must increase and end at 360 degrees')
        if mode_count is None:
            if (bins.ndim!=1 or len(bins)<2 or not np.isfinite(bins).all() or
                np.any(np.diff(bins)<=0.) or bins[0]!=0. or bins[-1]!=24. or np.any(bins!=np.floor(bins))):
                raise ValueError('local-time bins must increase in integer hours from 0 to 24')
            labels=bins[:-1]
        else:
            if (bins.shape!=(mode_count,) or not np.isfinite(bins).all() or
                np.any(bins!=np.floor(bins)) or np.any((bins<0.)|(bins>99.))):
                raise ValueError('dust fraction bins must name one two-digit field per physical mode')
            labels=bins
        fields=[]
        for label in labels:
            value=source[f'{prefix}{int(label):02d}']
            dimensions=set(value.dims)-{'lat','lon'}
            if len(dimensions)!=1: raise ValueError('dust fraction fields require season, latitude and longitude axes')
            axis=dimensions.pop()
            if value.sizes[axis]!=len(seasons): raise ValueError('dust fraction fields must match seasonal nodes')
            fields.append(bilinear_map(grid,value.transpose(axis,'lat','lon'),source.lonb,source.latb))
    return seasons,bins,np.stack(fields,axis=-1)


def load_injection_fractions_on_grid(grid,path,mode_count):
    """Read source radius_scenario.nc: seasonal opacity shares, not particle radii."""
    if not isinstance(mode_count,int) or isinstance(mode_count,bool) or mode_count<1:
        raise ValueError('dust injection fractions require a positive mode count')
    seasons,_,fractions=_load_fraction_fields(grid,path,'rad',mode_count)
    if np.any(fractions<0.): raise ValueError('injected dust fractions must be nonnegative')
    return dict(seasons=seasons,fractions=fractions)


def load_local_time_fractions_on_grid(grid,path):
    """Read source ltfracHH fields using bin-start hours and native bilinear maps."""
    seasons,bins,fractions=_load_fraction_fields(grid,path,'ltfrac',None)
    return dict(seasons=seasons,bins_hours=bins,fractions=fractions)
