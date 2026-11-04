"""Small persistent result bundle shared by aerosol CLI and server runs."""
import json
import os
from pathlib import Path
import tempfile
import numpy as np


def _atomic_json(path,value):
    path=Path(path)
    handle,name=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(handle,'w') as stream:
            json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)


def write_results(output_dir,report,case_fields,configuration):
    """Save summary, UI-compatible headline maps and matched NetCDF fields.

    Map rows are latitude and columns longitude. All supplied cases must share
    that grid; an incompatible comparison fails instead of being interpolated.
    """
    import xarray as xr
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if not case_fields: raise ValueError('at least one result case is required')
    primary='active' if 'active' in case_fields else next(iter(case_fields))
    headline=case_fields[primary]
    latitude=np.asarray(headline['lat']);longitude=np.asarray(headline['lon'])
    shape=(len(latitude),len(longitude));variables={}
    for case,fields in case_fields.items():
        if not np.array_equal(fields['lat'],latitude) or not np.array_equal(fields['lon'],longitude):
            raise ValueError('matched aerosol result grids differ')
        for name,grid in fields['maps'].items():
            values=np.asarray(grid['data'],dtype=float)
            if values.shape!=shape or not np.all(np.isfinite(values)):
                raise ValueError('result map shape or values are invalid')
            key=name if case==primary else f'{case}_{name}'
            variables[key]=(('lat','lon'),values,dict(units=grid['units'],long_name=grid['label'],experiment_case=case))
    summary=dict(report)
    if 'off' in case_fields and 'active' in case_fields:
        control=case_fields['off']['maps']['surface_temperature']
        active=case_fields['active']['maps']['surface_temperature']
        if control['units']!='K' or active['units']!='K':
            raise ValueError('matched surface temperatures must be kelvin')
        difference=np.asarray(active['data'])-np.asarray(control['data'])
        variables['surface_temperature_active_minus_off']=(('lat','lon'),difference,
            dict(units='K',long_name='Final surface temperature: active minus matched control'))
        cases=report.get('cases',{})
        if 'off' in cases and 'active' in cases:
            summary['final_area_mean_surface_difference_k']=cases['active']['surface_temperature_k']-cases['off']['surface_temperature_k']
    metadata=dict(primary_case=primary,cases=json.dumps(list(case_fields)),
        elapsed_seconds=float(report['elapsed_seconds']),status=report['status'],
        interpretation='transient model comparison; climate baseline and long-term warming unverified',
        configuration=json.dumps(configuration,sort_keys=True,allow_nan=False))
    dataset=xr.Dataset(variables,coords=dict(lat=latitude,lon=longitude),attrs=metadata)
    dataset.lat.attrs.update(units='degrees_north');dataset.lon.attrs.update(units='degrees_east')
    handle,name=tempfile.mkstemp(dir=output,prefix='results.',suffix='.nc')
    os.close(handle)
    try:
        dataset.to_netcdf(name,engine='h5netcdf')
        os.replace(name,output/'results.nc')
    finally:
        if os.path.exists(name): os.unlink(name)
    _atomic_json(output/'fields.json',headline)
    _atomic_json(output/'summary.json',summary)
    return summary
