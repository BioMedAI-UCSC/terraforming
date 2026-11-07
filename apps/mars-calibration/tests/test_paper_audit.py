from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
from audit_paper_results import paired_timestep, paired_skill, field_diagnostics, SOL


def fields():
    nodes,_=np.polynomial.legendre.leggauss(2)
    dims=('time','sigma','lat','lon')
    shape=(3,2,2,2)
    return xr.Dataset({name:(dims,np.full(shape,value,dtype=float)) for name,value in
                       [('air_temperature',200.),('eastward_wind',0.),('northward_wind',0.),('air_pressure',610.)]},
                      coords=dict(time=np.array([1000.,1001.,1004.])/SOL,sigma=[.25,.75],
                                  lat=np.rad2deg(np.arcsin(nodes)),lon=[0.,180.]))


def test_uneven_time_weights_and_absolute_clock_offset():
    a=fields(); b=fields()
    b.air_temperature.values[1:]+=2.
    _,report=paired_timestep(a,b,first_dt=300.,second_dt=150.)
    assert report['sample_span_seconds']==pytest.approx(4.)
    result=report['fields']['air_temperature']
    assert result['equal_snapshot_rmse']==pytest.approx(np.sqrt(8/3))
    assert result['time_integrated_rmse']==pytest.approx(np.sqrt(3.5))
    with pytest.raises(ValueError,match='physical output times'):
        paired_timestep(a,b.assign_coords(time=b.time+1/SOL),first_dt=300.,second_dt=150.)


def test_extrema_location_and_local_sound_speed():
    ds=fields()
    ds.eastward_wind.values[1,0,1,1]=500.
    report=field_diagnostics(ds,dt_seconds=300.)
    assert report['max_wind_ms']==500.
    assert report['max_wind_location']['lon']==180.
    assert report['max_mach']==pytest.approx(500/np.sqrt(770/(770-188.92)*188.92*200))
    assert report['maximum_below_co2_frost_point_k']==0.


def test_uncertainty_requires_paired_forecasts():
    frame=pd.DataFrame([dict(window=w,split='test',block=b,seconds=10.,model=m,temperature_rmse_k=e)
                        for w,b in [('a','my34_q0'),('b','my35_q0')]
                        for m,e in [('physical',3.),('calibrated',2.)]])
    result=paired_skill(frame)
    assert result['year_blocks']==2
    assert result['gain_k']==1.
    with pytest.raises(ValueError,match='paired'):
        paired_skill(frame.iloc[:-1])
