#!/usr/bin/env python3
"""Recompute paper evidence without training, retuning, or modifying source runs.

Large external fields are read incrementally. The manifest distinguishes bundled
evidence from local-only inputs; a checksum does not make an artifact public.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[1]
SOL = 88775.244


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def paired_skill(frame):
    keys = ['window', 'split', 'block', 'seconds']
    if frame.duplicated(keys+['model']).any():
        raise ValueError('duplicate window/model/lead')
    b = frame[frame.model=='physical'].set_index(keys).temperature_rmse_k
    f = frame[frame.model=='calibrated'].set_index(keys).temperature_rmse_k
    if not b.index.equals(f.index) or not np.isfinite([b.values,f.values]).all():
        raise ValueError('paired physical and fitted samples required in the same order')
    merged = pd.DataFrame({'physical_mse': b*b, 'fitted_mse': f*f}).reset_index()
    test = merged[merged.split=='test'].groupby(['window','block'])[['physical_mse','fitted_mse']].mean().reset_index()
    test['year'] = test.block.str.extract(r'^(my\d+)(?:_|$)', expand=False)
    if test.year.isna().any():
        raise ValueError('explicit myYEAR blocks required for the historical year-cluster interval')
    blocks = list(test.groupby('year'))
    if len(blocks)<2:
        raise ValueError('at least two held-out years required')
    gains=[]
    for indices in itertools.product(range(len(blocks)), repeat=len(blocks)):
        sample = pd.concat([blocks[i][1] for i in indices])
        gains.append(np.sqrt(sample.physical_mse.mean())-np.sqrt(sample.fitted_mse.mean()))
    return dict(baseline_rmse_k=float(np.sqrt(test.physical_mse.mean())),
                fitted_rmse_k=float(np.sqrt(test.fitted_mse.mean())),
                gain_k=float(np.sqrt(test.physical_mse.mean())-np.sqrt(test.fitted_mse.mean())),
                year_blocks=len(blocks), windows=len(test),
                paired_year_percentile_interval_95_k=np.quantile(gains,[.025,.975]).tolist(),
                limitation='conditional on two observed Mars years and one fitted run; not population skill or optimizer uncertainty')


def field_diagnostics(ds, *, dt_seconds, radius_m=3389500., gas_constant=188.92, cp=770.):
    """Locate extremes and calculate local Mach and horizontal advective CFL.

    CFL uses physical meridional/zonal grid spacing, including cos(latitude).
    It is a diagnostic, not a stability theorem for a spectral IMEX solver.
    """
    required=['air_temperature','eastward_wind','northward_wind','air_pressure']
    if any(v not in ds for v in required):
        raise ValueError('3-D thermodynamic and wind fields required')
    air, u, v, pressure = [ds[k].transpose('time','sigma','lat','lon').values for k in required]
    if not all(np.isfinite(a).all() for a in (air,u,v,pressure)) or (air<=0).any() or (pressure<=0).any():
        raise ValueError('finite positive temperatures/pressures required')
    speed=np.hypot(u,v)
    gamma=cp/(cp-gas_constant)
    mach=speed/np.sqrt(gamma*gas_constant*air)
    lat=np.deg2rad(ds.lat.values)
    lon=np.unwrap(np.deg2rad(ds.lon.values))
    dx=radius_m*np.cos(lat)*float(np.min(np.diff(lon)))
    dy=radius_m*np.min(np.diff(lat))
    cfl=dt_seconds*(np.abs(u)/dx[None,None,:,None]+np.abs(v)/dy)
    def location(values):
        index=np.unravel_index(np.argmax(values),values.shape)
        return {name:float(ds[name].values[i]) for name,i in zip(['time','sigma','lat','lon'],index)}
    # Same Clausius--Clapeyron constants as the model, evaluated at layer pressure.
    frost=3182.48/(23.3494-np.log(np.maximum(pressure/100.,1e-6)))
    return dict(max_wind_ms=float(speed.max()), max_wind_location=location(speed),
                air_temperature_at_max_wind_k=float(air[np.unravel_index(np.argmax(speed),speed.shape)]),
                max_mach=float(mach.max()), max_mach_location=location(mach),
                max_horizontal_advective_cfl=float(cfl.max()),
                minimum_air_temperature_k=float(air.min()),
                maximum_below_co2_frost_point_k=float(np.maximum(frost-air,0).max()),
                sampled_mach_exceeds_one=bool(mach.max()>1),
                co2_frost_assumption='model total-pressure approximation; not a general gas-mixture saturation calculation',
                interpretation='sampled extrema; local Mach/CFL diagnostics do not establish solver stability or a drag-error cause')


def paired_timestep(a, b, *, first_dt, second_dt):
    if not np.isfinite([first_dt,second_dt]).all() or min(first_dt,second_dt)<=0:
        raise ValueError('positive finite timestep metadata required')
    for coord in ('lat','lon','sigma'):
        if not np.array_equal(a[coord],b[coord]):
            raise ValueError('paired grids must match')
    seconds=np.asarray(a.time)*SOL
    other=np.asarray(b.time)*SOL
    if len(seconds)!=len(other) or not np.allclose(seconds,other,rtol=0,atol=.01):
        raise ValueError('paired physical output times must agree within 0.01 s')
    if len(seconds)<2 or not np.isfinite(seconds).all() or np.any(np.diff(seconds)<=0):
        raise ValueError('at least two finite increasing output times required')
    lat=np.deg2rad(a.lat.values)
    nodes,weights=np.polynomial.legendre.leggauss(len(lat))
    if not np.allclose(np.sin(lat),nodes,atol=1e-6):
        raise ValueError('expected Gaussian latitudes')
    weights=weights/weights.sum()
    # Trapezoid quadrature handles uneven saved times; also retain the historical
    # equally weighted snapshot statistic rather than silently relabeling it.
    rows=[]
    for field in ('air_temperature','eastward_wind','northward_wind'):
        error=a[field].transpose('time','sigma','lat','lon').values-b[field].transpose('time','sigma','lat','lon').values
        if not np.isfinite(error).all():
            raise ValueError('nonfinite paired fields')
        mse=np.mean(np.sum(error**2*weights[None,None,:,None],axis=2),axis=(1,2))
        rows.extend(dict(field=field,absolute_seconds=float(t),seconds=float(t-seconds[0]),rmse=float(np.sqrt(m))) for t,m in zip(seconds,mse))
    table=pd.DataFrame(rows)
    summary={}
    for field,sub in table.groupby('field',sort=False):
        summary[field]=dict(equal_snapshot_rmse=float(np.sqrt(np.mean(sub.rmse**2))),
            time_integrated_rmse=float(np.sqrt(np.trapezoid(sub.rmse**2,sub.seconds)/(seconds[-1]-seconds[0]))),
            first_sample_rmse=float(sub.rmse.iloc[0]),last_sample_rmse=float(sub.rmse.iloc[-1]))
    return table,dict(first_dt_seconds=first_dt,second_dt_seconds=second_dt,
                      common_samples=len(seconds),sample_span_seconds=float(seconds[-1]-seconds[0]),
                      first_absolute_seconds=float(seconds[0]),last_absolute_seconds=float(seconds[-1]),fields=summary,
                      limitation='paired transient sensitivity; not an equilibrated error estimate or a 450/300 s certification')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--skip-fields',action='store_true')
    args=parser.parse_args()
    root=args.data_root.resolve(); out=args.output
    out.mkdir(parents=True,exist_ok=True)
    inputs=[]
    def record(relative, *, bundle=False):
        p=root/relative
        item=dict(path=relative,sha256=digest(p),bytes=p.stat().st_size,
                  availability='bundled' if bundle else 'local_only_not_published')
        if bundle:
            target=out/'inputs'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,target)
            item['bundled_path']=str(target.relative_to(out))
        inputs.append(item)
        return p
    csv=record('iclr-results/01-mola-topography-comparisons/metrics.csv',bundle=True)
    metrics=pd.read_csv(csv)
    columns=['bias','rmse','correlation','model_mean','reference_mean']
    if len(metrics)!=48 or metrics.duplicated(['Ls','reference','field']).any() or not np.isfinite(metrics[columns]).all().all():
        raise ValueError('48 unique finite season-reference-field rows required')
    metrics.groupby(['reference','field','units'])[columns].mean().to_csv(out/'reference-summary.csv')
    for name in ('training.csv','skill.csv','checkpoint.npz','provenance.json','acceptance.json'):
        record('outputs/differentiable/calibration/'+name,bundle=True)
    frame=pd.read_csv(root/'outputs/differentiable/calibration/skill.csv')
    training=pd.read_csv(root/'outputs/differentiable/calibration/training.csv')
    calibration=paired_skill(frame)
    with np.load(root/'outputs/differentiable/calibration/checkpoint.npz') as z:
        calibration['selected_controls']=(np.array([.05,.05,.25])+z['best']*np.array([1.95,1.95,3.75])).tolist()
    calibration.update(updates=int(training['update'].iloc[-1]),
        train_loss_last=float(training.train_loss_before_update.iloc[-1]),
        validation_loss_last=float(training.validation_loss.iloc[-1]),
        last_5_train_loss_change=float(training.train_loss_before_update.iloc[-1]-training.train_loss_before_update.iloc[-6]),
        convergence='not established: fixed 50-update budget, no archived gradient/stationarity stop')
    report=dict(schema_version=1,analysis_kind='historical evidence reanalysis; no new coupled simulation or calibration',
        calibration=calibration,
        reference=dict(temperature_bias_range_k=[float(metrics[metrics.field=='Surface temperature'].bias.min()),float(metrics[metrics.field=='Surface temperature'].bias.max())],
                       limitation='Ames-informed (0.25,0.25,1) tuning; unmatched temporal/vertical sampling; model/reanalysis characterization, not independent validation'),
        pending=['untuned/held-out reference rerun','converged merged-physics reanalysis fit','full 450/300 s long-run convergence',
                 'raw pre-projection long-run energy/momentum accounting','exact submitted manuscript source','public raw field/input release'])
    if not args.skip_fields:
        full=record('outputs/Untitled/full.nc')
        half=record('outputs/paper-year-gpu3/half_timestep.nc')
        full_config=json.loads(record('outputs/Untitled/config.json',bundle=True).read_text())
        half_report=json.loads(record('outputs/paper-year-gpu3/ablations.json',bundle=True).read_text())
        half_dt=next(r['case']['dt'] for r in half_report['records'] if r['case']['name']=='half_timestep')
        with xr.open_dataset(full) as a,xr.open_dataset(half) as b:
            rows,comparison=paired_timestep(a,b,first_dt=full_config['dt_seconds'],second_dt=half_dt)
            rows.to_csv(out/'timestep-by-time.csv',index=False)
            report['timestep_300_150']=comparison
            report['winds']=field_diagnostics(a,dt_seconds=full_config['dt_seconds'])
    for file in ('environment.json','timing.json'):
        record('iclr-results/03-benchmarks/performance-timing/'+file,bundle=True)
    report['performance']=dict(external_speedup='not measured; no claim',original_gpu_model='not recorded')
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (out/'input-manifest.json').write_text(json.dumps(dict(schema_version=1,inputs=inputs),indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
