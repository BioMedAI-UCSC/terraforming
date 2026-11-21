"""Report the first five-sol experiment outputs; arguments: runs native coordinates output."""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import xarray as xr


def statistics(values, weights):
    values = np.asarray(values)
    weights = np.broadcast_to(weights, values.shape)
    assert np.isfinite(values).all() and np.isfinite(weights).all()
    assert np.all(weights >= 0.) and weights.sum() > 0.
    weights = weights / weights.sum()
    return dict(mean=float(np.sum(values * weights)), rms=float(np.sqrt(np.sum(values**2 * weights))),
                minimum=float(values.min()), maximum=float(values.max()))


def pressure_slice(values, pressure, target):
    values, pressure = values.reshape(56, -1), pressure.reshape(56, -1)
    assert np.isfinite(values).all()
    assert np.all(pressure > 0.) and np.all(np.diff(pressure, axis=0) > 0.)
    return np.array([np.interp(np.log(target), np.log(p), v, left=np.nan, right=np.nan)
                     for p, v in zip(pressure.T, values.T, strict=True)])


def profiles(u, temperature, pressure, height, weights):
    weights = weights.ravel() / weights.sum()
    result = dict(top_zonal_wind_m_s=statistics(u[0].ravel(), weights),
                  top_temperature_k=statistics(temperature[0].ravel(), weights),
                  pressure_pa=[statistics(layer.ravel(), weights) for layer in pressure],
                  height_above_surface_m=[statistics(layer.ravel(), weights) for layer in height],
                  common_pressure=[])
    for target in (.01, .03, .1, .3, 1., 3., 10., 30., 100.):
        wind = pressure_slice(u, pressure, target)
        air = pressure_slice(temperature, pressure, target)
        valid = np.isfinite(wind) & np.isfinite(air)
        result['common_pressure'].append(dict(pressure_pa=target, area_fraction=float(weights[valid].sum()),
            zonal_wind_m_s=statistics(wind[valid], weights[valid]) if valid.any() else None,
            temperature_k=statistics(air[valid], weights[valid]) if valid.any() else None))
    return result


def analyse(runs, native, coordinates):
    results, hashes = {}, {}

    def record(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return path

    for case in ('T21', 'T42'):
        root = runs / case / 'reference'
        config = json.loads(record(root / 'config.json').read_text())
        context = dict(initial_pressure_scheme=config['initial_pressure_scheme'],
            native_top_divergence_damping=config.get('native_top_divergence_damping'),
            truncation=config['truncation'], layers=config['layers'], dt_seconds=config['dt'],
            runtime=[json.loads(record(path).read_text()) for path in sorted((runs / case).glob('runtime-*.json'))])
        rows = list(csv.DictReader(record(root / 'checkpoint_diagnostics.csv').open()))
        assert rows and all(np.isfinite(float(v)) for r in rows for v in r.values())
        if int(rows[-1]['step']) != 1920:
            trace = record(runs / f'{case}.log').read_text()
            failure = re.search(r'FloatingPointError: reference became non-finite before step (\d+);', trace)
            if failure is None:
                raise ValueError(f'{case} has neither a completed five-sol case nor a recorded finite-state failure')
            restarts = sorted((root / 'checkpoints').glob('restart_*.npz'))
            if restarts:
                record(restarts[-1])
            record(root / 'executed_source.zip')
            results[case] = dict(status='failed_nonfinite', last_valid_diagnostic_step=int(rows[-1]['step']),
                last_saved_restart_step=int(restarts[-1].stem.split('_')[-1]) if restarts else None,
                failure_before_step=int(failure[1]), source_sha256=config['source_sha256'],
                input_sha256=config['reference_input_sha256'], **context)
            continue
        with xr.open_dataset(record(root / 'checkpoints/mean_000001920.nc'), decode_times=False) as ds:
            assert ds.attrs['complete_window'] == 1 and ds.attrs['averaging_steps'] == 1920
            assert np.allclose(ds.time_bounds.values, [[0., 5.]], rtol=0., atol=1e-10)
            nodes, weights = np.polynomial.legendre.leggauss(ds.sizes['lat'])
            assert np.allclose(nodes, np.sin(np.deg2rad(ds.lat)), atol=1e-7)
            weights = np.broadcast_to(weights[:, None], (ds.sizes['lat'], ds.sizes['lon']))
            sigma = ds.sigma.values[:, None, None]
            pressure = sigma * ds.surface_pressure.values[0]
            # Same isothermal reference-height convention as MarsMapFields; not geometric height.
            body = config['body']
            scale_height = body['gas_constant_j_kg_k'] * body['reference_temperature_k'] / body['gravity_m_s2']
            height = np.broadcast_to(-scale_height * np.log(sigma), pressure.shape)
            results[case] = profiles(ds.eastward_wind.values[0], ds.air_temperature.values[0], pressure, height, weights)
            results[case]['status'] = 'completed'
            results[case]['height_definition'] = 'Isothermal 200 K reference height; not simulated geometric height.'
            results[case]['tracer_minimum'] = {name: float(ds[name].min()) for name in ds if name.startswith('tracer_')}
            results[case]['time_bounds_sols'] = ds.time_bounds.values.tolist()
        results[case].update(**context, source_sha256=config['source_sha256'], input_sha256=config['reference_input_sha256'],
            relative_inventory_change={key: float(rows[-1][key]) / float(rows[0][key]) - 1.
                for key in ('total_co2_mass_kg', 'total_water_mass_kg', 'total_dust_mass_kg')},
            maximum_checkpoint_wind_m_s=max(float(row['max_wind_speed_ms']) for row in rows))
        record(root / 'restart.npz')
    metadata = json.loads(record(coordinates / 'metadata.json').read_text())
    native_manifest = json.loads(record(native / 'baseline.json').read_text())
    for case in ('C24', 'C48'):
        fields_path = record(native / f'{case}-first-average.npz')
        coordinates_path = record(coordinates / f'{case}.npz')
        assert hashes[str(fields_path)] == native_manifest[case]['archive_sha256']
        assert hashes[str(coordinates_path)] == metadata[case]['archive_sha256']
        with np.load(fields_path, allow_pickle=False) as fields, np.load(coordinates_path, allow_pickle=False) as coords:
            assert np.allclose(coords['time_bnds'], [0., 5.])
            top = coords['phalf'][0, 0] * 100.  # Saved reference coordinate uses millibars.
            thickness_pa = coords['delp'].transpose(1, 0, 2, 3)
            half = np.concatenate([np.full_like(thickness_pa[:1], top),
                                   top + np.cumsum(thickness_pa, axis=0)], axis=0)
            np.testing.assert_allclose(half[-1], fields['ps'], rtol=1e-6, atol=1e-3)
            pressure = np.diff(half, axis=0) / np.diff(np.log(half), axis=0)
            thickness = coords['delz'].transpose(1, 0, 2, 3)
            height = np.cumsum(thickness[::-1], axis=0)[::-1] - thickness / 2.
            results[case] = profiles(fields['ucomp'].transpose(1, 0, 2, 3), fields['temp'].transpose(1, 0, 2, 3),
                                     pressure, height, fields['area'])
            results[case]['height_definition'] = 'Saved five-sol mean layer thickness, accumulated from local surface to layer midpoint.'
            results[case]['pressure_definition'] = 'Saved mean delp accumulated from saved fixed phalf top; saved ak/bk are zero placeholders.'
            results[case]['coordinate_metadata'] = metadata[case]
            results[case]['native_revision'] = native_manifest[case]['native_revision']
            results[case]['executable_sha256'] = native_manifest[case]['executable_sha256']
    return dict(cases=results, artifact_sha256=hashes, limitations=[
        'Wind RMS is the spatial RMS of the time-mean zonal component.',
        'Common-pressure fields use column log-pressure interpolation of five-sol means, without extrapolation or horizontal remapping.',
        'Dinosaur pressure uses mean surface pressure; native pressure uses saved mean delp. This is not the mean of fields interpolated at each model step.',
        'Native geometric midpoint heights and Dinosaur isothermal reference heights have different definitions.',
        'Five-sol transient cases do not establish climate convergence or isolated device throughput.'])


if __name__ == '__main__':
    # Worked weighted-field check: independent mean 2.5 and RMS sqrt(7).
    known = statistics(np.array([1., 3.]), np.array([1., 3.]))
    assert known['mean'] == 2.5 and known['rms'] == np.sqrt(7.)
    runs, native, coordinates, output = map(Path, sys.argv[1:])
    result = analyse(runs, native, coordinates)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
