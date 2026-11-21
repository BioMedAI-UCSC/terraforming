"""Archive native first-window coordinates beside the existing native fields."""
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr

root = Path('/analysis')
reports = json.loads((root / 'baseline.json').read_text())
output = Path('/work/pr123/native-coordinates')
output.mkdir(parents=True, exist_ok=True)
records = {}
for case, report in reports.items():
    arrays, files = {}, []
    for tile in range(1, 7):
        path = Path(report['sampling']['path'].replace('tile1.nc', f'tile{tile}.nc'))
        with xr.open_dataset(path, decode_times=False) as dataset:
            assert dataset.sizes['pfull'] == 56
            for name in ('ak', 'bk', 'pfull', 'phalf', 'delz', 'delp', 'time_bnds'):
                value = dataset[name].isel(time=0) if 'time' in dataset[name].dims else dataset[name]
                data = np.asarray(value)
                assert np.isfinite(data).all(), (case, tile, name)
                arrays.setdefault(name, []).append(data)
            files.append(dict(path=str(path), time_attributes=dict(dataset.time.attrs),
                              attributes={name: dict(dataset[name].attrs) for name in arrays},
                              first_window=dataset.time_bnds.isel(time=0).values.tolist()))
            assert np.allclose(dataset.time_bnds.isel(time=0), [0., 5.]), files[-1]
        nml = path.parent / 'input.nml'
        files[-1]['input_nml_sha256'] = hashlib.sha256(nml.read_bytes()).hexdigest()
    archive = output / f'{case}.npz'
    np.savez_compressed(archive, **{name: np.stack(values) for name, values in arrays.items()})
    records[case] = dict(files=files, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                         input_nml=(path.parent / 'input.nml').read_text())
(output / 'metadata.json').write_text(json.dumps(records, indent=2))
print(json.dumps({case: record['archive_sha256'] for case, record in records.items()}), flush=True)
