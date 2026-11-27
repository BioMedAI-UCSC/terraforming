"""Verify pinned polar/terrain inputs and their conservative Gaussian remap."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import xarray as xr
from src.framework.gcm.coordinates import coordinate_system
from src.celestials.planets.mars.boundary_inputs import (INPUT_BLOBS,_read_field,
    polar_water_on_grid,terrain_standard_deviation_on_grid)


def validate(source):
    cases=[]
    for truncation in ('T21','T42'):
        grid=coordinate_system(truncation,4).horizontal
        area=np.asarray(grid.quadrature_weights);area=area/area.sum()
        for filename,var in [('npcflag8.nc','npcflag')]+[
                ('palmer_drag_input_c24_c48_c96.nc',f'SD{n}') for n in (24,48,96)]:
            mapped=_read_field(grid,source/'data',filename,var)
            with xr.open_dataset(source/'data'/filename) as ds:
                value=np.asarray(ds[var])
                bounds=np.linspace(-np.pi/2.,np.pi/2.,value.shape[0]+1)
                latitude_area=np.diff(np.sin(bounds))
                expected=np.sum(value*latitude_area[:,None])/(2.*value.shape[1])
            actual=float(np.sum(area*mapped))
            np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=3e-12)
            if var=='npcflag':
                water=polar_water_on_grid(grid,source/'data')
                np.testing.assert_array_equal(water,np.where(mapped>.5,500.,0.))
            else:
                np.testing.assert_array_equal(terrain_standard_deviation_on_grid(grid,source/'data',int(var[2:])),mapped)
            cases.append(dict(truncation=truncation,field=var,source_mean=float(expected),
                mapped_mean=actual,mapped_min=float(mapped.min()),mapped_max=float(mapped.max())))
        np.testing.assert_array_equal(terrain_standard_deviation_on_grid(grid,source/'data',36),
            terrain_standard_deviation_on_grid(grid,source/'data',24))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference_input_blobs=INPUT_BLOBS,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_boundary_inputs.py','package/src/celestials/planets/mars/boundary_inputs.py',
             'package/src/celestials/planets/mars/surface.py')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=validate(args.ames_source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} boundary maps passed")


if __name__=='__main__': main()
