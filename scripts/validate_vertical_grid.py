"""Compare all released hybrid grids with unchanged native set_eta."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.vertical_grid import coefficients,pressure_interfaces,sigma_boundaries,supported_layers
from stage_vertical_grids import compile_reference


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    with tempfile.TemporaryDirectory(prefix='vertical-grid-parity-') as temporary:
        native,reference=compile_reference(source,Path(temporary),compiler)
        for n in supported_layers():
            for pref in (61.,705.,725.,5e4):
                a,b,ks,top=native(n,pref)
                actual_a,actual_b=coefficients(n,pref)
                np.testing.assert_allclose(actual_a,a,rtol=2e-15,atol=1e-18)
                np.testing.assert_array_equal(actual_b,b)
                for factor in (.5,1.,1.5):
                    surface=factor*pref;expected=a+b*surface
                    actual=pressure_interfaces(n,pref,surface)
                    np.testing.assert_allclose(actual,expected,rtol=2e-15,atol=1e-17)
                    sigma=sigma_boundaries(n)
                    if factor==1.:
                        np.testing.assert_allclose(sigma[1:]*pref,expected[1:],rtol=2e-15)
                    cases.append(dict(layers=n,reference_pressure_pa=pref,surface_pressure_pa=surface,
                        pure_pressure_layers=ks,native_top_pressure_pa=top,
                        max_pressure_error_pa=float(np.max(abs(actual-expected))),
                        sigma_adaptation_max_pressure_error_pa=float(np.max(abs(sigma*surface-expected)))))
    root=Path(__file__).parents[1]
    paths=('scripts/validate_vertical_grid.py','scripts/stage_vertical_grids.py',
        'package/src/framework/physics/vertical_grid.py',
        'package/src/framework/physics/vertical_grid_coefficients.npz')
    return dict(status='pass',reference=reference,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
        limits='Hybrid equations match native. Dinosaur freezes reference-column fractions and uses a zero top and arithmetic centers; terrain and full-level discretization differences remain explicit.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native vertical-grid cases passed")
