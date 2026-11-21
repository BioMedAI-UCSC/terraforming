"""Check saturation pressure/humidity against unchanged native FMS kernels."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.saturation_pressure import load_tables,vapor_pressure,specific_humidity
from stage_saturation_tables import compile_reference,native_tables


def validate(source,mars_source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[];rng=np.random.default_rng(20261115)
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    staged=load_tables()
    with tempfile.TemporaryDirectory(prefix='saturation-native-') as tmp:
        native,provenance=compile_reference(source,mars_source,Path(tmp),compiler)
        for simple,model in ((False,'smithsonian'),(True,'simple')):
            tables,minimum=native_tables(native,simple)
            # Cross-platform libm differences are bounded by 2e-8 relative.
            np.testing.assert_allclose(staged[model+'_tables'],tables,rtol=2e-8,atol=1e-60)
            assert staged[model+'_minimum_k']==minimum
            t=f(np.r_[np.linspace(minimum,623.16,801),rng.uniform(minimum,623.16,1000),
                minimum+np.array([-.151,-.149,-.051,-.049,0.,.049,.051]),
                273.16+np.array([-.051,-.049,0.,.049,.051]),623.16+np.array([.049,.051])])
            p=f(np.geomspace(10.,1e6,len(t)));q=f(rng.uniform(0.,.5,len(t)))
            for phase_index,phase in enumerate(('blend','liquid','ice-liquid')):
                for humidity in (.2,1.,1.1):
                    for exact in (False,True):
                        output=np.empty((len(t),5),order='F');bad=np.empty(len(t),dtype=np.int32)
                        native.checks(len(t),t,p,q,humidity,int(exact),phase_index,output,bad)
                        pressure=vapor_pressure(t,model=model,phase=phase)
                        actual=specific_humidity(t,p,relative_humidity=humidity,vapor_ratio=q,
                            use_exact_qs=exact,model=model,phase=phase)
                        valid=bad==0
                        np.testing.assert_array_equal(pressure.valid,valid)
                        for value,expected in zip((*pressure[:2],*actual[:3]),output.T,strict=True):
                            np.testing.assert_allclose(np.asarray(value)[valid],expected[valid],rtol=2e-8,atol=1e-60)
                            assert np.isnan(np.asarray(value)[~valid]).all()
                        cases.append(dict(model=model,phase=phase,relative_humidity=humidity,
                            exact_qs=exact,points=len(t),valid_points=int(sum(valid)),
                            max_humidity_error=float(np.max(abs(np.asarray(actual.specific_humidity)[valid]-output[valid,2])))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_saturation_pressure.py','scripts/stage_saturation_tables.py','scripts/native_reference.py',
        'package/src/framework/physics/saturation_pressure.py','package/src/framework/physics/saturation_tables.npz']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Original Smithsonian/simple table generators and all three phase tables, nearest-node '
            'quadratic lookup and compute_qs formulas. Native index truncation is retained, including '
            'its lower-edge extension. Invalid temperatures return NaNs and valid=False in JAX; '
            'the FMS public wrapper would raise FATAL. Moist convection coupling is separate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fms-source',type=Path,required=True)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.fms_source,args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native saturation lookup/humidity cases passed")
