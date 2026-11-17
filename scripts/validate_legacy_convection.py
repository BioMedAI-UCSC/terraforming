"""Compare legacy convective fields against unchanged released routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.legacy_convection import adjust
from src.celestials.planets.mars.reference_constants import AMES_BODY_3D
from native_reference import compile_routines
from validate_dry_convection import UPDATE_BLOB,native_arrays


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[];rng=np.random.default_rng(20261115)
    header='''module reference_mod
use iso_c_binding
implicit none
real :: kappa
real,parameter :: grav=3.72d0
logical :: mix_tracers=.true.,mix_momentum=.true.
contains
'''
    adapter='''
subroutine legacy(ncol,nz,ntr,kap,t,u,v,r,ph,pf,tout,uout,vout,rout) bind(C)
integer(c_int),value :: ncol,nz,ntr
real(c_double),value :: kap
real(c_double),intent(in) :: t(ncol,1,nz),u(ncol,1,nz),v(ncol,1,nz),r(ncol,1,nz,ntr)
real(c_double),intent(in) :: ph(ncol,1,nz+1),pf(ncol,1,nz)
real(c_double),intent(out) :: tout(ncol,1,nz),uout(ncol,1,nz),vout(ncol,1,nz),rout(ncol,1,nz,ntr)
real :: zero(ncol,1,nz),zeror(ncol,1,nz,ntr),delp(ncol,1,nz)
kappa=kap;zero=0;zeror=0;delp=ph(:,:,2:)-ph(:,:,:nz)
call legacy_convect(1,ncol,1,1,ntr,1.d0,nz,u,v,t,r,zero,zero,zero,zeror, &
    uout,vout,tout,rout,delp,ph,pf)
tout=t+tout;uout=u+uout;vout=v+vout;rout=r+rout
end subroutine
end module
'''
    with tempfile.TemporaryDirectory(prefix='legacy-convection-native-') as tmp:
        library,provenance=compile_routines(source/'atmos_param_mars/update_mars_atmos.F90',
            UPDATE_BLOB,['legacy_convect','potemp1'],header,adapter,Path(tmp),compiler)
        native=ctypes.CDLL(str(library)).legacy
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
        native.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[array]*10;native.restype=None
        for n in (2,4,12,40):
            forward=jax.jit(adjust)
            for top in (0.,.03):
                for name in ('stable','unstable','mixed','neutral','neutral-stable-side','neutral-unstable-side'):
                    c=4;half=np.broadcast_to(np.linspace(top,610.,n+1)[:,None],(n+1,c)).copy()
                    full=.5*(half[1:]+half[:-1]);kap=AMES_BODY_3D.kappa
                    theta=(np.linspace(300.,200.,n)[:,None] if name=='stable' else
                        np.linspace(200.,300.,n)[:,None] if name=='unstable' else
                        np.linspace(250.+1e-8,250.-1e-8,n)[:,None] if name=='neutral-stable-side' else
                        np.linspace(250.-1e-8,250.+1e-8,n)[:,None] if name=='neutral-unstable-side' else
                        np.full((n,c),250.) if name=='neutral' else rng.uniform(160.,320.,(n,c)))
                    t=theta*(full/610.)**kap;u,v=rng.normal(0.,20.,(2,n,c));r=rng.uniform(0.,1e-5,(n,c,3))
                    values=(t,u,v,r,half,full,kap);inputs=native_arrays(values)
                    outputs=[np.empty_like(x) for x in inputs[:4]]
                    native(c,n,3,kap,*inputs,*outputs)
                    expected=[x[:,0,:].T for x in outputs[:3]]
                    expected.append(outputs[3][:,0,:,:].transpose(1,0,2))
                    actual=forward(*values)
                    for index,(value,reference) in enumerate(zip(actual,expected,strict=True)):
                        # Exact neutral T is a discontinuity for nonuniform tracers;
                        # libm/XLA power roundoff can select different sides.
                        if name=='neutral' and index==3:
                            assert np.all(np.isfinite(value)) and np.all(np.isfinite(reference))
                            continue
                        np.testing.assert_allclose(value,reference,rtol=2e-12,atol=2e-12,
                            err_msg=str((n,top,name)))
                    cases.append(dict(layers=n,top_pressure_pa=top,profile=name,
                        tracer_gate_applied=name!='neutral',
                        max_tracer_error=float(np.max(abs(actual[3]-expected[3]))),
                        max_temperature_error_k=float(np.max(abs(actual[0]-expected[0])))))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,cases=cases,
        strict_full_field_cases=sum(case['tracer_gate_applied'] for case in cases),
        neutral_temperature_wind_cases=sum(not case['tracer_gate_applied'] for case in cases),
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_legacy_convection.py','scripts/native_reference.py','scripts/validate_dry_convection.py',
             'package/src/framework/physics/legacy_convection.py')},
        limits='Returned temperature/wind/tracer fields only. Unreturned ghost/interfacial diagnostics '
            'are omitted; their zero-top logarithms do not affect returned full-level fields. '
            'Source inner-loop continuation and unchanged winds are preserved. '
            'Exact neutral temperature has no tracer-parity gate: source libm versus XLA power '
            'roundoff selects opposite sides of a discontinuous tracer-mixing branch. '
            'Both sides at 1e-8 K potential-temperature offsets retain the strict full-field gates. '
            'Conservation residuals and global coupling are separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{report['strict_full_field_cases']} full-field legacy-convection cases passed; "
        f"{report['neutral_temperature_wind_cases']} neutral temperature/wind cases checked separately")
