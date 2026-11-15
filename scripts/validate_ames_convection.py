"""Compare JAX convection with the unmodified, pinned Ames pass2 routine."""
from __future__ import annotations

import argparse
import ctypes
import dataclasses
import hashlib
import json
import platform
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.ames_convection import pass2
from src.celestials.planets.mars.ames import AMES_BODY_3D
from ames_reference import compile_routines


AMES_COMMIT = "ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26"
UPDATE_BLOB = "d0351317dfb6edd4c83c01c67d9ad3a39bff3b2d"


def compile_reference(source, destination, compiler):
    path = source / "atmos_param_mars/update_mars_atmos.F90"
    # Time is unused in pass2. These declarations replace only FMS plumbing.
    header = """module reference_mod
use iso_c_binding
implicit none
real :: kappa
logical :: mix_tracers=.true., mix_momentum=.true.
type time_type
integer :: unused=0
end type
contains
"""
    adapter = """
subroutine ames_pass2(ncol,nz,ntr,kappa_in,t,u,v,r,ph,pf,tout,uout,vout,rout) bind(C)
integer(c_int), value :: ncol,nz,ntr
real(c_double), value :: kappa_in
real(c_double), intent(in) :: t(ncol,1,nz),u(ncol,1,nz),v(ncol,1,nz),r(ncol,1,nz,ntr)
real(c_double), intent(in) :: ph(ncol,1,nz+1),pf(ncol,1,nz)
real(c_double), intent(out) :: tout(ncol,1,nz),uout(ncol,1,nz),vout(ncol,1,nz),rout(ncol,1,nz,ntr)
real :: zero(ncol,1,nz),zero_r(ncol,1,nz,ntr),delp(ncol,1,nz)
type(time_type) :: clock
kappa=kappa_in
zero=0.
zero_r=0.
delp=ph(:,:,2:nz+1)-ph(:,:,1:nz)
call pass2(1,ncol,1,1,nz,ntr,1.,clock,u,v,t,r,zero,zero,zero,zero_r, &
           uout,vout,tout,rout,delp,ph,pf)
tout=t+tout
uout=u+uout
vout=v+vout
rout=r+rout
end subroutine
end module
"""
    library,provenance = compile_routines(path,UPDATE_BLOB,["pass2"],header,adapter,destination,compiler)
    native = ctypes.CDLL(str(library)).ames_pass2
    array = np.ctypeslib.ndpointer(dtype=np.float64, flags="F_CONTIGUOUS")
    native.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double] + [array] * 10
    native.restype = None
    return native,dict(commit=AMES_COMMIT,**provenance)


def native_arrays(values):
    t, u, v, r, ph, pf, _ = values
    nz, ntr = t.shape[0], r.shape[-1]
    def scalar(x):
        return np.asfortranarray(x.reshape(x.shape[0], -1).T[:,None,:])
    return [scalar(t),scalar(u),scalar(v),
        np.asfortranarray(r.reshape(nz,-1,ntr).transpose(1,0,2)[:,None,:,:]),
        scalar(ph),scalar(pf)]


def call_native(native, values, arrays, outputs):
    t, _, _, r, _, _, kappa = values
    native(int(np.prod(t.shape[1:])), t.shape[0], r.shape[-1], kappa, *arrays, *outputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ames-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compiler", default="gfortran")
    parser.add_argument("--columns", type=int, default=2048)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    if args.columns < 1 or args.layers < 2 or args.repeats < 1:
        parser.error("positive columns/repeats and at least two layers required")
    compiler = shutil.which(args.compiler)
    if compiler is None:
        parser.error("Fortran compiler is required for an independent reference")
    jax.config.update("jax_enable_x64", True)
    compiled = jax.jit(pass2)
    rng = np.random.default_rng(20261114)
    cases = []
    with tempfile.TemporaryDirectory(prefix="ames-convection-") as temporary:
        native, provenance = compile_reference(args.ames_source, Path(temporary), compiler)
        for name in ("stable", "unstable", "mixed", "nonuniform"):
            n, c = args.layers, args.columns
            sigma = (np.linspace(0.,1.,n+1) if name != "nonuniform"
                     else np.linspace(0.,1.,n+1)**1.7)
            ph = sigma[:,None] * rng.uniform(400.,900.,c)[None]
            pf = (ph[:-1] + ph[1:]) * .5
            theta = (np.linspace(280.,180.,n)[:,None] if name == "stable"
                     else np.linspace(180.,280.,n)[:,None])
            if name in ("mixed", "nonuniform"):
                theta = rng.uniform(160.,300.,(n,c))
            kappa = AMES_BODY_3D.kappa
            t = theta * ((sigma[:-1]+sigma[1:])*.5)[:,None]**kappa
            u, v = rng.normal(0.,50.,(2,n,c))
            r = rng.uniform(0.,1e-3,(n,c,3))
            values = (np.broadcast_to(t,(n,c)).copy(),u,v,r,ph,pf,kappa)
            arrays = native_arrays(values)
            outputs = [np.empty_like(x) for x in arrays[:4]]
            call_native(native,values,arrays,outputs)
            reference = [x[:,0,:].T.reshape(values[0].shape) for x in outputs[:3]]
            reference.append(outputs[3][:,0,:,:].transpose(1,0,2).reshape(r.shape))
            device_values = tuple(jax.device_put(x) for x in values)
            start = time.perf_counter()
            actual = compiled(*device_values)
            jax.block_until_ready(actual)
            compile_seconds = time.perf_counter()-start
            errors = []
            for actual_field, reference_field in zip(actual,reference,strict=True):
                actual_field = np.asarray(actual_field)
                np.testing.assert_allclose(actual_field,reference_field,rtol=2e-12,atol=2e-12)
                errors.append(float(np.max(np.abs(actual_field-reference_field))))
            start = time.perf_counter()
            for _ in range(args.repeats):
                call_native(native,values,arrays,outputs)
            native_seconds = (time.perf_counter()-start)/args.repeats
            start = time.perf_counter()
            for _ in range(args.repeats):
                jax.block_until_ready(compiled(*device_values))
            jax_seconds = (time.perf_counter()-start)/args.repeats
            cases.append(dict(case=name,max_absolute_errors=dict(zip(
                ("temperature_k","u_ms","v_ms","tracer_ratio"),errors,strict=True)),
                native_seconds=native_seconds,jax_seconds=jax_seconds,
                native_over_jax_speed_ratio=native_seconds/jax_seconds,
                first_jax_call_seconds=compile_seconds))
    report = dict(status="pass",reference=provenance,cases=cases,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/ames_convection.py").read_bytes()).hexdigest(),
        columns=args.columns,layers=args.layers,repeats=args.repeats,
        constants=dict(body=dataclasses.asdict(AMES_BODY_3D),source="Ames v3.2.1 patches/srcFMS.patch"),
        platform=platform.platform(),jax_version=jax.__version__,
        devices=[str(x) for x in jax.devices()],
        limits="pass2 kernel only; FP64 tolerance agreement, not bitwise identity or whole-model climate/speed equivalence")
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__ == "__main__":
    main()
