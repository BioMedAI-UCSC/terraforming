"""Check terrain gravity-wave drag against unchanged released native routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.terrain_drag import tendencies,infrared_damping
from native_reference import compile_routines


BLOB='e25163faee55b4c4d31d90f2d09db13148230123'


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
type time_type
integer :: unused=0
end type
real :: grav=3.72,cp_air=735.94,rdgas=189.02,kap=5e-6
logical :: do_IR_damping=.true.,mcpu0
real :: topo_SD(1,1)
contains
"""
    adapter="""
subroutine terrain(n,u,v,t,p,ph,z,zh,sd,damping,du,dv) bind(C)
integer(c_int), value :: n,damping
real(c_double), intent(in) :: u(n),v(n),t(n),p(n),ph(n+1),z(n),zh(n+1)
real(c_double), value :: sd
real(c_double), intent(out) :: du(n),dv(n)
real :: ux(1,1,n),vx(1,1,n),tx(1,1,n),px(1,1,n),phx(1,1,n+1)
real :: zx(1,1,n),zhx(1,1,n+1),dux(1,1,n),dvx(1,1,n)
type(time_type) :: clock
ux(1,1,:)=u
vx(1,1,:)=v
tx(1,1,:)=t
px(1,1,:)=p
phx(1,1,:)=ph
zx(1,1,:)=z
zhx(1,1,:)=zh
topo_SD=sd
do_IR_damping=damping/=0
call palmer_drag(1,1,1,1,n,300.d0,ux,vx,tx,px,phx,zx,zhx,dux,dvx,clock)
du=dux(1,1,:)
dv=dvx(1,1,:)
end subroutine
subroutine damping(p,d,w) bind(C)
real(c_double), value :: p
real(c_double), intent(out) :: d(1),w(1)
call IR_damping(p,d(1),w(1))
end subroutine
subroutine diag(du,dv,stress,clock)
real :: du(:,:,:),dv(:,:,:),stress(:,:)
type(time_type) :: clock
end subroutine
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
end module
"""
    library,provenance=compile_routines(source/'atmos_param_mars/palmer_topo_drag.F90',
        BLOB,['palmer_drag','IR_damping'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.terrain.argtypes=[ctypes.c_int]+[array]*7+[ctypes.c_double,ctypes.c_int,array,array]
    native.terrain.restype=None
    native.damping.argtypes=[ctypes.c_double,array,array]
    native.damping.restype=None
    return native,provenance


def column(n,mode):
    ph=np.geomspace(.001,610.,n+1);p=np.sqrt(ph[:-1]*ph[1:])
    z=10000.*np.log(610./p);zh=10000.*np.log(610./ph)
    t=np.linspace(145.,215.,n)
    x=np.linspace(0.,1.,n)
    u=12.+45.*x;v=-3.+8.*x
    if mode=='critical': u[:n//2]=-20.
    if mode=='unstable': t=220.*(p/p[-1])**.3
    if mode=='near-uniform': u=30.+.01*x;v=5.+.02*x
    if mode=='strong-shear': u=5.+100.*x**2;v=30.*np.sin(3.*x)
    return tuple(np.asfortranarray(a) for a in (u,v,t,p,ph,z,zh))


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='terrain-native-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for p in np.geomspace(.001,1.e6,40):
            d=np.empty(1);w=np.empty(1);native.damping(p,d,w)
            np.testing.assert_allclose(infrared_damping(p),(d[0],w[0]),rtol=4e-12,atol=3e-12)
        for n in (8,12,40):
            run=jax.jit(tendencies,static_argnames=('ir_damping',))
            for mode in ('sheared','critical','unstable','near-uniform','strong-shear'):
                args=column(n,mode)
                for sd in (10.,300.,1500.):
                    for damping in (False,True):
                        du=np.empty(n);dv=np.empty(n)
                        native.terrain(n,*args,sd,int(damping),du,dv)
                        result=run(*args,sd,ir_damping=damping)
                        np.testing.assert_allclose(result.u_tendency,du,rtol=3e-10,atol=2e-13)
                        np.testing.assert_allclose(result.v_tendency,dv,rtol=3e-10,atol=2e-13)
                        cases.append(dict(layers=n,mode=mode,terrain_std_m=sd,ir_damping=damping,
                            max_u_error_m_s2=float(np.max(np.abs(result.u_tendency-du))),
                            max_v_error_m_s2=float(np.max(np.abs(result.v_tendency-dv))),
                            unresolved_stress_pa=float(result.unresolved_stress_pa)))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_terrain_drag.py','package/src/framework/physics/terrain_drag.py')},
        limits='calm-wind and zero-shear limits have separate analytic/gradient tests; original missing impulses retained and reported')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} terrain columns passed")


if __name__=='__main__': main()
