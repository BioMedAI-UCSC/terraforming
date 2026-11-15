"""Compare JAX two-stream flux kernels with unchanged Ames gfluxv/gfluxi."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.ames_twostream import infrared_fluxes,visible_fluxes
from src.framework.physics.ames_radiation_grid import radiation_grid
from ames_reference import compile_routines
from validate_ames_radiation_tables import RADIATION_BLOB


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer :: l_nlayrad,l_nlevrad,l_levels,l_layers
integer, parameter :: l_taumax=35
real*8, parameter :: maxexp=35.d0,ubari=.5d0,pi=acos(-1.d0)
real*8 :: planckir(1,8501)
contains
"""
    adapter="""
subroutine ames_grid(n,half,full,t,tg,top,p,tl,pm,tm,dp) bind(C)
integer(c_int), value :: n
real(c_double), value :: tg,top
real(c_double), intent(in) :: half(n+1),full(n),t(n)
real(c_double), intent(out) :: p(2*n+3),tl(2*n+3),pm(2*n+3),tm(2*n+3),dp(n)
real*8 :: pl(2*n+3),input_t(2*n+3)
integer :: k
l_layers=n
l_levels=2*n+3
pl=0.d0
input_t=0.d0
do k=1,n
    pl(2*k+1)=half(k)
    pl(2*k+2)=full(k)
    input_t(2*k+2)=t(k)
enddo
pl(2*n+3)=half(n+1)
call fillpt(pl,half(1),tg,top,input_t,p,tl,pm,tm,dp)
end subroutine
subroutine ames_ir(n,rsf,top,surface,dtau,w,g,b,offset,up,down,topnet) bind(C)
integer(c_int), value :: n
real(c_double), value :: rsf,top,surface
real(c_double), intent(in) :: dtau(n),w(n),g(n),b(n+1),offset(n)
real(c_double), intent(out) :: up(n),down(n),topnet(1)
real*8 :: tlev(2*n+1),cumul(2*n+1),tau
integer :: l,nt
l_nlayrad=n
l_nlevrad=n+1
l_levels=2*n+1
tlev=0.
cumul=0.
planckir=0.
tau=0.
do l=1,n
    tlev(2*l)=100+l
    nt=int(tlev(2*l)*10.d0-499.d0)
    planckir(1,nt)=b(l)
    cumul(2*l)=tau
    cumul(2*l+1)=tau+offset(l)
    tau=tau+dtau(l)
enddo
tlev(2*n+1)=101+n
nt=int(tlev(2*n+1)*10.d0-499.d0)
planckir(1,nt)=b(n+1)
call gfluxi(n+1,tlev,1,1.d0,dtau,cumul,w,g,rsf,top,surface,topnet(1),up,down)
end subroutine
subroutine ames_visible(n,mu0,f0,rsf,top,surface,dtau,w,g,cumul,up,down,result) bind(C)
integer(c_int), value :: n
real(c_double), value :: mu0,f0,rsf,top,surface
real(c_double), intent(in) :: dtau(n),w(n),g(n),cumul(2*n+1)
real(c_double), intent(out) :: up(n),down(n),result(4)
real*8 :: tau(n+1),fu,fd,diff,detau
integer :: l
l_nlayrad=n
l_nlevrad=n+1
l_levels=2*n+1
tau(1)=0.
do l=1,n
    tau(l+1)=tau(l)+dtau(l)
enddo
call gfluxv(dtau,tau,cumul,w,g,mu0,f0,rsf,top,surface,up,down,diff,fu,fd,detau)
result=[fu,fd,diff,detau]
end subroutine
end module
"""
    library,provenance=compile_routines(source/"atmos_param_mars/Rad/rtmod_mgcm.F90",
        RADIATION_BLOB,["fillpt","dsolver","dtridgl","gfluxv","gfluxi"],
        header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags="F_CONTIGUOUS")
    native.ames_ir.argtypes=[ctypes.c_int]+[ctypes.c_double]*3+[array]*8
    native.ames_ir.restype=None
    native.ames_visible.argtypes=[ctypes.c_int]+[ctypes.c_double]*5+[array]*7
    native.ames_visible.restype=None
    native.ames_grid.argtypes=[ctypes.c_int]+[array]*3+[ctypes.c_double]*2+[array]*5
    native.ames_grid.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update("jax_enable_x64",True)
    rng=np.random.default_rng(20261115)
    cases=[]
    grid_cases=[]
    with tempfile.TemporaryDirectory(prefix="ames-two-stream-") as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (1,4,12,40):
            half=np.geomspace(.001,6.1,n+1)
            full=(half[1:]-half[:-1])/np.log(half[1:]/half[:-1])
            t=np.asfortranarray(rng.uniform(145.,250.,n))
            expected=[np.zeros(2*n+3) for _ in range(4)]+[np.zeros(n)]
            native.ames_grid(n,half,full,t,230.,140.,*expected)
            actual=radiation_grid(half,full,t,230.,140.)
            grid_errors=[]
            for a,ref in zip(actual,expected,strict=True):
                np.testing.assert_allclose(a,ref,rtol=2e-14,atol=1e-13)
                grid_errors.append(float(np.max(np.abs(np.asarray(a)-ref))))
            grid_cases.append(dict(layers=n,max_absolute_errors=grid_errors))
            for label,base_tau in (("thin",1e-4),("moderate",.1),("thick",3.)):
                tau=np.asfortranarray(rng.uniform(.5,1.5,n)*base_tau)
                w=np.asfortranarray(rng.uniform(.05,.95,n))
                g=np.asfortranarray(rng.uniform(.05,.85,n))
                b=np.asfortranarray(np.linspace(.1,.3,n+1))
                offset=np.asfortranarray(tau*.5);offset[-1]=tau[-1]
                ir_up=np.zeros(n);ir_down=np.zeros(n);ir_top=np.zeros(1)
                native.ames_ir(n,.1,.001,.27,tau,w,g,b,offset,ir_up,ir_down,ir_top)
                actual=infrared_fluxes(tau,w,g,b,offset,.1,.001,.27)
                errors=[]
                for a,ref in zip(actual,(ir_up,ir_down,ir_top[0]),strict=True):
                    np.testing.assert_allclose(a,ref,rtol=2e-10,atol=2e-12)
                    errors.append(float(np.max(np.abs(np.asarray(a)-ref))))
                cumulative=np.zeros(2*n+1)
                cumulative[1::2]=np.concatenate([[0.],np.cumsum(tau[:-1])])
                cumulative[2:-1:2]=np.cumsum(tau[:-1])-.5*tau[:-1]
                cumulative[-1]=np.sum(tau)
                delta=float(np.sum(tau*(1.-w*g*g)))
                surface=.2*.6*100.*np.exp(-min(delta/.6,35.))
                up=np.zeros(n);down=np.zeros(n);result=np.zeros(4)
                native.ames_visible(n,.6,100.,.2,0.,surface,tau,w,g,cumulative,up,down,result)
                visible=visible_fluxes(tau,w,g,cumulative,.6,100.,.2,0.,surface)
                visible_errors=[]
                for a,ref in zip(visible,(up,down,*result),strict=True):
                    np.testing.assert_allclose(a,ref,rtol=2e-10,atol=2e-12)
                    visible_errors.append(float(np.max(np.abs(np.asarray(a)-ref))))
                cases.append(dict(layers=n,case=label,infrared_max_errors=errors,visible_max_errors=visible_errors))
    return dict(status="pass",reference=provenance,cases=cases,radiation_grid_cases=grid_cases,
        radiation_grid_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/ames_radiation_grid.py").read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/ames_twostream.py").read_bytes()).hexdigest(),
        vertical_solver_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/vertical_solve.py").read_bytes()).hexdigest(),
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits="gfluxv/gfluxi with supplied optical/Planck arrays; full radiation driver and coupled global validation remain pending")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ames-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which("gfortran")
    if compiler is None:
        parser.error("gfortran is required for the independent reference")
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
