"""Check JAX PBL kernels against unchanged NASA Ames Fortran routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.ames_pbl import eddy_coefficients,surface_exchange,column_step
from ames_reference import compile_routines


PBL_BLOB="38b3f41e4d07f9a2e3c9d7a689e50577cac7858c"


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
real*8 :: grav,cp,rgas
real*8, parameter :: KAPA=.25684d0
logical, parameter :: latent_heat=.false.,do_sublimation=.false.,do_injection=.false.
logical :: htflux_recalc=.false.,mcpu0
real*8 :: z0_pbl,epsl0_pbl,vk_pbl,alphl0_pbl,ric_pbl,dtmu_pbl,rmu1mu_pbl,rmu
contains
"""
    adapter="""
subroutine ames_eddy(n,g,dt,psi,dxi,xinum,z,du,km,kh,q0) bind(C)
integer(c_int), value :: n
real(c_double), value :: g,dt
real(c_double), intent(in) :: psi(2*n+1,3),dxi(2*n+1),xinum(2*n+1),z(2*n+1)
real(c_double), intent(inout) :: du(2*n+1)
real(c_double), intent(out) :: km(2*n+1),kh(2*n+1),q0(2*n+1)
real*8 :: ric,vk,alphl0,epsl0
grav=g
ric=.195d0
vk=.4d0
alphl0=.1d0
epsl0=.1d0
km=0.
kh=0.
q0=0.
call eddycoef(3,ric,vk,alphl0,epsl0,dxi,q0,du,xinum,z,km,kh,psi,dt,n)
end subroutine
subroutine ames_boundary(n,g,tg,z0,psi,z,result) bind(C)
integer(c_int), value :: n
real(c_double), value :: g,tg,z0
real(c_double), intent(in) :: psi(2*n+1,3),z(2*n+1)
real(c_double), intent(out) :: result(6)
real*8 :: vk,rlnzz,ric,ustar,thstar,dphi,w,cdh,cdm
grav=g
vk=.4d0
ric=.195d0
rlnzz=log(z(2*n)/z0)
call bndcond(3,tg,vk,rlnzz,z0,ustar,thstar,ric,dphi,w,z,psi,cdh,cdm,n)
result(1)=cdh
result(2)=cdm
result(3)=ustar
result(4)=thstar
result(5)=w
result(6)=g*z(2*n)/(psi(2*n,3)*w*w+1.d-9)*dphi
end subroutine
subroutine ames_column(n,ntr,g,rg,cpin,dt,t,u,v,r,ph,pf,tg,co2ice,waterice, &
    tout,uout,vout,rout,km,kh,fluxes) bind(C)
integer(c_int), value :: n,ntr
real(c_double), value :: g,rg,cpin,dt,tg,co2ice,waterice
real(c_double), intent(in) :: t(n),u(n),v(n),r(n,ntr),ph(n+1),pf(n)
real(c_double), intent(out) :: tout(n),uout(n),vout(n),rout(n,ntr),km(2*n+1),kh(2*n+1),fluxes(9)
real*8 :: tl(2*n+3),pl(2*n+3),du(2*n+1),qrad(n),tt(n),ut(n),vt(n),qt(n,ntr),qgt
real*8 :: rhouch,heat,strx,stry,sup,sdn,h2oflux,co2g,ustar,thstar,cdm,cdh,dsensa,dsenss
real*4 :: latent
integer :: k,l
logical :: polarcap
grav=g
rgas=rg
cp=cpin
tl=0.
pl=0.
pl(3)=ph(1)
pl(2)=pl(3)*.5
pl(1)=pl(2)*1.e-6
do k=1,n
    l=2*k+2
    tl(l)=t(k)
    pl(l)=pf(k)
    pl(l+1)=ph(k+1)
    if(k.eq.n) then
        tl(l+1)=.5*(tg+t(k))
    else
        tl(l+1)=.5*(t(k+1)+t(k))
    endif
enddo
tl(3)=tl(4)
du=0.
km=0.
kh=0.
qrad=0.
h2oflux=0.
co2g=co2ice
polarcap=waterice.gt.100.
call initpbl(n,dt,du)
call pbl_driver(ntr+3,ntr,n,tl,pl,u,v,r,waterice,tg,ph(n+1),dt,h2oflux,qrad, &
    polarcap,tt,ut,vt,qt,qgt,rhouch,heat,strx,stry,kh,km,sup,sdn,du,latent,co2g, &
    ustar,thstar,cdm,cdh,dsensa,dsenss)
tout=t+tt*dt
uout=u+ut*dt
vout=v+vt*dt
rout=r+qt*dt
fluxes=[heat,strx,stry,rhouch,ustar,thstar,cdm,cdh,dsenss]
end subroutine
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
end module
"""
    names=["initpbl","pbl_driver","newpbl","eddycoef","bndcond","scldef","watsat_pbl",
        "matrix","solve","descale","solve_unstable","fh","fhprime","fm","fmprime"]
    library,provenance=compile_routines(source/"atmos_param_mars/BLcode/pblmod_mgcm.F90",
        PBL_BLOB,names,header,adapter,destination,compiler)
    lib=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags="F_CONTIGUOUS")
    lib.ames_eddy.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_double]+[array]*8
    lib.ames_eddy.restype=None
    lib.ames_boundary.argtypes=[ctypes.c_int]+[ctypes.c_double]*3+[array]*3
    lib.ames_boundary.restype=None
    lib.ames_column.argtypes=[ctypes.c_int]*2+[ctypes.c_double]*4+[array]*6+[ctypes.c_double]*3+[array]*7
    lib.ames_column.restype=None
    return lib,provenance


def validate(source,compiler):
    jax.config.update("jax_enable_x64",True)
    rng=np.random.default_rng(20261115)
    cases=[]
    with tempfile.TemporaryDirectory(prefix="ames-pbl-") as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (4,12,40):
            for stability in ("stable","unstable","zero-wind"):
                psi=np.asfortranarray(rng.normal(0.,20.,(2*n+1,3)))
                psi[:,2]=np.linspace(220.,180.,2*n+1)
                if stability=="unstable":
                    psi[:,2]=psi[::-1,2].copy()
                if stability=="zero-wind":
                    psi[:,:2]=0.
                dxi=np.asfortranarray(rng.uniform(.01,.1,2*n+1))
                xinum=np.asfortranarray(rng.uniform(-.02,-.001,2*n+1))
                z=np.asfortranarray(np.linspace(20000.,10.,2*n+1))
                du=np.asfortranarray(rng.uniform(0.,.01,2*n+1))
                previous=du.copy()
                km,kh,q0=(np.zeros_like(du) for _ in range(3))
                native.ames_eddy(n,3.72,300.,psi,dxi,xinum,z,du,km,kh,q0)
                actual=eddy_coefficients(psi[1:-2:2],psi[3::2],dxi[1:-2:2],
                    dxi[3::2],xinum[2:-1:2],z[2:-1:2],previous[2:-1:2],300.,3.72)
                errors=[]
                for a,b in zip(actual,(km,kh,q0,du),strict=True):
                    a=np.asarray(a); b=b[2:-1:2]
                    np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)
                    errors.append(float(np.max(np.abs(a-b))))
                surface=np.zeros(6,dtype=np.float64)
                tg=psi[-2,2]+(-10. if stability=="stable" else 10.)
                native.ames_boundary(n,3.72,tg,.01,psi,z,surface)
                boundary=np.asarray(surface_exchange(psi[-2,2],tg,psi[-2,0],psi[-2,1],z[-2],.01,3.72))
                np.testing.assert_allclose(boundary,surface,rtol=2e-12,atol=2e-12)
                ph=np.asfortranarray(np.linspace(0.,1.,n+1)**1.3*610.)
                pf=np.asfortranarray((ph[:-1]+ph[1:])*.5)
                t=np.asfortranarray(psi[1::2,2])
                u=np.asfortranarray(psi[1::2,0])
                v=np.asfortranarray(psi[1::2,1])
                r=np.asfortranarray(rng.uniform(0.,1e-4,(n,2)))
                tnew,unew,vnew=(np.zeros_like(t) for _ in range(3))
                rnew=np.zeros_like(r)
                fullkm,fullkh=(np.zeros(2*n+1) for _ in range(2))
                fluxes=np.zeros(9)
                co2ice=1. if stability=="unstable" else 0.
                native.ames_column(n,2,3.72,189.02,735.94,300.,t,u,v,r,ph,pf,tg,co2ice,0.,
                    tnew,unew,vnew,rnew,fullkm,fullkh,fluxes)
                full=column_step(t,u,v,r,ph,pf,tg,300.,189.02,735.94,3.72,surface_co2_ice=co2ice)
                full_errors=[]
                for a,b in zip(full[:4],(tnew,unew,vnew,rnew),strict=True):
                    np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)
                    full_errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                diagnostics=full[4]
                for key,reference in (("momentum_diffusivity_m2_s",fullkm[2:-1:2]),
                        ("heat_diffusivity_m2_s",fullkh[2:-1:2])):
                    np.testing.assert_allclose(diagnostics[key],reference,rtol=2e-12,atol=2e-12)
                actual_fluxes=[diagnostics[x] for x in ("heat_flux_w_m2","stress_x_pa","stress_y_pa")]
                np.testing.assert_allclose(actual_fluxes,fluxes[:3],rtol=2e-12,atol=2e-12)
                cases.append(dict(layers=n,case=stability,
                    max_absolute_eddy_errors=dict(zip(("km","kh","richardson","shear_squared"),errors,strict=True)),
                    max_absolute_surface_error=float(np.max(np.abs(boundary-surface))),
                    max_absolute_column_errors=dict(zip(("temperature_k","u_ms","v_ms","tracer_ratio"),full_errors,strict=True))))
    return dict(status="pass",reference=provenance,cases=cases,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/ames_pbl.py").read_bytes()).hexdigest(),
        vertical_solver_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            "package/src/framework/physics/vertical_solve.py").read_bytes()).hexdigest(),
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits="gas PBL driver with default switches only; global surface energy coupling and model validation remain pending")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ames-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which("gfortran")
    if compiler is None:
        parser.error("gfortran is required for an independent reference")
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
