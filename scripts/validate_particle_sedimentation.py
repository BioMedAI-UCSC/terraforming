"""Compare dust/coated-cloud moment settling with unchanged native routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.particle_sedimentation import sedimentation_step
from src.framework.physics.water_moments import size_grid
from native_reference import compile_routines,extract_routines

BLOB='df1424b6258a68d7dfb6c7b5ebfb7180224070ad'
SOLVER_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler):
    solver,_=extract_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',SOLVER_BLOB,['dtridgl'])
    header="""module constants_mod
implicit none
real*8,parameter :: grav=3.72d0,cp_air=735.94d0,rdgas=189.02d0,pi=acos(-1.d0)
end module
module initracer_mod
implicit none
integer,parameter :: nbin=4,ntrace_mom=6,iMa_dt=1,iNb_dt=2,iMa_cld=3,iNb_cld=4,iMa_cor=5,iMa_vap=6
integer,parameter :: ndust_mass=1,nice_mass=1,nt_nontag=6
integer :: mom_indx(6)=(/1,2,3,4,5,6/),sedim_indx(6)=0
integer :: dust_mass_indx(1)=1,dust_nb_indx(1)=2,dust_cor_indx(1)=5,ice_mass_indx(1)=3,ice_nb_indx(1)=4
real*8,parameter :: athird=1.d0/3.d0,dpden_ice=917.d0,dpden_dt=2500.d0
real*8,parameter :: scale_dt=3.d0*(1.38064852d-23/1.66054d-27)/44.d0
real*8 :: aerdens(6)=(/2500.d0,2500.d0,917.d0,917.d0,917.d0,0.d0/)
real*8 :: stdv(6)=(/.63676d0,.63676d0,.3087d0,.3087d0,.3087d0,0.d0/),aerad(4)
end module
module reference_mod
use iso_c_binding
use constants_mod
use initracer_mod
implicit none
logical :: scaveng=.true.
contains
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
"""
    adapter='\n'+solver[0]+"""
subroutine settle(n,dt,ts,half,full,t,q,kh,radii,scavenge,out,deposit,density,radius) bind(C)
integer(c_int),value :: n,scavenge
real(c_double),value :: dt,ts
real(c_double),intent(in) :: half(n+1),full(n),t(n),q(n,6),kh(n-1),radii(4)
real(c_double),intent(out) :: out(n,6),deposit(6),density(n,6),radius(n,6)
real*8 :: p(2*n+3),tl(2*n+3),qpi(2*n+3,6),kd(2*n+1),rho(n)
integer :: l,k,flags(6)
p=0;tl=0;qpi=0;kd=0;density=0;radius=0
aerad=radii;scaveng=scavenge/=0
p(3)=half(1)
do l=1,n
 k=2*l+2
 p(k)=full(l);p(k+1)=half(l+1);tl(k)=t(l);qpi(k,:)=max(q(l,:),0.d0)
 if(l<n) then
  tl(k+1)=.5d0*(t(l)+t(l+1));kd(2*l+1)=kh(l)
 else
  tl(k+1)=.5d0*(t(l)+ts)
 endif
 rho(l)=full(l)/(rdgas*t(l))
enddo
tl(3)=tl(4)
call init_sedim(n,dt,qpi,density,radius,flags)
call sedim(n,dt,p,tl,rho,kd,qpi,radius,density,deposit,flags)
do l=1,n
 out(l,:)=qpi(2*l+2,:)
enddo
end subroutine
end module
"""
    path,provenance=compile_routines(source/'atmos_param_mars/sedim_mod.F90',BLOB,
        ['init_sedim','sedim','fallvel'],header,adapter,destination,compiler)
    provenance['solver_git_blob']=SOLVER_BLOB
    provenance['solver_sha256']=hashlib.sha256(solver[0].encode()).hexdigest()
    native=ctypes.CDLL(str(path));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.settle.argtypes=[ctypes.c_int]+[ctypes.c_double]*2+[array]*6+[ctypes.c_int]+[array]*4
    native.settle.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='particle-sedimentation-native-') as tmp:
        native,provenance=compile_reference(source,Path(tmp),compiler)
        for n in (2,4,12,40):
            for ps in (610.,5e4):
                half=np.r_[0.,np.geomspace(.1,ps,n)];full=.5*(half[:-1]+half[1:])
                t=np.linspace(150.,220.,n)
                base=np.array([1e-5,1e8,1e-6,1e6,1e-8,1e-6])
                for mode in ('clear','uniform','gradient'):
                    q=np.zeros((n,6)) if mode=='clear' else np.broadcast_to(base,(n,6)).copy()
                    if mode=='gradient': q*=np.geomspace(.01,10.,n)[:,None]
                    for dt in (5.,300.,3e4):
                        for mixing in (0.,.001,100.):
                            for scavenging in (False,True):
                                kh=np.full(n-1,mixing)
                                out=f(np.empty_like(q));deposit=np.empty(6)
                                den=f(np.empty_like(q));rad=f(np.empty_like(q))
                                native.settle(n,dt,230.,f(half),f(full),f(t),f(q),f(kh),f(size_grid()[0]),
                                    int(scavenging),out,deposit,den,rad)
                                actual=sedimentation_step(q,half,full,t,230.,kh,dt,scavenging=scavenging)
                                assert np.all(np.isfinite(actual.tracers)) and np.all(np.isfinite(out))
                                for index in range(6):
                                    np.testing.assert_allclose(actual.tracers[...,index],out[...,index],
                                        rtol=3e-10,atol=1e-20 if index in (0,2,4,5) else 1e-8)
                                np.testing.assert_allclose(actual.surface_deposit,deposit,rtol=3e-10,atol=1e-18)
                                np.testing.assert_allclose(actual.particle_density,den[...,:5],rtol=3e-14)
                                np.testing.assert_allclose(actual.radius_m,rad[...,:5],rtol=3e-14,atol=1e-20)
                                cases.append(dict(layers=n,pressure_pa=ps,mode=mode,dt_seconds=dt,
                                    mixing_m2_s=mixing,scavenging=scavenging,
                                    max_mass_ratio_error=float(np.max(abs(actual.tracers[:,[0,2,4,5]]-out[:,[0,2,4,5]]))),
                                    implicit_species=int(np.sum(actual.diagnostics['transport']['implicit']))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_particle_sedimentation.py','scripts/native_reference.py',
        'package/src/framework/physics/particle_sedimentation.py',
        'package/src/framework/physics/co2_microphysics.py','package/src/framework/physics/vertical_solve.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Six physical moments, source density/radius and explicit/implicit transport. '
            'Tagged populations and lifting overrides require separate validation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native particle-sedimentation cases passed")
