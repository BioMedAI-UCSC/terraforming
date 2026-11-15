"""Check dust lifting against the unchanged native interactive source block."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.dust_lifting import interactive_step
from native_reference import compile_routines,extract_routines

BLOB='1e470cc1efb44c6627cbe3168c69c2d1f7640570'


def compile_reference(source,destination,compiler):
    path=source/'atmos_param_mars/dust_update.F90'
    routines,_=extract_routines(path,BLOB,['dust_update'])
    text=path.read_text();routine=routines[0]
    declarations=text[text.index('logical,public :: firstcall'):text.index('namelist /dust_update_nml/')]
    local=routine[routine.index('! -----Local'):routine.index('mcpu0 =')]
    start=routine.index('if (interact) then')
    stop=routine.index('endif ! end interact',start)+len('endif ! end interact')
    block=routine[start:stop]
    header="""#define RELEASE
module constants_mod
implicit none
real,parameter :: grav=3.72d0,cp_air=735.94d0,rdgas=189.02d0,pi=acos(-1.d0)
end module
module reference_mod
use iso_c_binding
use constants_mod
implicit none
integer,parameter :: ndust_mass=1,bins_length=2,radbins_length=1
integer :: dust_mass_indx(1)=1,dust_cor_indx(1)=5
real,parameter :: dpden_dt=2500.d0,dev_dt=.63676d0,athird=1.d0/3.d0
real,parameter :: Reff_dd=2.d-6,Reff_stress=2.d-6
real :: sfc_dust_mass(1,1,1)
"""+declarations+'\ncontains\n'
    adapter="""
subroutine interactive(n,top,scheme,settings,half,full,temp,q,flags,outq,state) bind(C)
integer(c_int),value :: n,top,scheme
real(c_double),intent(in) :: settings(10),half(n+1),full(n),temp(n),q(n,6)
integer(c_int),intent(in) :: flags(8)
real(c_double),intent(out) :: outq(n,6),state(4)
integer,parameter :: is=1,ie=1,js=1,je=1,ntp=6
integer :: kd
real :: dt,t(1,1,n),r(1,1,n,6),p_half(1,1,n+1),p_full(1,1,n)
real :: p_pbl(1,1),tsfc(1,1),snow(1,1),frost(1,1),stress(1,1),lat(1,1),shflx(1,1)
integer :: k_pbl(1,1)
"""+local+"""
kd=n;dt=settings(1);tsfc=settings(2);stress=settings(3);shflx=settings(4)
p_pbl=settings(5);lat=settings(6);snow=settings(7);frost=settings(8)
tau_current=settings(9);sfc_dust_mass=settings(10);k_pbl=top+1
interact=.true.;stress_scheme=scheme
DDA=flags(1)/=0;DTH=flags(2)/=0;devils_fv3=flags(3)/=0
inject_pbl_dd=flags(4)/=0;inject_pbl_stress=flags(5)/=0;limited_reservoir=flags(6)/=0
convective_lift=flags(7)/=0;stress_lift=flags(8)/=0
do k=1,n
 t(1,1,k)=temp(k);p_full(1,1,k)=full(k);r(1,1,k,:)=q(k,:)
 p_half(1,1,k)=half(k)
enddo
p_half(1,1,n+1)=half(n+1)
rnew=r;rini=r
rnew(:,:,:,1:2)=max(rnew(:,:,:,1:2),1.d-31)
tnew=t;rho=p_full/(rdgas*tnew)
delp=p_half(:,:,2:n+1)-p_half(:,:,1:n)
dens=dpden_dt;dev=dev_dt;cst=.75d0/(pi*dens)*exp(-4.5d0*dev**2)
qextref=2;flux=0;fluxrem=0;fluxdda=0;fluxdth=0;fluxfv3=0
source_dd=0;source_ws=0;source_bg=0;dmcol=0;dncol=0;dccol=0
masspbl=sum(delp(:,:,top+1:n),dim=3)/grav
masspblfix=delp(:,:,n)/grav
"""+block+"""
do k=1,n
 outq(k,:)=rnew(1,1,k,:)
enddo
state(1)=sfc_dust_mass(1,1,1);state(2)=source_dd(1,1,1);state(3)=source_ws(1,1,1)
state(4)=threshold(1,1)
end subroutine
end module
"""
    library,provenance=compile_routines(path,BLOB,['flux_stress'],header,adapter,destination,compiler)
    provenance.update(interactive_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        local_declarations_sha256=hashlib.sha256(local.encode()).hexdigest(),
        adaptation='Unchanged flux_stress body and contiguous if(interact) source block; '
            'single-column input/output adapter; RELEASE preprocessing; background/dual/tag modes excluded')
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.interactive.argtypes=[ctypes.c_int]*3+[array]*5+[integers]+[array]*2
    native.interactive.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='dust-lifting-native-') as tmp:
        native,provenance=compile_reference(source,Path(tmp),compiler)
        n=4;half=f([0.,50.,200.,400.,610.]);full=f(.5*(half[:-1]+half[1:]))
        t=f([160.,180.,210.,230.]);q=f(np.broadcast_to([1e-5,1e8,0.,0.,0.,1e-6],(n,6)))
        for scheme in range(1,6):
            for flags in ((1,0,0,1,0,0,1,1),(0,1,0,0,1,0,1,1),
                          (0,0,1,1,1,0,1,1),(1,1,1,0,0,1,1,1),
                          (0,0,0,0,0,0,0,1)):
                scenarios=[(*values,260.,20.,30.) for values in ((0.,0.,0.,.1,.5),(.022,0.,0.,.1,-.5),
                    (.05,0.,0.,.1,.5),(.05,2.,0.,.1,.5),(.05,2.,0.,.1,0.),
                    (.05,0.,2.,.1,-.5),(.05,0.,0.,101.,0.))]
                scenarios.extend([(.05,0.,0.,.1,.5,210.,20.,30.),
                    (.05,0.,0.,.1,-.5,260.,20.,29.9),(.05,0.,0.,.1,.5,260.,0.,30.1)])
                for stress,snow,frost,tau,lat,ts,heat,reservoir in scenarios:
                    settings=f([100.,ts,stress,heat,200.,lat,snow,frost,tau,reservoir])
                    out=f(np.empty_like(q));state=np.empty(4)
                    native.interactive(n,1,scheme,settings,half,full,t,q,np.array(flags,dtype=np.int32),out,state)
                    result=interactive_step(q[:,0],q[:,1],t,half,full,ts,stress,heat,1,200.,
                        lat,snow,frost,tau,reservoir,100.,stress_options=dict(scheme=scheme),
                        devil_options=dict(dda=bool(flags[0]),dth=bool(flags[1]),fv3=bool(flags[2])),
                        inject_pbl_devils=bool(flags[3]),inject_pbl_stress=bool(flags[4]),
                        limited_reservoir=bool(flags[5]),convective_lift=bool(flags[6]),stress_lift=bool(flags[7]))
                    np.testing.assert_allclose(result.mass_ratio,out[:,0],rtol=3e-12,atol=1e-20)
                    np.testing.assert_allclose(result.number_ratio,out[:,1],rtol=3e-12,atol=1e-8)
                    np.testing.assert_allclose(result.surface_dust_kg_m2,state[0],rtol=3e-14,atol=1e-14)
                    np.testing.assert_allclose(result.diagnostics['devil_lift_kg_m2'],state[1],rtol=3e-12,atol=1e-20)
                    np.testing.assert_allclose(result.diagnostics['stress_lift_kg_m2'],state[2],rtol=3e-12,atol=1e-20)
                    cases.append(dict(scheme=scheme,flags=flags,stress_pa=stress,snow_kg_m2=snow,
                        frost_kg_m2=frost,opacity=tau,latitude_rad=lat,
                        temperature_surface_k=ts,sensible_heat_flux_w_m2=heat,
                        initial_surface_dust_kg_m2=reservoir,
                        max_mass_ratio_error=float(np.max(abs(result.mass_ratio-out[:,0])))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_dust_lifting.py','scripts/native_reference.py','package/src/framework/physics/dust_lifting.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='One physical interactive dust mode. Background scenario control, dual modes and tags are separate. '
            'Collapsed PBL efficiency uses its finite limit; inactive stress divisions use finite surrogates.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native interactive dust-lifting cases passed")
