"""Compare CIA subinterval depths against the original optci15 and kinter."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.collision_absorption import collision_induced_depths,load_tables
from native_reference import compile_routines

RADIATION_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer :: l_layers,l_levels,l_nlayrad,l_nlevrad
integer,parameter :: l_nspectv=7,l_nspecti=8,l_ngauss=17,l_nrefv=6,l_pint=51,l_refh2o=10,l_ntref=7,l_npref=11
real*8,parameter :: cmk=3.51d22
logical :: use_boxinterp12=.true.,do_cia=.true.
real*8 :: co2i(7,51,10,8,17),pfgasref(51),pgasref(11),tgasref(7)
real*8 :: kgbar_tab(8,15),kbbar_tab(8,15),khbar_tab(8,15),cia_co2,cia_h2
real*8 :: wrefh2o(10)=[1.d-7,1.d-6,1.d-5,1.d-4,1.d-3,.01d0,.1d0,.2d0,.3d0,.4d0]
contains
"""
    adapter="""
subroutine collision(n,carbon,hydrogen,ps,t,kg,kb,kh,answer) bind(C)
integer(c_int),value :: n
real(c_double),value :: carbon,hydrogen
real(c_double),intent(in) :: ps(2*n+3),t(2*n+3),kg(8,15),kb(8,15),kh(8,15)
real(c_double),intent(out) :: answer(2*n+2,8)
real*8 :: p(2*n+3),pm(2*n+3),tm(2*n+3),water(2*n+3)
real*8 :: ref(2*n+4),one(2*n+4),zero(2*n+4),zi(2*n+4,8)
real*8 :: depth(n+1,8,17),cumulative(2*n+3,8,17),w(n+1,8,17),g(n+1,8,17),gas(8,16)
integer :: i
l_layers=n
l_levels=2*n+3
l_nlayrad=n+1
l_nlevrad=n+2
cia_co2=carbon
cia_h2=hydrogen
kgbar_tab=kg
kbbar_tab=kb
khbar_tab=kh
co2i=-200.d0
do i=1,51
 pfgasref(i)=-6.d0+(i-1)*.2d0
enddo
do i=1,7
 tgasref(i)=50.d0*i
enddo
p=ps/100.d0
pm=p
tm=t
water=1.d-7
one=1.d0
zero=0.d0
ref=0.d0
zi=0.d0
call optci15(depth,cumulative,p,t,one,zi,zi,zi,g,w,ref,tm,pm,gas,water, &
 one,zero,zi,zi,zi,one,zero,zi,zi,zi)
do i=1,2*n+2
 answer(i,:)=cumulative(i+1,:,17)-cumulative(i,:,17)
enddo
end subroutine
end module
"""
    lib,provenance=compile_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,
        ['optci15','tpindex15','boxinterp','kinter'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(lib));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.collision.argtypes=[ctypes.c_int]+[ctypes.c_double]*2+[array]*6
    native.collision.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);data=load_tables();cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='collision-absorption-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (2,12,40):
            for surface in (610.,50000.,1e6):
                half=np.geomspace(.001,surface,n+1);half[0]=0.
                p=np.zeros(2*n+3);p[2::2]=half;p[3::2]=.5*(half[1:]+half[:-1])
                for cold,hot in ((50.,90.),(140.,300.),(750.,950.)):
                    t=np.linspace(cold,hot,2*n+3)
                    for carbon,hydrogen in ((1.,0.),(.9,.1),(.5,.5)):
                        expected=f(np.zeros((2*n+2,8)))
                        native.collision(n,carbon,hydrogen,f(p),f(t),f(data['co2_rotational']),
                            f(data['co2_translational']),f(data['co2_h2']),expected)
                        actual=collision_induced_depths(p,t,co2_molar_fraction=carbon,h2_molar_fraction=hydrogen)
                        # Cumulative native output loses some precision on differencing thin layers.
                        bound=64.*np.finfo(float).eps*np.max(np.abs(expected),axis=0,keepdims=True)
                        error=np.abs(np.asarray(actual)-expected)
                        assert np.all(error<=3e-12*np.abs(expected)+bound+1e-15)
                        cases.append(dict(layers=n,surface_pressure_pa=surface,temperature_range_k=[cold,hot],
                            co2_molar_fraction=carbon,h2_molar_fraction=hydrogen,max_absolute_error=float(np.max(error))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_collision_absorption.py','scripts/stage_collision_absorption.py',
        'package/src/framework/physics/collision_absorption.py','package/src/framework/physics/collision_absorption_tables.npz']
    return dict(status='pass',reference=provenance,cases=cases,
        optical_inputs=json.loads(str(data['source_provenance_json'])),
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
