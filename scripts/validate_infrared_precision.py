"""Validate stable IR fluxes against unchanged Fortran equations in IEEE quadruple precision."""
import ctypes
import hashlib
from pathlib import Path
import subprocess

import numpy as np
from native_reference import extract_routines

RADIATION_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler,*,mode='12band'):
    """Compile original bodies at 128 bits; convert only at the C boundary.

    The ordinary FP64 oracle remains useful for measuring cancellation. This
    oracle tests the equations without accepting that cancellation as physics.
    """
    bands=5 if mode=='12band' else 8
    channels=33 if mode=='15band' else 17
    names=['sfluxi','gfluxi','dsolver','dtridgl']
    routines,blob=extract_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,names)
    header=f"""module reference_mod
implicit none
integer :: l_nlayrad,l_nlevrad,l_levels
integer,parameter :: l_nspecti={bands},l_ngauss={channels},l_taumax=35
real*8,parameter :: maxexp=35.d0,ubari=.5d0,pi=acos(-1.d0),tlimiti=5.d-3
real*8 :: planckir({bands},8501),fzeroi({bands}),gweight({channels-1}),dwni({bands})
logical :: do_irflux_scale=.false.
contains
"""
    adapter=f"""module c_adapter
use iso_c_binding
use reference_mod
implicit none
contains
subroutine ames_lw(n,rsf,scale,dtau,w,g,cumul,gas,p,t,planck,widths,clear,weights,up,down,top) bind(C)
integer(c_int),value :: n,scale
real(c_double),value :: rsf
real(c_double),intent(in) :: dtau(n,{bands},{channels}),w(n,{bands},{channels}),g(n,{bands},{channels})
real(c_double),intent(in) :: cumul(2*n+1,{bands},{channels}),gas({bands},{channels-1}),p(2*n+1),t(2*n+1)
real(c_double),intent(in) :: planck({bands},8501),widths({bands}),clear({bands}),weights({channels-1})
real(c_double),intent(out) :: up(n),down(n),top(1)
real(kind=16) :: net(n),tops({bands}),nets({bands},n),ups({bands},n),downs({bands},n)
real(kind=16) :: result_up(n),result_down(n),result_top
l_nlayrad=n
l_nlevrad=n+1
l_levels=2*n+1
do_irflux_scale=scale/=0
fzeroi=real(clear,kind=16)
gweight=real(weights,kind=16)
planckir=real(planck,kind=16)
dwni=real(widths,kind=16)
call sfluxi(real(p,kind=16),real(t,kind=16),real(dtau,kind=16),real(cumul,kind=16), &
 real(rsf,kind=16),real(g,kind=16),real(w,kind=16),result_top,net,result_up,result_down, &
 real(gas,kind=16),tops,nets,ups,downs,.false.)
up=real(result_up,kind=c_double)
down=real(result_down,kind=c_double)
top(1)=real(result_top,kind=c_double)
end subroutine
end module
"""
    destination=Path(destination)
    code=destination/'quad_reference.f90';bridge=destination/'quad_adapter.f90'
    code.write_text(header+'\n'.join(routines)+'\nend module\n');bridge.write_text(adapter)
    common=['-O3','-fPIC','-cpp']
    quad_flags=[*common,'-fdefault-real-8','-fdefault-double-8','-freal-8-real-16']
    commands=[
        [compiler,*quad_flags,'-c',str(code),'-o',str(destination/'reference.o')],
        [compiler,*common,'-c',str(bridge),'-o',str(destination/'adapter.o')],
        [compiler,'-shared',str(destination/'reference.o'),str(destination/'adapter.o'),'-o',str(destination/'quad.so')]]
    for command in commands:
        result=subprocess.run(command,cwd=destination,capture_output=True,text=True)
        if result.returncode: raise RuntimeError(f'quadruple reference compilation failed:\n{result.stderr}')
    native=ctypes.CDLL(str(destination/'quad.so'))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.ames_lw.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_int]+[array]*14
    native.ames_lw.restype=None
    return native,dict(source_git_blob=blob,routine_sha256={name:hashlib.sha256(body.encode()).hexdigest()
        for name,body in zip(names,routines,strict=True)},real_precision_bits=128,
        source_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        adapter_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),compiler_flags=quad_flags,
        adaptation='unchanged routine bodies; compiler promotes REAL*8 to IEEE binary128; separate C adapter converts FP64 input/output')
