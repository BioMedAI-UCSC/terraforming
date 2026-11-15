"""Compare spectral aggregation against original Ames sfluxv/sfluxi."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.ames_radiation import load_ames_co2_tables
from src.framework.physics.ames_spectral import visible_spectrum,infrared_spectrum
from ames_reference import compile_routines
from validate_ames_radiation_tables import RADIATION_BLOB


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer :: l_nlayrad,l_nlevrad,l_levels
integer, parameter :: l_nspectv=7,l_nspecti=5,l_ngauss=17,l_taumax=35
real*8, parameter :: maxexp=35.d0,ubari=.5d0,pi=acos(-1.d0),tlimits=1.d-3,tlimiti=5.d-3
real*8 :: planckir(5,8501),fzerov(7),fzeroi(5),gweight(16),dwni(5)
logical :: do_irflux_scale=.false.
contains
"""
    adapter="""
subroutine ames_sw(n,mu,rsf,dtau,w,g,cumul,gas,sol,clear,weights,up,down,top) bind(C)
integer(c_int), value :: n
real(c_double), value :: mu,rsf
real(c_double), intent(in) :: dtau(n,7,17),w(n,7,17),g(n,7,17),cumul(2*n+1,7,17)
real(c_double), intent(in) :: gas(7,16),sol(7),clear(7),weights(16)
real(c_double), intent(out) :: up(n),down(n),top(1)
real*8 :: tau(n+1,7,17),net(n),diff,detau(7,17)
integer :: k
l_nlayrad=n
l_nlevrad=n+1
l_levels=2*n+1
fzerov=clear
gweight=weights
tau(1,:,:)=0.
do k=1,n
    tau(k+1,:,:)=tau(k,:,:)+dtau(k,:,:)
enddo
call sfluxv(dtau,tau,cumul,rsf,w,g,mu,sol,top(1),net,up,down,diff,gas,detau,.false.)
end subroutine
subroutine ames_lw(n,rsf,scale,dtau,w,g,cumul,gas,p,t,planck,widths,clear,weights,up,down,top) bind(C)
integer(c_int), value :: n,scale
real(c_double), value :: rsf
real(c_double), intent(in) :: dtau(n,5,17),w(n,5,17),g(n,5,17),cumul(2*n+1,5,17)
real(c_double), intent(in) :: gas(5,16),p(2*n+1),t(2*n+1),planck(5,8501),widths(5),clear(5),weights(16)
real(c_double), intent(out) :: up(n),down(n),top(1)
real*8 :: net(n),tops(5),nets(5,n),ups(5,n),downs(5,n)
l_nlayrad=n
l_nlevrad=n+1
l_levels=2*n+1
do_irflux_scale=scale/=0
fzeroi=clear
gweight=weights
planckir=planck
dwni=widths
call sfluxi(p,t,dtau,cumul,rsf,g,w,top(1),net,up,down,gas,tops,nets,ups,downs,.false.)
end subroutine
end module
"""
    lib,provenance=compile_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',
        RADIATION_BLOB,['sfluxv','sfluxi','getdetau','gfluxv','gfluxi','dsolver','dtridgl'],
        header,adapter,destination,compiler)
    native=ctypes.CDLL(str(lib))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.ames_sw.argtypes=[ctypes.c_int]+[ctypes.c_double]*2+[array]*11
    native.ames_sw.restype=None
    native.ames_lw.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.c_int]+[array]*14
    native.ames_lw.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    data=load_ames_co2_tables()
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    rng=np.random.default_rng(104)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='ames-spectral-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (2,5,13):
            for label,gas_scale in (('clear-merged',0.),('mixed',.005),('absorbing',1.)):
                for kind,bands in (('visible',7),('infrared',5)):
                    shape=(n,bands,17)
                    dtau=f(rng.uniform(.01,.3,shape))
                    w=f(rng.uniform(.05,.8,shape));g=f(rng.uniform(.05,.8,shape))
                    cumulative=np.zeros((2*n+1,bands,17))
                    cumulative[1::2]=np.concatenate([np.zeros_like(dtau[:1]),np.cumsum(dtau[:-1],axis=0)],axis=0)
                    cumulative[2:-1:2]=np.cumsum(dtau[:-1],axis=0)-dtau[:-1]*.5
                    cumulative[-1]=dtau.sum(axis=0)
                    cumulative=f(cumulative)
                    gas=f(rng.uniform(0.,2.,(bands,16))*gas_scale)
                    output=[np.zeros(n),np.zeros(n),np.zeros(1)]
                    if kind=='visible':
                        solar=f(data['solar_weights']*590.)
                        native.ames_sw(n,.6,.2,dtau,w,g,cumulative,gas,solar,
                            f(data['clear_fraction_sw']),f(data['gauss_weights']),*output)
                        actual=visible_spectrum(dtau,w,g,cumulative,gas,.6,solar,.2)
                        variants=[(False,actual,output)]
                    else:
                        pressure=f(np.geomspace(.001,6.1,2*n+1))
                        temperature=f(150.+5.*np.arange(2*n+1))
                        variants=[]
                        for scale in (False,True):
                            output=[np.zeros(n),np.zeros(n),np.zeros(1)]
                            native.ames_lw(n,.1,int(scale),dtau,w,g,cumulative,gas,pressure,temperature,
                                f(data['ames_planck_intensity_ir'].T),f(np.diff(data['ir_bounds_cm1'])),
                                f(data['clear_fraction_ir']),f(data['gauss_weights']),*output)
                            actual=infrared_spectrum(dtau,w,g,cumulative,gas,pressure,temperature,.1,
                                scale_surface_flux=scale)
                            variants.append((scale,actual,output))
                    for scale,actual,expected in variants:
                        errors=[]
                        for value,ref in zip(actual,(*expected[:2],expected[2][0]),strict=True):
                            np.testing.assert_allclose(value,ref,rtol=2e-10,atol=3e-10)
                            errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                        cases.append(dict(layers=n,case=label,kind=kind,scale_surface_flux=scale,max_errors_w_m2=errors))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_ames_spectral.py','package/src/framework/physics/ames_spectral.py',
        'package/src/framework/physics/ames_twostream.py','package/src/framework/physics/ames_radiation.py',
        'package/src/framework/physics/vertical_solve.py','package/src/framework/physics/ames_co2_12band.npz']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits='spectral aggregation with supplied optics; continuous Planck interpolation checked at reference temperature nodes; full radiation driver remains pending')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None:
        parser.error('gfortran is required for the independent reference')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
