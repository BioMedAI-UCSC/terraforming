"""Compare subgrid optics against unchanged Ames optcv15/optci15."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.correlated_k import load_ames_co2_tables
from src.framework.physics.radiation_grid import radiation_grid
from src.framework.physics.optical_depth import gas_subgrid,aggregate_optics
from native_reference import compile_routines
from validate_correlated_k_tables import RADIATION_BLOB


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer :: l_layers,l_levels,l_nlayrad,l_nlevrad
integer, parameter :: l_nspectv=7,l_nspecti=5,l_ngauss=17,l_nrefv=6,l_pint=51,l_refh2o=10,l_ntref=7,l_npref=11
real*8, parameter :: cmk=3.51d22
logical :: use_boxinterp12=.true.,do_cia=.false.
real*8 :: co2v(7,51,10,7,17),co2i(7,51,10,5,17)
real*8 :: pfgasref(51),pgasref(11),tgasref(7),tauray(7)
real*8 :: kgbar_tab(8,15)=0.,kbbar_tab(8,15)=0.,khbar_tab(8,15)=0.,cia_co2=1.,cia_h2=0.
real*8 :: wrefh2o(10)=[1.d-7,1.d-6,1.d-5,1.d-4,1.d-3,.01d0,.1d0,.2d0,.3d0,.4d0]
contains
"""
    adapter="""
subroutine ames_init(sw,ir,pressure,temperature,ray) bind(C)
real(c_double), intent(in) :: sw(7,51,7,16),ir(7,51,5,16),pressure(51),temperature(7),ray(7)
integer :: i
co2v=-99.
co2i=-99.
co2v(:,:,1,:,1:16)=sw
co2i(:,:,1,:,1:16)=ir
pfgasref=log10(pressure)
tgasref=temperature
tauray=ray
do i=1,11
    pgasref(i)=i-7
enddo
end subroutine
subroutine optical_depth(n,p,tm,pm,amount,xv,sv,gv,xi,si,gi,dv,wv,av,cv,tgv,di,wi,ai,ci,tgi) bind(C)
integer(c_int), value :: n
real(c_double), intent(in) :: p(2*n+3),tm(2*n+3),pm(2*n+3),amount(2*n+2)
real(c_double), intent(in) :: xv(2*n+2,7),sv(2*n+2,7),gv(2*n+2,7)
real(c_double), intent(in) :: xi(2*n+2,5),si(2*n+2,5),gi(2*n+2,5)
real(c_double), intent(out) :: dv(n+1,7,17),wv(n+1,7,17),av(n+1,7,17),cv(2*n+3,7,17),tgv(7,16)
real(c_double), intent(out) :: di(n+1,5,17),wi(n+1,5,17),ai(n+1,5,17),ci(2*n+3,5,17),tgi(5,16)
real*8 :: qxv(2*n+4,7),qsv(2*n+4,7),gsv(2*n+4,7)
real*8 :: qxi(2*n+4,5),qsi(2*n+4,5),gsi(2*n+4,5)
real*8 :: ref(2*n+4),cloudref(2*n+4),zv(2*n+4,7),zi(2*n+4,5),qref(2*n+4),one(2*n+4)
real*8 :: tauv(n+2,7,17),water(2*n+3)
l_layers=n
l_levels=2*n+3
l_nlayrad=n+1
l_nlevrad=n+2
qxv=1.
qsv=0.
gsv=0.
qxi=0.
qsi=0.
gsi=0.
qxv(2:l_levels,:)=xv
qsv(2:l_levels,:)=sv
gsv(2:l_levels,:)=gv
qxi(2:l_levels,:)=xi
qsi(2:l_levels,:)=si
gsi(2:l_levels,:)=gi
ref=0.
ref(2:l_levels)=amount*qxv(2:l_levels,6)
qref=qxv(:,6)
cloudref=0.
zv=0.
zi=0.
one=1.
water=1.d-7
call optcv15(dv,tauv,cv,p,qxv,qsv,gsv,wv,av,ref,tm,pm,tgv,water, &
    one,cloudref,zv,zv,zv,one,cloudref,zv,zv,zv)
call optci15(di,ci,p,tm,qref,qxi,qsi,gsi,ai,wi,ref,tm,pm,tgi,water, &
    one,cloudref,zi,zi,zi,one,cloudref,zi,zi,zi)
end subroutine
end module
"""
    lib,provenance=compile_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,
        ['optcv15','optci15','tpindex15','boxinterp','kinter'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(lib))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.ames_init.argtypes=[array]*5;native.ames_init.restype=None
    native.optical_depth.argtypes=[ctypes.c_int]+[array]*20;native.optical_depth.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    data=load_ames_co2_tables()
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    rng=np.random.default_rng(104)
    cases=[]
    with tempfile.TemporaryDirectory(prefix='ames-optics-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        ray=f(np.geomspace(.0001,.01,7))
        native.ames_init(f(data['log10_k_sw_ames']),f(data['log10_k_ir_ames']),
            f(data['pressure_mbar_ames']),f(data['temperature_k']),ray)
        for n in (1,4,12,40):
            for zero_top in (False,True):
                half=np.geomspace(.1,610.,n+1)
                if zero_top: half[0]=0.
                grid=radiation_grid(half,(half[1:]+half[:-1])*.5,np.linspace(145.,240.,n),250.)
                for loading in (0.,.1):
                    amount=f(np.diff(grid.pressure)/100.*loading)
                    cross=[]
                    for bands in (7,5):
                        ext=f(rng.uniform(.5,3.,(2*n+2,bands)))
                        cross.extend([ext,f(ext*rng.uniform(.1,.9,ext.shape)),f(rng.uniform(0.,.8,ext.shape))])
                    outputs=[]
                    for bands in (7,5):
                        outputs.extend([np.zeros((n+1,bands,17),order='F') for _ in range(3)])
                        outputs.extend([np.zeros((2*n+3,bands,17),order='F'),np.zeros((bands,16),order='F')])
                    native.optical_depth(n,f(grid.pressure/100.),f(grid.lookup_temperature),
                        f(grid.lookup_pressure/100.),amount,*cross,*outputs)
                    gases=gas_subgrid(grid)
                    errors=[]
                    for index,infrared in enumerate((False,True)):
                        x,s,g=cross[index*3:index*3+3]
                        ray_depth=None if infrared else np.diff(grid.pressure)[:,None]/100.*ray[None]
                        actual=aggregate_optics(gases[index],amount[:,None]*x,amount[:,None]*s,
                            amount[:,None]*s*g,ray_depth,infrared=infrared)
                        kind_errors=[]
                        for value,ref in zip(actual,outputs[index*5:index*5+5],strict=True):
                            np.testing.assert_allclose(value,ref,rtol=3e-12,atol=3e-12)
                            kind_errors.append(float(np.max(np.abs(np.asarray(value)-ref))))
                        errors.append(kind_errors)
                    cases.append(dict(layers=n,zero_top=zero_top,dust_loading=loading,max_errors=errors))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_optical_depth.py','package/src/framework/physics/optical_depth.py',
        'package/src/framework/physics/correlated_k.py','package/src/framework/physics/radiation_grid.py',
        'package/src/framework/physics/co2_12band_tables.npz']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='dry default-12-band optical aggregation with prescribed particle cross sections; CIA disabled as in the default reference')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
