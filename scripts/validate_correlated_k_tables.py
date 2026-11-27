"""Check radiation pressure staging and interpolation against native Ames."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.correlated_k import load_ames_co2_tables,_interp2_log_table,_interp_linear_k
from native_reference import compile_routines,extract_routines


RADIATION_BLOB="40d8879c3735bbec6cabc63817cad5220f281532"


def validate(source,compiler):
    data=load_ames_co2_tables()
    for kind,name in (("ir","CO2H2O_IR_12_95_INTEL"),("sw","CO2H2O_V_12_95_INTEL")):
        if hashlib.sha256((source/"data"/name).read_bytes()).hexdigest()!=str(data[f"source_{kind}_sha256"]):
            raise ValueError("reference radiation data differs from the staged input")
    path=str((source/"data").resolve()).replace("'","''")+"/"
    header="""module reference_mod
use iso_c_binding
implicit none
integer, parameter :: l_ntref=7,l_npref=11,l_refh2o=10,l_nspecti=5,l_nspectv=7,l_ngauss=17,l_pint=51
logical :: use_boxinterp12=.true.
logical :: ames_15band=.false.,use_extended_cor_ks=.false.
real*8, parameter :: pi=acos(-1.d0)
real*8 :: wnoi(5),dwni(5),wavei(5),planckir(5,8501)
real*8 :: co2i(7,51,10,5,17),co2v(7,51,10,7,17),fzeroi(5),fzerov(7),pfgasref(51)
real*8 :: wrefh2o(10)=[1.d-7,1.d-6,1.d-5,1.d-4,1.d-3,.01d0,.1d0,.2d0,.3d0,.4d0]
character(len=1024) :: rtdata_path='"""+path+"""'
contains
"""
    adapter="""
subroutine ames_tables(ir,sw) bind(C)
real(c_double), intent(out) :: ir(7,51,10,5,16),sw(7,51,10,7,16)
real*8 :: pg(11)
integer :: n
do n=1,11
    pg(n)=10.d0**(n-7)
enddo
call laginterp(pg,pfgasref)
ir=co2i(:,:,:,:,1:16)
sw=co2v(:,:,:,:,1:16)
end subroutine
subroutine ames_planck(values) bind(C)
real(c_double), intent(out) :: values(5,8501)
call setspi()
values=planckir
end subroutine
subroutine ames_lookup(t,p,q,ir,sw) bind(C)
real(c_double), value :: t,p,q
real(c_double), intent(out) :: ir(5,16),sw(7,16)
integer :: mt(2),mp(2),mw,nw,ng
real*8 :: tref(7),lp,ans
logical :: pinter,tinter
tref=[50.d0,100.d0,150.d0,200.d0,250.d0,300.d0,350.d0]
lp=log10(p)
call tpindex15(lp,t,q,pfgasref,tref,mt,mp,mw,pinter,tinter)
do ng=1,16
    do nw=1,5
        call boxinterp(co2i(mt(1),mp(1),mw,nw,ng),co2i(mt(2),mp(1),mw,nw,ng), &
            co2i(mt(1),mp(2),mw,nw,ng),co2i(mt(2),mp(2),mw,nw,ng), &
            tref(mt(1)),tref(mt(2)),pfgasref(mp(1)),pfgasref(mp(2)),t,lp,tinter,pinter,ans)
        ir(nw,ng)=ans
    enddo
    do nw=1,7
        call boxinterp(co2v(mt(1),mp(1),mw,nw,ng),co2v(mt(2),mp(1),mw,nw,ng), &
            co2v(mt(1),mp(2),mw,nw,ng),co2v(mt(2),mp(2),mw,nw,ng), &
            tref(mt(1)),tref(mt(2)),pfgasref(mp(1)),pfgasref(mp(2)),t,lp,tinter,pinter,ans)
        sw(nw,ng)=ans
    enddo
enddo
end subroutine
end module
"""
    routines,_=extract_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,['optcv','optci'])
    blocks=[]
    for routine in routines:
        start=routine.index('kcoef(1) =')
        blocks.append(routine[start:routine.index('            taugas',start)])
    linear_adapter="""
subroutine ames_linear_tables(ir,sw) bind(C)
real(c_double),intent(out) :: ir(7,51,10,5,16),sw(7,51,10,7,16)
use_boxinterp12=.false.
call ames_tables(ir,sw)
end subroutine
subroutine ames_linear_lookup(t,p,q,ir,sw) bind(C)
real(c_double),value :: t,p,q
real(c_double),intent(out) :: ir(5,16),sw(7,16)
integer :: k,nw,ng,mt(1),mp(1),nh2o(1)
real*8 :: tref(7),lcoef(4),lkcoef(1,4),wratio(1),kcoef(4),ans
tref=[50.d0,100.d0,150.d0,200.d0,250.d0,300.d0,350.d0]
k=1
call tpindex(p,t,q,pfgasref,tref,lcoef,mt(k),mp(k),nh2o(k),wratio(k))
lkcoef(k,:)=lcoef
"""
    for kind,bands,block in zip(('sw','ir'),(7,5),blocks,strict=True):
        linear_adapter+=f'\ndo ng=1,16\ndo nw=1,{bands}\n'+block+f'\n{kind}(nw,ng)=ans\nenddo\nenddo\n'
    linear_adapter+='end subroutine\nend module\n'
    adapter=adapter.replace('end module',linear_adapter)
    jax.config.update("jax_enable_x64",True)
    with tempfile.TemporaryDirectory(prefix="ames-radiation-") as temporary:
        library,provenance=compile_routines(source/"atmos_param_mars/Rad/rtmod_mgcm.F90",
            RADIATION_BLOB,["laginterp","lagrange","tpindex","tpindex15","boxinterp","setspi"],
            header,adapter,Path(temporary),compiler)
        native=ctypes.CDLL(str(library))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags="F_CONTIGUOUS")
        native.ames_tables.argtypes=[array,array];native.ames_tables.restype=None
        native.ames_lookup.argtypes=[ctypes.c_double]*3+[array,array];native.ames_lookup.restype=None
        ir=np.empty((7,51,10,5,16),order="F")
        sw=np.empty((7,51,10,7,16),order="F")
        native.ames_tables(ir,sw)
        native.ames_planck.argtypes=[array];native.ames_planck.restype=None
        planck=np.empty((5,8501),order='F')
        native.ames_planck(planck)
        np.testing.assert_allclose(data['ames_planck_intensity_ir'],planck.T,rtol=2e-12,atol=2e-15)
        planck_error=float(np.max(np.abs(data['ames_planck_intensity_ir']-planck.T)))
        table_errors={}
        for kind,actual in (("ir",ir),("sw",sw)):
            staged=data[f"log10_k_{kind}_mixtures_ames"]
            np.testing.assert_allclose(staged,actual,rtol=2e-12,atol=2e-12)
            table_errors[kind]=float(np.max(np.abs(staged-actual)))
        rng=np.random.default_rng(20261115)
        max_lookup_errors=dict(ir=0.,sw=0.)
        for temperature,pressure in zip(rng.uniform(40.,400.,100),10.**rng.uniform(-7.,5.,100),strict=True):
            ir_out=np.empty((5,16),order="F");sw_out=np.empty((7,16),order="F")
            native.ames_lookup(temperature,pressure,1e-7,ir_out,sw_out)
            for kind,reference in (("ir",ir_out),("sw",sw_out)):
                actual=np.asarray(_interp2_log_table(data[f"log10_k_{kind}_ames"],
                    data['temperature_k'],data['pressure_mbar_ames'],temperature,pressure))
                np.testing.assert_allclose(actual,reference,rtol=2e-12,atol=2e-12)
                max_lookup_errors[kind]=max(max_lookup_errors[kind],float(np.max(np.abs(actual-reference))))
        provenance['linear_interpolation_block_sha256']={kind:hashlib.sha256(block.encode()).hexdigest()
            for kind,block in zip(('sw','ir'),blocks,strict=True)}
        native.ames_linear_tables.argtypes=[array]*2;native.ames_linear_tables.restype=None
        native.ames_linear_lookup.argtypes=[ctypes.c_double]*3+[array]*2
        native.ames_linear_lookup.restype=None
        native.ames_linear_tables(ir,sw)
        for kind,reference in (('ir',ir),('sw',sw)):
            np.testing.assert_allclose(10.**data[f'log10_k_{kind}_mixtures_ames'],reference,
                rtol=3e-13,atol=1e-300,equal_nan=False)
        linear_errors=dict(ir=0.,sw=0.);linear_cases=0
        for temperature in (40.,50.,200.,203.03,350.,380.):
            for pressure in (1e-9,1e-6,.3,3.,613.,1e4,2e4):
                for water in (0.,1e-7,1e-5,.003,.25):
                    native.ames_linear_lookup(temperature,pressure,water,ir_out,sw_out)
                    for kind,reference in (('ir',ir_out),('sw',sw_out)):
                        actual=np.asarray(_interp_linear_k(data[f'log10_k_{kind}_mixtures_ames'],
                            data['temperature_k'],data['pressure_mbar_ames'],temperature,pressure,
                            data['water_mass_fractions'],water))
                        # Bound cancellation by the absolute weighted corner sum, not the near-zero result.
                        table=10.**data[f'log10_k_{kind}_mixtures_ames']
                        ta=data['temperature_k'];pa=np.log10(data['pressure_mbar_ames'])
                        it=np.clip(np.searchsorted(ta,temperature,side='left')-1,0,len(ta)-2)
                        ip=np.clip(np.searchsorted(pa,np.log10(pressure),side='left')-1,0,len(pa)-3)
                        ft=(temperature-ta[it])/(ta[it+1]-ta[it])
                        fp=(np.log10(pressure)-pa[ip])/(pa[ip+1]-pa[ip])
                        scale=sum(abs(w)*np.max(table[t,p],axis=0) for w,t,p in
                            (((1-fp)*(1-ft),it,ip),(fp*(1-ft),it,ip+1),
                             (fp*ft,it+1,ip+1),((1-fp)*ft,it+1,ip)))
                        bound=3e-12*np.abs(reference)+64*np.finfo(np.float64).eps*scale
                        assert np.isfinite(actual).all() and np.all(np.abs(actual-reference)<=bound), \
                            f'{kind} T={temperature} p={pressure} q={water}: interpolation roundoff bound exceeded'
                        linear_errors[kind]=max(linear_errors[kind],float(np.max(np.abs(actual-reference))))
                    linear_cases+=1
    return dict(status="pass",reference=provenance,max_log10_table_errors=table_errors,
        max_log10_lookup_errors=max_lookup_errors,lookup_cases=100,
        linear_lookup_cases=linear_cases,max_linear_k_errors=linear_errors,
        max_planck_intensity_error=planck_error,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage_sha256=hashlib.sha256(Path(__file__).with_name('stage_co2_radiation_tables.py').read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            'package/src/framework/physics/correlated_k.py').read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            'package/src/framework/physics/co2_12band_tables.npz').read_bytes()).hexdigest(),
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits="Both 12-band interpolation paths, including unchanged linear-k blocks, T/log-pressure extrapolation and mixed water below the source's out-of-bounds last-bin read. Planck interpolation is continuous for derivatives; coupled radiation and cloud feedback need separate validation")


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
