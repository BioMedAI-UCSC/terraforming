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
from src.framework.physics.ames_radiation import load_ames_co2_tables,_interp2_log_table
from ames_reference import compile_routines


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
    jax.config.update("jax_enable_x64",True)
    with tempfile.TemporaryDirectory(prefix="ames-radiation-") as temporary:
        library,provenance=compile_routines(source/"atmos_param_mars/Rad/rtmod_mgcm.F90",
            RADIATION_BLOB,["laginterp","lagrange","tpindex15","boxinterp","setspi"],
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
    return dict(status="pass",reference=provenance,max_log10_table_errors=table_errors,
        max_log10_lookup_errors=max_lookup_errors,lookup_cases=100,
        max_planck_intensity_error=planck_error,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        stage_sha256=hashlib.sha256(Path(__file__).with_name('stage_ames_co2_radiation.py').read_bytes()).hexdigest(),
        kernel_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            'package/src/framework/physics/ames_radiation.py').read_bytes()).hexdigest(),
        asset_sha256=hashlib.sha256(Path(__file__).parents[1].joinpath(
            'package/src/framework/physics/ames_co2_12band.npz').read_bytes()).hexdigest(),
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits="12-band gas tables, dry lookup and Planck nodes only; continuous Planck interpolation is a differentiability adaptation; coupled radiation and cloud feedback need separate validation")


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
