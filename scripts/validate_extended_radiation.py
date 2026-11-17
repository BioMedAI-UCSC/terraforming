"""Check 15-band gas lookup, native spectral integration and cloud-radiation derivatives."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.correlated_k import load_radiation_tables,correlated_k_optical_depths
from src.framework.physics.spectral_flux import visible_spectrum,infrared_spectrum
from native_reference import compile_routines
from validate_spectral_flux import compile_reference as compile_spectra

RADIATION_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_lookup(source,destination,compiler,mode):
    nt=13 if mode=='15band' else 16
    header=f"""module reference_mod
use iso_c_binding
implicit none
integer,parameter :: l_ntref={nt},l_npref=25,l_pint=51,l_refh2o=10
logical,parameter :: use_boxinterp12=.false.
real*8 :: wrefh2o(10)=[1.d-7,1.d-6,1.d-5,1.d-4,1.d-3,.01d0,.1d0,.2d0,.3d0,.4d0]
contains
"""
    adapter="""
subroutine lookup(nc,nb,ng,p,t,water,pa,ta,table,answer) bind(C)
integer(c_int),value :: nc,nb,ng
real(c_double),intent(in) :: p(nc),t(nc),water(nc),pa(25),ta(l_ntref),table(l_ntref,25,10,nb,ng)
real(c_double),intent(out) :: answer(nc,nb,ng)
real*8 :: pref(25),ans
integer :: i,b,g,mt(2),mp(2),mw
logical :: pi,ti
pref=log10(pa)
do i=1,nc
 call tpindex15(log10(p(i)),t(i),water(i),pref,ta,mt,mp,mw,pi,ti)
 do g=1,ng
  do b=1,nb
   call boxinterp(table(mt(1),mp(1),mw,b,g),table(mt(2),mp(1),mw,b,g), &
    table(mt(1),mp(2),mw,b,g),table(mt(2),mp(2),mw,b,g),ta(mt(1)),ta(mt(2)), &
    pref(mp(1)),pref(mp(2)),t(i),log10(p(i)),ti,pi,ans)
   answer(i,b,g)=10.d0**ans
  enddo
 enddo
enddo
end subroutine
end module
"""
    lib,provenance=compile_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,
        ['tpindex15','boxinterp'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(lib))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.lookup.argtypes=[ctypes.c_int]*3+[array]*7;native.lookup.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    cases=[];references={};rng=np.random.default_rng(104)
    with tempfile.TemporaryDirectory(prefix='extended-radiation-') as temporary:
        for mode in ('15band','extended'):
            data=load_radiation_tables(mode);ng=len(data['gauss_weights'])
            root=Path(temporary)/mode;root.mkdir();(root/'lookup').mkdir();(root/'spectra').mkdir()
            lookup,lookup_prov=compile_lookup(source,root/'lookup',compiler,mode)
            spectra,spectral_prov=compile_spectra(source,root/'spectra',compiler,mode=mode)
            references[mode]=dict(lookup=lookup_prov,spectra=spectral_prov)
            temperature=data['temperature_k'];pressure=data['pressure_mbar_ames']
            nodes=np.meshgrid(temperature,pressure,data['water_mass_fractions'],indexing='ij')
            centers=np.meshgrid(.5*(temperature[:-1]+temperature[1:]),
                np.sqrt(pressure[:-1]*pressure[1:]),data['water_mass_fractions'],indexing='ij')
            samples=[('range-sweep',np.linspace(30.,950.,100),np.geomspace(1e-10,1e6,100),
                np.resize(np.array([0.,1e-7,3e-6,.0001,.005,.05,.1,.25,.4,.8]),100))]
            samples.extend((label,*(field.ravel() for field in fields)) for label,fields in
                (('all-table-nodes',nodes),('all-cell-centers',centers)))
            for label,t,p,water in samples:
                n=len(t)
                depths=correlated_k_optical_depths(t,p*100.,np.full(n,100.),mode=mode,water_mass_fraction=water)
                for kind,bands in (('sw',7),('ir',8)):
                    expected=f(np.zeros((n,bands,ng)))
                    lookup.lookup(n,bands,ng,f(p),f(t),f(water),f(pressure),f(temperature),
                        f(data[f'log10_k_{kind}_mixtures_ames']),expected)
                    actual=np.moveaxis(np.asarray(depths[0 if kind=='sw' else 1])[:,:-1],(0,1),(1,2))/3.51e22
                    np.testing.assert_array_equal(np.isnan(actual),np.isnan(expected))
                    np.testing.assert_array_equal(np.isposinf(actual),np.isposinf(expected))
                    np.testing.assert_array_equal(np.isneginf(actual),np.isneginf(expected))
                    np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=1e-200)
                    finite=np.isfinite(expected)
                    cases.append(dict(mode=mode,kind=kind,case=label,lookup_points=n,
                        max_finite_coefficient_error=float(np.max(np.abs(actual[finite]-expected[finite]))),
                        matching_nan_coefficients=int(np.sum(np.isnan(expected))),
                        matching_positive_infinite_coefficients=int(np.sum(np.isposinf(expected))),
                        matching_nonfinite_coefficients=int(np.sum(~finite))))
            for layers in (2,5,13):
                for label,loading in (('merged',0.),('mixed',.005),('absorbing',1.)):
                    for kind,bands in (('visible',7),('infrared',8)):
                        shape=(layers,bands,ng+1)
                        tau=f(rng.uniform(.01,.3,shape));w=f(rng.uniform(.05,.8,shape));g=f(rng.uniform(.05,.8,shape))
                        cumulative=np.zeros((2*layers+1,bands,ng+1))
                        cumulative[1::2]=np.concatenate([np.zeros_like(tau[:1]),np.cumsum(tau[:-1],axis=0)])
                        cumulative[2:-1:2]=np.cumsum(tau[:-1],axis=0)-tau[:-1]*.5
                        cumulative[-1]=tau.sum(axis=0);cumulative=f(cumulative)
                        gas=f(rng.uniform(0.,2.,(bands,ng))*loading)
                        expected=[np.zeros(layers),np.zeros(layers),np.zeros(1)]
                        if kind=='visible':
                            solar=f(data['solar_weights']*590.)
                            spectra.ames_sw(layers,.6,.2,tau,w,g,cumulative,gas,solar,
                                f(data['clear_fraction_sw']),f(data['gauss_weights']),*expected)
                            actual=visible_spectrum(tau,w,g,cumulative,gas,.6,solar,.2,mode=mode)
                        else:
                            pressure=f(np.geomspace(.001,6.1,2*layers+1));temperature=f(150.+5.*np.arange(2*layers+1))
                            spectra.ames_lw(layers,.1,0,tau,w,g,cumulative,gas,pressure,temperature,
                                f(data['ames_planck_intensity_ir'].T),f(np.diff(data['ir_bounds_cm1'])),
                                f(data['clear_fraction_ir']),f(data['gauss_weights']),*expected)
                            actual=infrared_spectrum(tau,w,g,cumulative,gas,pressure,temperature,.1,mode=mode)
                        errors=[]
                        for a,b in zip(actual,(*expected[:2],expected[2][0]),strict=True):
                            np.testing.assert_allclose(a,b,rtol=2e-10,atol=3e-10)
                            errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                        cases.append(dict(mode=mode,kind=kind,layers=layers,case=label,max_flux_errors_w_m2=errors))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_extended_radiation.py','scripts/stage_extended_radiation_tables.py',
        'scripts/validate_spectral_flux.py','package/src/framework/physics/correlated_k.py',
        'package/src/framework/physics/spectral_flux.py','package/src/framework/physics/scattering_transfer.py',
        'package/src/framework/physics/co2_15band_tables.npz','package/src/framework/physics/co2_extended_tables.npz']
    return dict(status='pass',reference=references,cases=cases,
        gas_inputs={mode:json.loads(str(load_radiation_tables(mode)['source_provenance_json']))
            for mode in ('15band','extended')},
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Gas lookup and spectral integration. Positive infinite coefficients in released extended IR tables and resulting invalid lookups are reported, not silently replaced. CIA and fully composed particle feedback require their own checks')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
