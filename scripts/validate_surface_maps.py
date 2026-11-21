"""Check pinned TES/GRS input maps and conservative Gaussian-grid remapping."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
import xarray as xr
from src.framework.gcm._dinosaur import jax
from src.framework.gcm.coordinates import coordinate_system
from src.celestials.planets.mars.surface import reference_surface_fields_on_grid,conservative_surface_field
from src.framework.physics.soil_frost import subsurface_diffusivity,surface_thermal_inertia

INPUTS={
    'mars_thermal_inertia8_2011.nc':('thin','thermal_inertia','e91fc91677b866e9c3c9ddbe42e33204c96ae20f'),
    'mars_TES_albedo8_new.nc':('albedo','albedo','c6d3ae52404a89686809865a7621ed9b97f6337c'),
    'mars_sfc_emissivity8.nc':('emis','emissivity','8b92607e72bc87d594942c23cc2dd251aa097b94'),
    'GRS_interp16.nc':('grs_indices','grs_ice_latitude_deg','6420557ea160ebab630e3aa53e82f36f74f67c13'),
}



def validate_inertia_adjustments(source,compiler):
    from validate_soil_frost import SURFACE_BLOB
    raw=(source/'atmos_param_mars/mars_surface.F90').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==SURFACE_BLOB
    text=raw.decode();start=text.index('if( np_cap_ti > 0.0 )')
    end=text.index('\nendif',text.index('if( rescale_sh_ti > 0.0 )',start))+len('\nendif')
    block=text[start:end]
    code="""module reference_mod
use iso_c_binding
implicit none
contains
subroutine adjust(n,latitude,inertia,replacement,boundary,maximum,scaling,result) bind(C)
integer(c_int),value :: n
real(c_double),value :: replacement,boundary,maximum,scaling
real(c_double),intent(in) :: latitude(n),inertia(n)
real(c_double),intent(out) :: result(n)
real :: lat(n,1),thermal_inertia(n,1),np_cap_ti,np_cap_lat,np_cap_ti_max,rescale_sh_ti
real,parameter :: pi=acos(-1.d0)
logical,parameter :: mcpu0=.false.
lat(:,1)=latitude;thermal_inertia(:,1)=inertia
np_cap_ti=replacement;np_cap_lat=boundary;np_cap_ti_max=maximum;rescale_sh_ti=scaling
"""+block+"""
result=thermal_inertia(:,1)
end subroutine
end module
"""
    flags=['-O3','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
    cases=[]
    with tempfile.TemporaryDirectory(prefix='surface-inertia-native-') as temporary:
        directory=Path(temporary);(directory/'reference.f90').write_text(code)
        subprocess.run([compiler,*flags,'reference.f90','-o','reference.so'],cwd=directory,check=True)
        native=ctypes.CDLL(str(directory/'reference.so'))
        array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
        native.adjust.argtypes=[ctypes.c_int,array,array]+[ctypes.c_double]*4+[array]
        native.adjust.restype=None
        for boundary in (0.,30.,65.,90.):
            edges=np.array([-50.,30.,boundary])*np.pi/180.
            below=np.nextafter(edges,-np.inf);above=np.nextafter(edges,np.inf)
            # JAX CPU flushes subnormals; probe the closest normal latitude at zero.
            below=np.where(edges==0.,-np.finfo(float).tiny,below)
            above=np.where(edges==0.,np.finfo(float).tiny,above)
            latitude=np.unique(np.r_[np.linspace(-np.pi/2,np.pi/2,181),edges,below,above])
            inertia=np.linspace(100.,1000.,latitude.size)
            for replacement in (-900.,600.,1000.):
                for maximum in (700.,800.):
                    for scaling in (-100.,0.,.5,2.):
                        result=np.empty_like(inertia)
                        native.adjust(len(latitude),latitude,inertia,replacement,boundary,maximum,scaling,result)
                        options=dict(north_inertia=replacement,north_latitude_deg=boundary,
                            north_maximum=maximum,regional_scale=scaling)
                        actual=surface_thermal_inertia(latitude,inertia,**options,xp=np)
                        np.testing.assert_allclose(actual,result,rtol=2e-15,atol=2e-13,err_msg=str(options))
                        errors=[float(np.max(abs(actual-result)))]
                        # JIT uses the same supplied coordinates and source comparisons.
                        actual=np.asarray(jax.jit(lambda lat,ti:surface_thermal_inertia(lat,ti,**options))(latitude,inertia))
                        np.testing.assert_allclose(actual,result,rtol=2e-15,atol=2e-13,err_msg=str(options))
                        errors.append(float(np.max(abs(actual-result))))
                        cases.append(dict(latitude_count=len(latitude),**options,max_errors=errors))
    return dict(surface_git_blob=SURFACE_BLOB,block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),compiler_flags=flags,cases=cases)


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    adjustments=validate_inertia_adjustments(source,compiler)
    provenance={};cases=[]
    for filename,(_,_,expected) in INPUTS.items():
        raw=(source/'data'/filename).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=expected: raise ValueError(f'{filename} differs from pinned v3.2.1')
        provenance[filename]=dict(git_blob=blob,sha256=hashlib.sha256(raw).hexdigest())
    for truncation in ('T21','T42'):
        grid=coordinate_system(truncation,4).horizontal
        fields=reference_surface_fields_on_grid(grid,source/'data')
        quadrature=np.broadcast_to(np.asarray(grid.quadrature_weights),grid.nodal_shape)
        checks={}
        for filename,(variable,name,_) in INPUTS.items():
            with xr.open_dataset(source/'data'/filename) as opened:
                raw=np.asarray(opened[variable].transpose('lat','lon'),dtype=float)
                source_area=np.diff(np.sin(np.linspace(-np.pi*.5,np.pi*.5,raw.shape[0]+1)))
                expected=float(np.einsum('ij,i->',raw,source_area)/(2.*raw.shape[1]))
                mapped=conservative_surface_field(grid,opened[variable])
            mean=float(np.sum(mapped*quadrature)/quadrature.sum())
            np.testing.assert_allclose(mean,expected,rtol=3e-13,atol=3e-13)
            final=fields[name]
            north=np.asarray(grid.latitudes)[None,:]>np.deg2rad(65.)
            target=np.where(north,np.minimum(mapped,800.),mapped) if name=='thermal_inertia' else mapped
            np.testing.assert_allclose(final,target,rtol=2e-14,atol=2e-14)
            checks[name]=dict(source_area_mean=expected,remapped_area_mean=mean,
                final_min=float(final.min()),final_max=float(final.max()))
        diffusivity=subsurface_diffusivity(np.asarray(grid.latitudes)[None,:],
            fields['thermal_inertia'],fields['grs_ice_latitude_deg'])
        assert np.isfinite(diffusivity).all() and np.all(np.asarray(diffusivity)>0.)
        cases.append(dict(truncation=truncation,shape=list(grid.nodal_shape),fields=checks,
            emissivity_above_one_cells=int(np.sum(fields['emissivity']>1.)),
            soil_diffusivity_shape=list(diffusivity.shape)))
    repo=Path(__file__).parents[1]
    paths=['scripts/validate_surface_maps.py','package/src/celestials/planets/mars/surface.py',
        'package/src/framework/physics/soil_frost.py']
    return dict(status='pass',reference_commit='ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26',
        inputs=provenance,cases=cases,inertia_adjustments=adjustments,
        sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths},
        limits='Gaussian lat/lon conservative area remap; source emissivity above one is preserved; does not establish FV3/Dinosaur climate agreement')


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
    print(json.dumps(dict(status=report['status'],grids=len(report['cases']),inertia_cases=len(report['inertia_adjustments']['cases']))))
