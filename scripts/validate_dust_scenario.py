"""Compare dust-map sampling with unchanged FMS bilinear weight generation."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import numpy as np
import xarray as xr
from src.framework.gcm.coordinates import coordinate_system
from src.celestials.planets.mars.dust_scenario import (_bilinear_weights,bilinear_map,
    load_scenario_on_grid,INPUT_BLOBS)
from native_reference import compile_routines

FMS_BLOB='89f1a0b2efb58e77bf47cbc121ae2d382ba32e13'


def compile_reference(source,destination,compiler):
    header="""#define FMS_HI_KIND_ 8
#define HI_KIND_TYPE_ data
module reference_mod
use iso_c_binding
implicit none
integer,parameter :: FATAL=1,BILINEAR=1
real*8,parameter :: pi=acos(-1.d0),epsln=1.d-10
type weights_type
real*8,allocatable :: wti(:,:,:),wtj(:,:,:)
logical :: is_allocated=.false.
end type
type horiz_interp_type
integer :: nlon_src,nlat_src,nlon_dst,nlat_dst,interp_method
integer,allocatable :: i_lon(:,:,:),j_lat(:,:,:)
type(weights_type) :: data
end type
contains
"""
    adapter="""
subroutine weights(nx,ny,ox,oy,x,y,lon,lat,indices,w) bind(C)
integer(c_int),value :: nx,ny,ox,oy
real(c_double),intent(in) :: x(nx),y(ny),lon(ox,oy),lat(ox,oy)
integer(c_int),intent(out) :: indices(ox,oy,4)
real(c_double),intent(out) :: w(ox,oy,4)
type(horiz_interp_type) :: interp
call HORIZ_INTERP_BILINEAR_NEW_1D_(interp,x,y,lon,lat,0,.true.)
indices(:,:,1:2)=interp%i_lon-1
indices(:,:,3:4)=interp%j_lat-1
w(:,:,1:2)=interp%data%wti
w(:,:,3:4)=interp%data%wtj
end subroutine
integer function nearest_index(value,axis)
real*8 :: value,axis(:)
nearest_index=minloc(abs(axis-value),dim=1)
end function
subroutine mpp_error(level,message)
integer level
character(*) message
stop 1
end subroutine
integer function stdout()
stdout=6
end function
end module
"""
    library,provenance=compile_routines(source,FMS_BLOB,['HORIZ_INTERP_BILINEAR_NEW_1D_'],
        header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    indices=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.weights.argtypes=[ctypes.c_int]*4+[array]*4+[indices,array]
    native.weights.restype=None
    provenance['adaptation']='unchanged routine body, intended precision macros, module/type declarations and C adapter; monotone nearest_index uses minloc; unused diagnostics stubbed'
    return native,provenance


def validate(source,fms,compiler):
    cases=[];path=source/'data/DustScenario_Background.nc'
    with xr.open_dataset(path,decode_times=False) as ds, tempfile.TemporaryDirectory(prefix='dust-map-native-') as tmp:
        native,provenance=compile_reference(fms,Path(tmp),compiler)
        xb=np.asarray(ds.lonb,dtype=float);yb=np.asarray(ds.latb,dtype=float)
        x=np.deg2rad(.5*(xb[:-1]+xb[1:]));y=np.deg2rad(.5*(yb[:-1]+yb[1:]))
        grids=[(t,coordinate_system(t,4).horizontal) for t in ('T21','T42','T85','T170')]
        grids.append(('seam-poles',SimpleNamespace(longitudes=np.array([-.1,0.,np.pi,2.*np.pi,2.*np.pi+.1]),
            latitudes=np.array([-np.pi/2.,-1.,1.,np.pi/2.]))))
        for name,grid in grids:
            nx,ny=len(grid.longitudes),len(grid.latitudes)
            lon=np.asfortranarray(np.broadcast_to(np.asarray(grid.longitudes)[:,None],(nx,ny)))
            lat=np.asfortranarray(np.broadcast_to(np.asarray(grid.latitudes)[None,:],(nx,ny)))
            indices=np.empty((nx,ny,4),dtype=np.int32,order='F');weights=np.empty((nx,ny,4),order='F')
            native.weights(len(x),len(y),nx,ny,x,y,lon,lat,indices,weights)
            w,e,s,n,ww,ws=_bilinear_weights(grid,xb,yb)
            # At an exact endpoint equivalent neighboring pairs may use zero weight.
            for variable in ('taufill','zmax'):
                values=np.asarray(ds[variable].isel(time=[0,123,667]),dtype=float)
                i,j,k,l=(indices[...,a] for a in range(4))
                a,b,c,d=(weights[...,a] for a in range(4))
                expected=values[...,k,i]*a*c+values[...,k,j]*b*c+values[...,l,j]*b*d+values[...,l,i]*a*d
                actual=bilinear_map(grid,values,xb,yb)
                np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-13)
                cases.append(dict(grid=name,field=variable,max_error=float(np.max(np.abs(actual-expected)))))
            if name!='seam-poles':
                staged=load_scenario_on_grid(grid,path)
                np.testing.assert_allclose(staged['opacity'][[0,123,667]],
                    bilinear_map(grid,np.asarray(ds.taufill.isel(time=[0,123,667])),xb,yb)/2.75,rtol=3e-15)
                assert np.isfinite(staged['opacity']).all() and np.isfinite(staged['top_height_km']).all()
    verified_inputs={}
    grid=coordinate_system('T21',4).horizontal
    for filename in INPUT_BLOBS:
        staged=load_scenario_on_grid(grid,source/'data'/filename)
        assert staged['seasons'][0]==0. and staged['seasons'][-1]==360.
        if staged['periodic_endpoint_appended']:
            np.testing.assert_array_equal(staged['opacity'][-1],staged['opacity'][0])
            np.testing.assert_array_equal(staged['top_height_km'][-1],staged['top_height_km'][0])
        verified_inputs[filename]=dict(source_git_blob=staged['source_git_blob'],
            seasonal_samples=len(staged['seasons']),periodic_endpoint_appended=staged['periodic_endpoint_appended'],min_opacity=float(staged['opacity'].min()),
            min_top_height_km=float(staged['top_height_km'].min()))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,input_blobs=INPUT_BLOBS,verified_inputs=verified_inputs,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_dust_scenario.py','package/src/celestials/planets/mars/dust_scenario.py')})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--fms-bilinear-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,args.fms_bilinear_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} dust-map comparisons passed")


if __name__=='__main__': main()
