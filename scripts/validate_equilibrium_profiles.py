"""Compare NetCDF equilibrium readers and composed damping with native code."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import numpy as np
import xarray as xr
from src.framework.gcm._dinosaur import jax
from src.framework.physics.thermal_relaxation import load_equilibrium,fixed_tendency,height_tendency
from native_reference import extract_routines
from validate_dust_scenario import compile_reference as compile_weights
from validate_thermal_relaxation import compile_reference as compile_damping,PHYSICS_BLOB


def compile_reader(source,bilinear,destination,compiler,spectral):
    weights_dir=destination/'weights';weights_dir.mkdir()
    _,weights_provenance=compile_weights(bilinear,weights_dir,compiler)
    prefix=(weights_dir/'reference.f90').read_text().replace('module reference_mod','module bilinear_reference_mod')
    routines,blob=extract_routines(source/'atmos_param_mars/mars_physics.F90',PHYSICS_BLOB,['read_teq','read_teq_z'])
    header='''
module horiz_interp_mod
use bilinear_reference_mod,only: weights
implicit none
contains
subroutine horiz_interp(data,xb,yb,lon,lat,result,interp_method)
real,intent(in) :: data(:,:),xb(:),yb(:),lon(:,:),lat(:,:)
real,intent(out) :: result(:,:)
character(*),intent(in) :: interp_method
real :: x(size(xb)-1),y(size(yb)-1),w(size(lon,1),size(lon,2),4)
integer :: indices(size(lon,1),size(lon,2),4),i,j,a,b,c,d
if(interp_method/='bilinear') stop 1
x=.5*(xb(1:size(x))+xb(2:size(x)+1))
y=.5*(yb(1:size(y))+yb(2:size(y)+1))
call weights(size(x),size(y),size(lon,1),size(lon,2),x,y,lon,lat,indices,w)
indices=indices+1
do j=1,size(lon,2)
 do i=1,size(lon,1)
  a=indices(i,j,1);b=indices(i,j,2);c=indices(i,j,3);d=indices(i,j,4)
  result(i,j)=data(a,c)*w(i,j,1)*w(i,j,3)+data(b,c)*w(i,j,2)*w(i,j,3) &
             +data(b,d)*w(i,j,2)*w(i,j,4)+data(a,d)*w(i,j,1)*w(i,j,4)
 enddo
enddo
end subroutine
end module
module reference_mod
use iso_c_binding
use netcdf
implicit none
real,parameter :: RADIAN=180./acos(-1.)
type FmsNetcdfFile_t
integer :: id
end type
interface read_data
module procedure read_vector,read_volume
end interface
contains
subroutine check(status)
integer :: status
if(status/=nf90_noerr) then
 print *,nf90_strerror(status)
 stop 1
endif
end subroutine
logical function open_file(fileobj,filename,mode)
type(FmsNetcdfFile_t),intent(out) :: fileobj
character(*),intent(in) :: filename,mode
if(mode/='read') stop 1
call check(nf90_open(filename,nf90_nowrite,fileobj%id))
open_file=.true.
end function
subroutine get_variable_size(fileobj,name,sizes)
type(FmsNetcdfFile_t),intent(in) :: fileobj
character(*),intent(in) :: name
integer,intent(out) :: sizes(:)
integer :: id,ndims,dimensions(nf90_max_var_dims),k
call check(nf90_inq_varid(fileobj%id,name,id))
call check(nf90_inquire_variable(fileobj%id,id,ndims=ndims,dimids=dimensions))
sizes=1
do k=1,min(ndims,size(sizes))
 call check(nf90_inquire_dimension(fileobj%id,dimensions(k),len=sizes(k)))
enddo
end subroutine
subroutine read_vector(fileobj,name,value)
type(FmsNetcdfFile_t),intent(in) :: fileobj
character(*),intent(in) :: name
real,intent(out) :: value(:)
integer :: id
call check(nf90_inq_varid(fileobj%id,name,id))
call check(nf90_get_var(fileobj%id,id,value))
end subroutine
subroutine read_volume(fileobj,name,value)
type(FmsNetcdfFile_t),intent(in) :: fileobj
character(*),intent(in) :: name
real,intent(out) :: value(:,:,:)
integer :: id
call check(nf90_inq_varid(fileobj%id,name,id))
call check(nf90_get_var(fileobj%id,id,value))
end subroutine
subroutine close_file(fileobj)
type(FmsNetcdfFile_t),intent(in) :: fileobj
call check(nf90_close(fileobj%id))
end subroutine
'''
    adapter='''
subroutine profile(length,path,nx,ny,n,lon,lat,half,height,temperature,z) bind(C)
integer(c_int),value :: length,nx,ny,n,height
character(c_char),intent(in) :: path(length)
real(c_double),intent(in) :: lon(nx,ny),lat(nx,ny),half(n+1)
real(c_double),intent(out) :: temperature(nx,ny,n),z(nx,ny,n)
character(len=length) :: filename
integer :: i
do i=1,length
 filename(i:i)=path(i)
enddo
z=0.
if(height/=0) then
 call read_teq_z(filename,lon,lat,z,temperature)
else
 call read_teq(filename,lon,lat,half,temperature)
endif
end subroutine
end module
'''
    code=destination/'reader.f90'
    code.write_text(('#define SPEC_CORE\n' if spectral else '')+prefix+header+'\n'.join(routines)+adapter)
    flags=['-O3','-cpp','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
    includes=shlex.split(subprocess.check_output(['nf-config','--fflags'],text=True))
    libraries=shlex.split(subprocess.check_output(['nf-config','--flibs'],text=True))
    libraries+=shlex.split(subprocess.check_output(['nc-config','--libs'],text=True))
    library=destination/'reader.so'
    built=subprocess.run([compiler,*flags,*includes,str(code),*libraries,'-o',str(library)],
        cwd=destination,capture_output=True,text=True)
    if built.returncode: raise RuntimeError(built.stderr)
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.profile.argtypes=[ctypes.c_int,ctypes.c_char_p]+[ctypes.c_int]*3+[array]*3+[ctypes.c_int]+[array]*2
    native.profile.restype=None
    return native,dict(source_git_blob=blob,compiled_source_sha256=hashlib.sha256(code.read_bytes()).hexdigest(),
        routine_sha256={name:hashlib.sha256(body.encode()).hexdigest()
            for name,body in zip(('read_teq','read_teq_z'),routines,strict=True)},
        weights=weights_provenance,compiler_flags=flags,spectral_core=spectral,
        netcdf_version=subprocess.check_output(['nf-config','--version'],text=True).strip(),
        adaptation='Unchanged readers use netCDF-Fortran I/O adapters and unchanged FMS bilinear weights; a weighted sum applies those weights. FMS parallel I/O is not linked.')


def validate(source,fms,bilinear,compiler):
    jax.config.update('jax_enable_x64',True)
    f=lambda value:np.asfortranarray(value,dtype=np.float64)
    cases=[];references=[]
    with tempfile.TemporaryDirectory(prefix='equilibrium-profile-native-') as temporary:
        root=Path(temporary);damping_dir=root/'damping';damping_dir.mkdir()
        damping,damping_provenance=compile_damping(source,fms,damping_dir,compiler)
        for spectral in (False,True):
            build=root/str(spectral);build.mkdir()
            reader,provenance=compile_reader(source,bilinear,build,compiler,spectral)
            references.append(provenance)
            for source_levels,levels in ((4,4),(4,7),(7,4)):
                xb=np.linspace(0.,360.,9);yb=np.linspace(-90.,90.,5)
                x=.5*(xb[1:]+xb[:-1]);y=.5*(yb[1:]+yb[:-1])
                values=170.+15.*np.arange(source_levels)[:,None,None]+.02*y[None,:,None]+3.*np.sin(np.deg2rad(x))[None,None,:]
                heights=np.broadcast_to(np.linspace(1e5,-1000.,source_levels)[:,None,None],values.shape).copy()
                heights+=10.*np.cos(np.deg2rad(x))[None,None,:]
                for time_axis in (False,True):
                    dims=('level','lat','lon');fields={'teq':(dims,values),'zzf':(dims,heights),
                        'pfull':('level',np.geomspace(.01,10.,source_levels)),
                        'lonb':('lonb',xb),'latb':('latb',yb)}
                    data=xr.Dataset(fields,coords=dict(lon=x,lat=y))
                    if time_axis: data=data.expand_dims(time=[0.])
                    path=root/'profile.nc';data.to_netcdf(path)
                    grid=SimpleNamespace(longitudes=np.deg2rad(np.array([-.1,0.,22.5,80.,180.,359.9,360.1])),
                        latitudes=np.deg2rad(np.array([-90.,-60.,-5.,40.,90.])))
                    nx,ny=len(grid.longitudes),len(grid.latitudes)
                    lon=f(np.broadcast_to(grid.longitudes[:,None],(nx,ny)))
                    lat=f(np.broadcast_to(grid.latitudes[None,:],(nx,ny)))
                    for top in (0.,.01):
                        half=f(np.r_[top,np.geomspace(.1,1e4,levels)])
                        for height in (False,True) if source_levels==levels else (False,):
                            expected=f(np.empty((nx,ny,levels)));z=f(np.empty_like(expected))
                            encoded=str(path).encode()
                            reader.profile(len(encoded),encoded,nx,ny,levels,lon,lat,half,int(height),expected,z)
                            actual=load_equilibrium(grid,path,half,height=height,spectral_core=spectral)
                            target,source_height=actual if height else (actual,np.zeros_like(actual))
                            reference=expected.transpose(2,0,1);reference_z=z.transpose(2,0,1)
                            np.testing.assert_allclose(target,reference,rtol=3e-13,atol=1e-12)
                            np.testing.assert_allclose(source_height,reference_z,rtol=3e-13,atol=1e-10)
                            temperature=target+np.linspace(-7.,5.,levels)[:,None,None]
                            model_height=.9*source_height+.1*np.mean(source_height,axis=0)
                            layout=lambda value:f(np.asarray(value).reshape(levels,-1).T[:,None,:])
                            out=[f(np.empty((nx*ny,1,levels))) for _ in range(2)]
                            damping.thermal(levels,nx*ny,int(height),-4.,-1.,f(lat.reshape(-1,1)),
                                f(np.full((nx*ny,1),half[-1])),layout(np.broadcast_to(.5*(half[1:]+half[:-1])[:,None,None],target.shape)),
                                layout(temperature),layout(model_height),layout(reference_z),layout(reference),layout(np.ones_like(target)),*out)
                            tendency=(height_tendency(temperature,model_height,source_height,target)[0]
                                if height else fixed_tendency(temperature,target))
                            np.testing.assert_allclose(tendency,np.asarray(out[0])[:,0,:].T.reshape(target.shape),rtol=5e-13,atol=1e-12)
                            cases.append(dict(source_levels=source_levels,levels=levels,time_axis=time_axis,
                                spectral_core=spectral,top_pressure_pa=top,height=height,
                                max_temperature_error_k=float(np.max(abs(target-reference)))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_equilibrium_profiles.py','scripts/validate_dust_scenario.py',
        'scripts/validate_thermal_relaxation.py','scripts/native_reference.py',
        'package/src/framework/physics/thermal_relaxation.py',
        'package/src/celestials/planets/mars/dust_scenario.py']
    return dict(status='pass',readers=references,damping=damping_provenance,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
        limits='Synthetic NetCDF profiles, native equal-count bypass, pressure interpolation, poles/seam and composed fixed/height damping. Actual netCDF-Fortran I/O replaces FMS parallel I/O. Invalid inputs are rejected separately by Python tests.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--fms-source',type=Path,required=True)
    parser.add_argument('--fms-bilinear-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None or any(shutil.which(name) is None for name in ('nf-config','nc-config')):
        parser.error('gfortran, netCDF-Fortran and netCDF-C are required')
    report=validate(args.ames_source,args.fms_source,args.fms_bilinear_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native equilibrium-profile and composed damping cases passed")
