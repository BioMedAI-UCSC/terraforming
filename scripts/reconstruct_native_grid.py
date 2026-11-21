"""Recover default C24/C48 corner metadata from the frozen FV3 grid routines.

The original files remain unchanged. Unsupported grid controls are rejected.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np

SOURCES={
    'model/fv_grid_utils.F90':('60f1bc780968fd9079a4a5cbc06372bc8af7ec17acd83845721845e395cb2cf5',
        ['gnomonic_ed','symm_ed','latlon2xyz2','latlon2xyz','mirror_latlon',
         'cart_to_latlon','vect_cross','great_circle_dist']),
    'tools/fv_grid_tools.F90':('1cccc0ba40afe086e6b244c22e2e3bb6d5eed5af9deaa7902b46d7f31bc619df',
        ['mirror_grid','rot_3d','spherical_to_cartesian','cartesian_to_spherical']),
    'model/fv_arrays.F90':('bac6ff3b90c4db609998e1758f4aa8b5e47b4a9cce6884df3f57dd965123d044',[]),
}
HEADER='''module reference_mod
use iso_c_binding
implicit none
integer,parameter :: R_GRID=8,f_p=8
real*8,parameter :: pi=acos(-1.d0),radius=3386000.d0,torad=pi/180.d0
contains
logical function is_master()
is_master=.false.
end function
'''
ADAPTER='''
subroutine grid(n,out) bind(C)
integer(c_int),value :: n
real(c_double),intent(out) :: out(n+1,n+1,2,6)
integer i,j,k
call gnomonic_ed(n,out(:,:,1,1),out(:,:,2,1))
call symm_ed(n,out(:,:,1,1),out(:,:,2,1))
out(:,:,1,1)=out(:,:,1,1)-pi
call mirror_grid(out,0,n+1,n+1,2,6)
do k=1,6
do j=1,n+1
do i=1,n+1
out(i,j,1,k)=out(i,j,1,k)-pi/18.d0
if(out(i,j,1,k)<0.) out(i,j,1,k)=out(i,j,1,k)+2.*pi
if(abs(out(i,j,1,1))<1.d-10) out(i,j,1,1)=0.d0
if(abs(out(i,j,2,1))<1.d-10) out(i,j,2,1)=0.d0
enddo
enddo
enddo
end subroutine
end module
'''


def grid_cells(namelist):
    text=re.sub(r'!.*','',namelist.read_text()).lower()
    values=lambda name:re.findall(r'\b'+name+r'\s*=\s*([^,\n/]+)',text)
    controls={'grid_type':0.,'shift_fac':18.,'do_schmidt':False,
        'do_cube_transform':False,'regional':False,'nested':False,'grid_file':'input/grid_spec.nc'}
    for name,expected in controls.items():
        for value in values(name):
            value=value.strip().strip("'\"")
            if isinstance(expected,bool):
                value=value.strip('.')
                if value not in ('true','t','false','f'): raise ValueError(f'invalid grid control: {name}')
                value=value in ('true','t')
            elif isinstance(expected,float): value=float(value.replace('d','e'))
            if value!=expected: raise ValueError(f'unsupported grid control: {name}')
    sizes=[[item.strip() for item in values(name)] for name in ('npx','npy')]
    if any(len(items)!=1 for items in sizes) or sizes[0]!=sizes[1] or int(sizes[0][0]) not in (25,49):
        raise ValueError('grid reconstruction requires square C24 or C48')
    return int(sizes[0][0])-1


def reconstruct(source,namelist,compiler):
    from native_reference import extract_routines
    n=grid_cells(namelist);bodies=[];provenance={}
    for name,(expected,names) in SOURCES.items():
        raw=(source/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=expected: raise ValueError(f'frozen grid source differs: {name}')
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        routines,_=extract_routines(source/name,blob,names)
        bodies.extend(routines)
        provenance[name]=dict(sha256=expected,routine_sha256={key:hashlib.sha256(body.encode()).hexdigest()
            for key,body in zip(names,routines,strict=True)})
    with tempfile.TemporaryDirectory(prefix='native-grid-') as temporary:
        code=Path(temporary)/'grid.F90';code.write_text(HEADER+'\n'.join(bodies)+ADAPTER)
        library=Path(temporary)/'grid.so'
        flags=['-O2','-cpp','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
        subprocess.run([compiler,*flags,str(code),'-o',str(library)],cwd=temporary,check=True)
        native=ctypes.CDLL(str(library))
        native.grid.argtypes=[ctypes.c_int,np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')]
        native.grid.restype=None
        grid=np.empty((n+1,n+1,2,6),order='F');native.grid(n,grid)
        provenance['compiled_source_sha256']=hashlib.sha256(code.read_bytes()).hexdigest()
    lon,lat=(grid[:,:,index,:].transpose(2,1,0) for index in (0,1))
    points=np.stack([np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)],axis=-1)
    def triangle(a,b,c):
        return 2.*np.arctan2(abs(np.sum(a*np.cross(b,c),axis=-1)),1.+np.sum(a*b+b*c+c*a,axis=-1))
    a,b,c,d=points[:,:-1,:-1],points[:,:-1,1:],points[:,1:,1:],points[:,1:,:-1]
    area=triangle(a,b,c)+triangle(a,c,d)
    assert np.isfinite(grid).all() and np.all(area>0.)
    np.testing.assert_allclose(area.sum(),4.*np.pi,rtol=0.,atol=1e-12)
    provenance.update(cells=n,input_namelist_sha256=hashlib.sha256(namelist.read_bytes()).hexdigest(),
        controls=dict(grid_type=0,shift_fac=18.,do_schmidt=False,do_cube_transform=False),
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],compiler_flags=flags,
        total_solid_angle_sr=float(area.sum()),limits='Default frozen FV3 grid only. Corner geometry is reconstructed, not read from the defective static export. Original model outputs are unchanged.')
    return dict(corner_longitude_rad=lon,corner_latitude_rad=lat,cell_solid_angle_sr=area,
        provenance_json=np.array(json.dumps(provenance,sort_keys=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True,help='frozen GFDL_atmos_cubed_sphere source directory')
    parser.add_argument('--input-namelist',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    if args.output.exists(): parser.error('output already exists; choose a new sidecar path')
    result=reconstruct(args.source,args.input_namelist,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as handle: np.savez_compressed(handle,**result)
    print(json.dumps(dict(status='pass',cells=result['cell_solid_angle_sr'].shape[-1],output=str(args.output))))


if __name__=='__main__': main()
