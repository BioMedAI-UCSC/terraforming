"""Stage pinned size-dependent dust and cloud optical tables without changing precision."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

# Git blobs from the released v3.2.1 tree; optical arrays are REAL*4 in initcld.
TABLES={
    'dust_12_sw':('Dust_vis_wolff2010_JD_12bands.dat','67eb1202009f747c4ac3f25efa7480e426067022',1,20,7),
    'dust_12_ir':('Dust_ir_wolff2010_JD_12bands.dat','bdd8df6f24f8f12a46ac1f2efa16d1b643aad3cd',1,20,5),
    'dust_15_sw':('Dust_vis_wolff2010_JD_15bands.dat','dcb0f809400ce1f7fadd05ffd3b90ee469a39bb7',1,20,7),
    'dust_15_ir':('Dust_ir_wolff2010_JD_15bands.dat','379cd350b9ab29ed0c0e12554023353b9073a1b7',1,20,8),
    'water_moment_12_sw':('waterCoated_vis_JD_12bands.dat','70e7e3a3319030077f57f54cf8c728e40630488a',15,20,7),
    'water_moment_12_ir':('waterCoated_ir_JD_12bands.dat','89327bf6e191d797cf8e6886d6f275cef406966d',15,20,5),
    'water_moment_15_sw':('waterCoated_vis_JD_15bands.dat','70e7e3a3319030077f57f54cf8c728e40630488a',15,20,7),
    'water_moment_15_ir':('waterCoated_ir_JD_15bands.dat','25e95758f4f8b556345f0108679fe737e8d9a4b6',15,20,8),
    'co2_15_sw':('co2Coated_vis_MK_15bands29radii_1nratio.dat','50c5fb25e9d731ff2beda115ffce1384e0bd2c45',1,29,7),
    'co2_15_ir':('co2Coated_ir_MK_15bands29radii_1nratio.dat','43931da40129afe8cb0ed753bf5bdce423865058',1,29,8),
    'water_ice_15_sw':('h2oIceCoated_vis_KS_15bands29Radii_1nratio.dat','d27e0414220c8619a4bce5a73ce72771e19876c3',1,29,7),
    'water_ice_15_ir':('h2oIceCoated_ir_KS_15bands29Radii_1nratio.dat','f7ff973af209f47f59c362eed503f0e90802a826',1,29,8),
    'water_liquid_15_sw':('h2oLiqCoated_vis_KS_15bands29Radii_1nratio.dat','d12088febb3b34a335519d76e5d1057080343eac',1,29,7),
    'water_liquid_15_ir':('h2oLiqCoated_ir_KS_15bands29Radii_1nratio.dat','d60deb7fedfc21f0fefc715cdd1f42ca0f676908',1,29,8),
}


def size_grid(count,last_radius,span=None):
    """initcld radius centers and volume-based bin boundaries, in meters."""
    intervals=count-1 if span is None else span
    ratio=np.exp(np.log(last_radius/1.e-7)/intervals*3.)
    radii=np.empty(count);edges=np.empty(count+1)
    radii[0]=1.e-7;edges[0]=1.e-10;edges[-1]=1.e-2
    for i in range(count-1):
        radii[i+1]=radii[i]*ratio**(1./3.)
        edges[i+1]=(2.*ratio/(ratio+1.))**(1./3.)*radii[i]
    return radii,edges


def stage(data_directory,output):
    arrays={};provenance={}
    for name,(filename,expected,ratios,bins,bands) in TABLES.items():
        path=Path(data_directory)/filename;raw=path.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=expected:
            raise ValueError(f'optical input differs from pinned source: {filename}')
        values=np.loadtxt(path,dtype=np.float64)
        if values.shape!=(ratios*bins*3,bands) or not np.isfinite(values).all():
            raise ValueError(f'invalid optical table dimensions or values: {filename}')
        arrays[name]=values.reshape(ratios,bins,3,bands).astype(np.float32)
        provenance[filename]=dict(git_blob=blob,sha256=hashlib.sha256(raw).hexdigest())
    filename='QabsFCorrect.dat';raw=(Path(data_directory)/filename).read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!='13f649e1f06518afcd880df42c78e30b4f530c3e' or len(raw)!=1208 or (
            struct.unpack('<I',raw[:4])[0]!=1200 or raw[:4]!=raw[-4:]):
        raise ValueError('brightness correction differs from pinned Fortran input')
    arrays['water_absorption_correction']=np.frombuffer(raw[4:-4],dtype='<f4').reshape((20,15),order='F')
    provenance[filename]=dict(git_blob=blob,sha256=hashlib.sha256(raw).hexdigest())
    for kind,args in (('moment',(20,50.e-6)),('co2',(29,260.e-6)),('bulk',(29,50.e-6,19))):
        arrays[f'{kind}_radii_m'],arrays[f'{kind}_edges_m']=size_grid(*args)
    arrays['core_radius_ratios']=np.array([.1,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.8,.9,.99],np.float32)
    arrays['source_provenance_json']=json.dumps(provenance,sort_keys=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**arrays)


if __name__=='__main__':
    root=Path(__file__).parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=root/'package/src/framework/physics/particle_optical_tables.npz')
    args=parser.parse_args()
    stage(args.reference_data,args.output)
    print(f'staged {len(TABLES)} pinned optical tables: {args.output}')
