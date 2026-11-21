"""Stage the released CO2/CO2 and CO2/H2 collision-induced absorption inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

TABLES={
    'co2_rotational':('kgbar_8band_100_800.dat','508eed3dddd6cd0e86689fae7f05cfb67588848d'),
    'co2_translational':('kbbar_8band_100_800.dat','5c699474da6e6231c061a389fbda932e86db609f'),
    'co2_h2':('khbar_8band_T20_100_800.dat','2f1d46c136f9a5eae89b9969af29085296673d44'),
}


def stage(directory,output):
    values={};provenance={}
    for key,(name,expected) in TABLES.items():
        path=Path(directory)/name;raw=path.read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
        if blob!=expected: raise ValueError(f'collision table differs from pinned source: {name}')
        value=np.loadtxt(path,dtype=np.float64)
        if value.shape!=(8,15) or not np.isfinite(value).all() or np.any(value<0.):
            raise ValueError(f'invalid collision coefficients: {name}')
        values[key]=value
        provenance[name]=dict(git_blob=blob,sha256=hashlib.sha256(raw).hexdigest())
    values['temperature_k']=np.arange(100.,801.,50.)
    values['source_provenance_json']=json.dumps(provenance,sort_keys=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**values)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-data',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=Path(__file__).parents[1]/'package/src/framework/physics/collision_absorption_tables.npz')
    args=parser.parse_args();stage(args.reference_data,args.output)
    print(f'staged {args.output}')
