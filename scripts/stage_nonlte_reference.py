"""Stage the pinned MGITM atmosphere for reference non-LTE cooling."""
import argparse
import hashlib
from pathlib import Path
import numpy as np

BLOB='e4f7471bdedd4f7d52b06e717e73fb088369cf92'


def stage(path,output):
    raw=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=BLOB: raise ValueError('non-LTE reference differs from pinned source')
    data=np.fromstring(raw.decode(),sep=' ').reshape(8,70)[:,::-1]
    names=('altitude_m','pressure_mbar','temperature_k','co2_m3','o_m3','co_m3','n2_m3','cooling_k_s')
    data[0]*=1e3;data[1]*=.01;data[7]/=88775.
    assert np.all(np.diff(data[1])>0.) and np.all(data[2:7]>0.)
    np.savez_compressed(output,**dict(zip(names,data,strict=True)),source_git_blob=blob)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();stage(args.reference,args.output)


if __name__=='__main__': main()
