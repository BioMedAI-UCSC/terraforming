"""Generate released hybrid-pressure grids with unchanged native set_eta."""
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

PATCH_BLOB='4116b38b088d4f39a5d8c1da76acc1ffb07847db'
LAYERS=(24,28,30,36,37,38,40,42,46,56,65,71,76,87)


def compile_reference(source,destination,compiler):
    path=Path(source)/'patches/srcGFDL_acs.patch';raw=path.read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=PATCH_BLOB: raise ValueError('vertical-grid patch differs from the pinned release')
    patch=raw.decode()
    added='\n'.join(line[1:] for line in patch.splitlines() if line.startswith('+') and not line.startswith('+++'))
    start=added.index(' subroutine set_eta(km, ks, ptop, ak, bk, p_ref)')
    end=added.index(' end subroutine set_eta',start)+len(' end subroutine set_eta')
    routine=added[start:end]
    section=patch.split('diff --git a/tools/fv_eta_mars.h b/tools/fv_eta_mars.h\n',1)[1].split('\ndiff --git ',1)[0]
    header='\n'.join(line[1:] for line in section.splitlines() if line.startswith('+') and not line.startswith('+++'))+'\n'
    destination=Path(destination)
    (destination/'fv_eta_mars.h').write_text(header)
    adapter='''
subroutine coefficients(n,pref,ks,top,a,b) bind(C)
integer(c_int),value :: n
real(c_double),value :: pref
integer(c_int),intent(out) :: ks
real(c_double),intent(out) :: top,a(n+1),b(n+1)
call set_eta(n,ks,top,a,b,pref)
end subroutine
end module
'''
    code='module reference_mod\nuse iso_c_binding\nimplicit none\ncontains\n'+routine+adapter
    (destination/'reference.f90').write_text(code)
    flags=['-O3','-cpp','-DMARS_GCM','-I.','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
    result=subprocess.run([compiler,*flags,'reference.f90','-o','reference.so'],cwd=destination,
        capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stderr)
    native=ctypes.CDLL(str(destination/'reference.so'))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    native.coefficients.argtypes=[ctypes.c_int,ctypes.c_double,ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_double),array,array]
    native.coefficients.restype=None
    def evaluate(n,pref):
        a=np.empty(n+1);b=np.empty(n+1);ks=ctypes.c_int();top=ctypes.c_double()
        native.coefficients(n,pref,ctypes.byref(ks),ctypes.byref(top),a,b)
        return a,b,ks.value,top.value
    return evaluate,dict(patch_git_blob=blob,grid_header_sha256=hashlib.sha256(header.encode()).hexdigest(),
        routine_sha256=hashlib.sha256(routine.encode()).hexdigest(),
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),compiler_flags=flags,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        adaptation='unchanged added set_eta and fv_eta_mars.h; module declarations and C adapter')


def stage(source,output,compiler):
    with tempfile.TemporaryDirectory(prefix='vertical-grid-native-') as temporary:
        native,reference=compile_reference(source,Path(temporary),compiler)
        fields={}
        for n in LAYERS:
            a,b,ks,top=native(n,725.)
            assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
            assert np.all(np.diff(a+725.*b)>0.) and b[-1]==1. and a[-1]==0. and top==a[0]
            fields.update({f'a{n}':a,f'b{n}':b,f'ks{n}':np.asarray(ks)})
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fields['reference_json']=np.asarray(json.dumps(reference,sort_keys=True))
    np.savez_compressed(output,**fields)
    print(f'{len(LAYERS)} released vertical grids staged')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    stage(args.ames_source,args.output,compiler)
