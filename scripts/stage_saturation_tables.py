"""Stage FMS saturation tables from unchanged, pinned native kernels."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from native_reference import compile_routines

BLOB='00ce7089d0dc164d4936b7a8af4e20a2546f89f4'
PATCH_BLOB='6deea614300cc5340dfd5bd02fc2d5ddc1bdebee'


def compile_reference(source,mars_source,destination,compiler):
    source=source/'sat_vapor_pres/include/sat_vapor_pres_k.inc' if source.is_dir() else source
    patch=(mars_source/'patches/srcFMS.patch').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(patch)).encode()+b'\0'+patch).hexdigest()==PATCH_BLOB
    names=['SAT_VAPOR_PRES_INIT_K_','COMPUTE_ES_K_','COMPUTE_ES_LIQ_K_',
        'COMPUTE_ES_LIQ_ICE_K_','LOOKUP_ES_DES_K_0D_','LOOKUP_ES2_DES2_K_0D_',
        'LOOKUP_ES3_DES3_K_0D_','LOOKUP_ES_K_0D_','LOOKUP_ES2_K_0D_',
        'LOOKUP_ES3_K_0D_','COMPUTE_QS_K_0D_']
    bindings=['initialize_tables','compute_es_k','compute_es_liq_k','compute_es_liq_ice_k',
        'lookup_es_des_k','lookup_es2_des2_k','lookup_es3_des3_k','lookup_es_k',
        'lookup_es2_k','lookup_es3_k','compute_qs_k']
    header='#define FMS_SVP_KIND_ r8_kind\n'+''.join(f'#define {a} {b}\n' for a,b in zip(names,bindings,strict=True))+'''
module reference_mod
use iso_c_binding
implicit none
integer,parameter :: r8_kind=kind(1.d0)
real(r8_kind) :: dtres,tepsl,tminl,dtinvl
integer :: table_siz
real(r8_kind),allocatable :: TABLE(:),DTABLE(:),D2TABLE(:),TABLE2(:),DTABLE2(:),D2TABLE2(:)
real(r8_kind),allocatable :: TABLE3(:),DTABLE3(:),D2TABLE3(:)
logical :: use_exact_qs=.false.,module_is_initialized=.false.
contains
'''
    adapter='''
subroutine tables(simple,n,values) bind(C)
integer(c_int),value :: simple,n
real(c_double),intent(out) :: values(n,3,3)
real :: minimum
character(len=256) :: message
if (allocated(TABLE)) deallocate(TABLE,DTABLE,D2TABLE,TABLE2,DTABLE2,D2TABLE2,TABLE3,DTABLE3,D2TABLE3)
module_is_initialized=.false.
minimum=-223.d0
if (simple/=0) minimum=-173.d0
call initialize_tables(n,minimum,350.d0,273.16d0,2.5d6,461.50d0,1.d0,message, &
    .false.,simple/=0,.true.,.true.)
if (len_trim(message)>0) error stop 'native saturation table initialization failed'
values(:,1,1)=TABLE;values(:,2,1)=DTABLE;values(:,3,1)=D2TABLE
values(:,1,2)=TABLE2;values(:,2,2)=DTABLE2;values(:,3,2)=D2TABLE2
values(:,1,3)=TABLE3;values(:,2,3)=DTABLE3;values(:,3,3)=D2TABLE3
end subroutine
subroutine checks(n,t,p,q,humidity,exact,phase,output,bad) bind(C)
integer(c_int),value :: n,exact,phase
real(c_double),value :: humidity
real(c_double),intent(in) :: t(n),p(n),q(n)
real(c_double),intent(out) :: output(n,5)
integer(c_int),intent(out) :: bad(n)
integer :: i,nbad
real :: es,des,qs,dqs,scaled_es
use_exact_qs=exact/=0
do i=1,n
 if (phase==0) then
   call lookup_es_des_k(t(i),es,des,nbad)
   call compute_qs_k(t(i),p(i),189.02d0/461.50d0,461.50d0/189.02d0-1.d0, &
       qs,nbad,q=q(i),hc=humidity,dqsdT=dqs,esat=scaled_es)
 else if (phase==1) then
   call lookup_es2_des2_k(t(i),es,des,nbad)
   call compute_qs_k(t(i),p(i),189.02d0/461.50d0,461.50d0/189.02d0-1.d0, &
       qs,nbad,q=q(i),hc=humidity,dqsdT=dqs,esat=scaled_es,es_over_liq=.true.)
 else
   call lookup_es3_des3_k(t(i),es,des,nbad)
   call compute_qs_k(t(i),p(i),189.02d0/461.50d0,461.50d0/189.02d0-1.d0, &
       qs,nbad,q=q(i),hc=humidity,dqsdT=dqs,esat=scaled_es,es_over_liq_and_ice=.true.)
 endif
 output(i,:)=[es,des,qs,dqs,scaled_es];bad(i)=nbad
enddo
end subroutine
end module
'''
    library,provenance=compile_routines(source,BLOB,names,header,adapter,destination,compiler)
    provenance.update(fms_revision='71957e9d',mars_patch_git_blob=PATCH_BLOB,
        adaptation='Unchanged FMS kernel bodies with original kind/name macro bindings; '
            'NASA table bounds and water constants supplied by C adapters; no FMS I/O')
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.tables.argtypes=[ctypes.c_int]*2+[array];native.tables.restype=None
    native.checks.argtypes=[ctypes.c_int]+[array]*3+[ctypes.c_double]+[ctypes.c_int]*2+[array,integers]
    native.checks.restype=None
    return native,provenance


def native_tables(native,simple):
    minimum=(-173. if simple else -223.)+273.16
    size=5231 if simple else 5731
    values=np.empty((size,3,3),order='F')
    native.tables(int(simple),size,values)
    assert np.isfinite(values).all()
    return values.transpose(2,1,0).copy(),minimum


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fms-source',type=Path,required=True)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    with tempfile.TemporaryDirectory(prefix='saturation-tables-native-') as tmp:
        native,provenance=compile_reference(args.fms_source,args.ames_source,Path(tmp),compiler)
        data={}
        for simple,name in ((False,'smithsonian'),(True,'simple')):
            values,minimum=native_tables(native,simple)
            data[name+'_tables']=values;data[name+'_minimum_k']=minimum
        data['provenance_json']=json.dumps(provenance,sort_keys=True)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(args.output,**data)
    print('Staged both native saturation models and all three phase tables')
