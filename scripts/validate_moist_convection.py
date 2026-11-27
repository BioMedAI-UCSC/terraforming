"""Compare moist adjustment with unchanged pinned GFDL/FMS routines."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.moist_convection import adjust
from src.framework.physics.saturation_pressure import specific_humidity
from stage_saturation_tables import compile_reference as compile_saturation,native_tables,BLOB as SATURATION_BLOB
from native_reference import extract_routines

BLOB='e7f2641002c2d2c76b0966aaf585d49e72af0ebf'


def compile_reference(source,saturation_source,mars_source,directory,compiler):
    source=source/'atmos_param/moist_conv/moist_conv.F90' if source.is_dir() else source
    directory=Path(directory)
    saturation,provenance=compile_saturation(saturation_source,mars_source,directory,compiler)
    routines,blob=extract_routines(source,BLOB,['moist_conv','CONV_DETR'])
    saturation_path=(saturation_source/'sat_vapor_pres/include/sat_vapor_pres_k.inc'
        if saturation_source.is_dir() else saturation_source)
    derivative,_=extract_routines(saturation_path,SATURATION_BLOB,['LOOKUP_DES_K_0D_'])
    header='''#define FMS_SVP_KIND_ r8_kind
#define LOOKUP_DES_K_0D_ native_derivative
module time_manager_mod
type time_type
integer :: unused=0
end type
end module
module sat_vapor_pres_mod
use reference_mod,only: native_qs=>compute_qs_k,native_es=>lookup_es_des_k, &
    dtres,tepsl,tminl,dtinvl,table_siz,DTABLE,D2TABLE
implicit none
integer,parameter :: r8_kind=kind(1.d0)
interface compute_qs
module procedure qs0,qs3
end interface
interface lookup_es_des
module procedure es0,es3
end interface
contains
subroutine qs0(t,p,qs,dqsdT,hc,esat)
real,intent(in) :: t,p
real,intent(out) :: qs
real,intent(in),optional :: hc
real,intent(out),optional :: dqsdT,esat
integer :: bad
call native_qs(t,p,189.02d0/461.50d0,461.50d0/189.02d0-1.d0,qs,bad,hc=hc,dqsdT=dqsdT,esat=esat)
if (bad/=0) error stop 'native moist saturation overflow'
end subroutine
subroutine qs3(t,p,qs,dqsdT,hc,esat)
real,intent(in) :: t(:,:,:),p(:,:,:)
real,intent(out) :: qs(:,:,:),dqsdT(:,:,:),esat(:,:,:)
real,intent(in) :: hc
integer :: i,j,k
do k=1,size(t,3)
 do j=1,size(t,2)
  do i=1,size(t,1)
   call qs0(t(i,j,k),p(i,j,k),qs(i,j,k),dqsdT(i,j,k),hc,esat(i,j,k))
  enddo
 enddo
enddo
end subroutine
subroutine es0(t,es,des)
real,intent(in) :: t
real,intent(out) :: es,des
integer :: bad
call native_es(t,es,des,bad)
if (bad/=0) error stop 'native moist lapse saturation overflow'
end subroutine
subroutine es3(t,es,des)
real,intent(in) :: t(:,:,:)
real,intent(out) :: es(:,:,:),des(:,:,:)
integer :: i,j,k
do k=1,size(t,3)
 do j=1,size(t,2)
  do i=1,size(t,1)
   call es0(t(i,j,k),es(i,j,k),des(i,j,k))
  enddo
 enddo
enddo
end subroutine
subroutine descomp(t,des)
real,intent(in) :: t
real,intent(out) :: des
integer :: bad
call native_derivative(t,des,bad)
if (bad/=0) error stop 'native moist derivative saturation overflow'
end subroutine
'''+derivative[0]+'''
end module
module moist_reference
use iso_c_binding
use time_manager_mod
use sat_vapor_pres_mod
implicit none
integer,parameter :: FATAL=1,WARNING=2,NOTE=3
real,parameter :: HLv=2.5d6,HLs=2.834d6,cp_air=735.94d0,grav=3.72d0,rdgas=189.02d0,rvgas=461.50d0
real,parameter :: d622=rdgas/rvgas,d378=1.d0-d622,grav_inv=1.d0/grav,rocp=rdgas/cp_air
real :: HC=1.d0,beta=0.d0,TOLmin=.02d0,TOLmax=.1d0
integer :: ITSMOD=30,num_mca_tracers=0,warning_count=0
logical :: do_simple=.false.,module_is_initialized=.true.
integer :: id_tdt_conv=0,id_qdt_conv=0,id_prec_conv=0,id_snow_conv=0,id_q_conv_col=0,id_t_conv_col=0
integer :: id_qldt_conv=0,id_qidt_conv=0,id_qadt_conv=0,id_ql_conv_col=0,id_qi_conv_col=0,id_qa_conv_col=0
integer,allocatable :: id_tracer_conv(:),id_tracer_conv_col(:)
interface send_data
module procedure send2,send3
end interface
contains
subroutine error_mesg(context,message,level)
character(len=*) :: context,message
integer :: level
if (level==FATAL) error stop 'native moist convection fatal error'
if (level==WARNING) warning_count=warning_count+1
end subroutine
logical function send2(id,values,clock,is,js)
integer :: id,is,js
real :: values(:,:)
type(time_type) :: clock
send2=.false.
end function
logical function send3(id,values,clock,is,js,k,rmask)
integer :: id,is,js,k
real :: values(:,:,:)
real,optional :: rmask(:,:,:)
type(time_type) :: clock
send3=.false.
end function
'''
    adapter='''
subroutine column(n,dt,settings,parameters,t,q,half,full,out,precip,unconverged) bind(C)
integer(c_int),value :: n
real(c_double),value :: dt
real(c_double),intent(in) :: settings(3),t(n),q(n),half(n+1),full(n)
integer(c_int),intent(in) :: parameters(2)
real(c_double),intent(out) :: out(n,2),precip(2)
integer(c_int),intent(out) :: unconverged
real :: tt(1,1,n),qq(1,1,n),hh(1,1,n+1),pp(1,1,n),dtt(1,1,n),dqq(1,1,n)
real :: rain(1,1),snow(1,1),tr(1,1,n,1),trdt(1,1,n,1)
logical :: cold(1,1)
type(time_type) :: clock
HC=settings(1);TOLmin=settings(2);TOLmax=settings(3);ITSMOD=parameters(1);do_simple=parameters(2)/=0
tt(1,1,:)=t;qq(1,1,:)=q;hh(1,1,:)=half;pp(1,1,:)=full
cold=t(n)<273.15d0;tr=0;warning_count=0
call moist_conv(tt,qq,pp,hh,cold,dtt,dqq,rain,snow,1.d0/dt,clock,1,1,tr,trdt)
if (any(trdt/=0.d0)) error stop 'source unexpectedly mixes other tracers'
out(:,1)=dtt(1,1,:);out(:,2)=dqq(1,1,:);precip=[rain(1,1),snow(1,1)]
unconverged=merge(1,0,warning_count>0)
end subroutine
end module
'''
    code=directory/'moist.f90';code.write_text(header+'\n'.join(routines)+adapter)
    library=directory/'moist.so'
    flags=['-O3','-cpp','-ffree-line-length-none','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
    result=subprocess.run([compiler,*flags,str(code),str(directory/'reference.so'),'-o',str(library)],
        cwd=directory,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stderr)
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.column.argtypes=[ctypes.c_int,ctypes.c_double,array,integers]+[array]*6+[ctypes.POINTER(ctypes.c_int)]
    native.column.restype=None
    return native,saturation,dict(source_git_blob=blob,saturation=provenance,compiler_flags=flags,
        routine_sha256={name:hashlib.sha256(body.encode()).hexdigest() for name,body in
            zip(('moist_conv','CONV_DETR'),routines,strict=True)},
        native_derivative_sha256=hashlib.sha256(derivative[0].encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
        adaptation='Unchanged moist_conv/CONV_DETR and FMS saturation bodies; scalar/array FMS adapters '
            'without I/O; optional stratiform fields omitted, as in the Mars caller; source warnings tallied')


def validate(source,saturation_source,mars_source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    options=[({},(4,12,40)),(dict(humidity=.8),(12,)),(dict(humidity=1.1),(4,)),
        (dict(simple_adjustment=True),(12,)),(dict(saturation_model='simple'),(12,)),
        (dict(simple_adjustment=True,saturation_model='simple'),(4,)),
        (dict(iterations=2,tolerance_max=.02),(4,)),
        (dict(iterations=4,tolerance_min=.03,tolerance_max=.24),(12,))]
    with tempfile.TemporaryDirectory(prefix='moist-convection-native-') as tmp:
        native,saturation,provenance=compile_reference(source,saturation_source,mars_source,Path(tmp),compiler)
        for config,layers in options:
            model=config.get('saturation_model','smithsonian')
            native_tables(saturation,model=='simple')
            forward=jax.jit(lambda t,q,h,p:adjust(t,q,h,p,300.,**config))
            for n in layers:
                for surface_pressure in (610.,50000.):
                    half=np.linspace(0.,surface_pressure,n+1);full=.5*(half[1:]+half[:-1])
                    for profile in ('dry','supersaturated-stable','active','isolated-block'):
                        t=np.full(n,200.) if profile in ('dry','supersaturated-stable') else np.linspace(
                            140. if surface_pressure==610. else 220.,230. if surface_pressure==610. else 300.,n)
                        qs=np.asarray(specific_humidity(t,full,relative_humidity=config.get('humidity',1.),model=model).specific_humidity)
                        q=qs*(.5 if profile=='dry' else 1.15)
                        if profile=='isolated-block':
                            q[:n//3]*=.5;q[-max(1,n//4):]*=.5
                        settings=f([config.get('humidity',1.),config.get('tolerance_min',.02),config.get('tolerance_max',.1)])
                        parameters=np.array([config.get('iterations',30),config.get('simple_adjustment',False)],dtype=np.int32)
                        output=np.empty((n,2),order='F');precip=np.empty(2);warning=ctypes.c_int()
                        native.column(n,300.,settings,parameters,f(t),f(q),f(half),f(full),output,precip,ctypes.byref(warning))
                        actual=forward(t,q,half,full)
                        for value,expected in zip(actual[:4],(*output.T,*precip),strict=True):
                            np.testing.assert_allclose(value,expected,rtol=3e-10,atol=2e-12,
                                err_msg=str((config,n,surface_pressure,profile)))
                        assert actual.diagnostics['unconverged']==bool(warning.value)
                        cases.append(dict(config=config,layers=n,surface_pressure_pa=surface_pressure,
                            profile=profile,native_unconverged=bool(warning.value),
                            max_temperature_tendency_error_k_s=float(np.max(abs(actual.temperature_tendency-output[:,0])))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_moist_convection.py','scripts/stage_saturation_tables.py','scripts/native_reference.py',
        'package/src/framework/physics/moist_convection.py','package/src/framework/physics/saturation_pressure.py',
        'package/src/framework/physics/saturation_tables.npz']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Mars-call path with no optional stratiform cloud fields; source other-tracer tendency is zero. '
            'Initial lapse omits HC, subsequent lapse uses HC; failed block searches exit the column; '
            'the final iteration at each tolerance skips flag refresh. Those source rules are retained. '
            'Surface routing and global coupling are separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gfdl-source',type=Path,required=True)
    parser.add_argument('--fms-source',type=Path,required=True)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.gfdl_source,args.fms_source,args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native moist-convection cases passed")
