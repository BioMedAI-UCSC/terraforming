"""Validate gravity-wave forcing against pinned, unchanged GFDL routines."""
import argparse
import ctypes
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.nonorographic_drag import (tendencies,wave_forcing,spectrum,
    initial_drag_state,driver_step,source_parameters)
from src.framework.physics.vertical_grid import supported_layers,sigma_boundaries,initialization_reference_interfaces
from native_reference import extract_routines

BLOB='fa555275885d8038641f3b99c7f284f84ff64516'
PATCH_BLOB='97732e0303a2e48935d1eece22a55c47a8a1baac'


def compile_reference(source,mars_source,destination,compiler):
    source=source/'atmos_param/cg_drag/cg_drag.F90' if source.is_dir() else source
    names=['gwfc','cg_drag_calc','cg_drag_time_vary']
    routines,blob=extract_routines(source,BLOB,names)
    # The shared extractor expects arguments; this routine has none.
    names.append('cg_drag_endts')
    routines.append(re.search(r'(?ims)^subroutine cg_drag_endts\s*\n.*?^end subroutine cg_drag_endts',
        source.read_text()).group())
    patch=(mars_source/'patches/srcatmos_phys.patch').read_bytes()
    assert hashlib.sha1(b'blob '+str(len(patch)).encode()+b'\0'+patch).hexdigest()==PATCH_BLOB
    raw=source.read_text()
    patched=Path(destination)/'atmos_param/cg_drag/cg_drag.F90'
    patched.parent.mkdir(parents=True)
    patched.write_text(raw)
    section=patch.decode().split('diff --git a/atmos_param/cg_drag/cg_drag.F90',1)[1].split('\ndiff --git ',1)[0]
    subprocess.run(['patch','--batch','-p1'],input='diff --git a/atmos_param/cg_drag/cg_drag.F90'+section,
        cwd=destination,text=True,capture_output=True,check=True)
    patched_routines,_=extract_routines(patched,'c8d71f86aba6dbe0b008825cfc5109e112c4349a',names[:-1])
    assert patched_routines==routines[:-1], 'NASA patch changed a validated forcing routine'
    patched_text=patched.read_text()
    start=patched_text.index('      do k=1,kmax')
    stop=patched_text.index('      source_level = MIN (source_level, kmax-1)',start)
    selection=patched_text[start:stop]+patched_text[stop:].splitlines()[0]+'\n'
    start=raw.index('      nc = 2.0*cmax/dc + 1')
    stop=raw.index('!    initialize netcdf diagnostic fields.',start)
    initialization=raw[start:raw.rfind('!--------------------------------------------------------------------',start,stop)]
    header="""module constants_mod
implicit none
real,parameter :: grav=3.72d0,rdgas=189.02d0,cp_air=735.94d0,pi=acos(-1.d0)
end module
module time_manager_mod
type time_type
integer :: seconds=0
end type
end module
module reference_mod
use iso_c_binding
use constants_mod
use time_manager_mod
implicit none
integer,parameter :: FATAL=1
integer :: nk=1,nc,flag=1,cgdrag_alarm=0,cg_drag_freq=0
integer :: id_kedx_cgwd=0,id_kedy_cgwd=0,id_bf_cgwd=0,id_gwfx_cgwd=0,id_gwfy_cgwd=0
real :: Bw=.4,Bn=0.,cw=25.,cmax=72.,dc=1.2
logical :: dump_flux=.false.,do_conserve_energy=.false.
real,allocatable :: c0(:),kwv(:),k2(:),source_amp(:,:),gwd_u(:,:,:),gwd_v(:,:,:)
integer,allocatable :: source_level(:,:)
contains
subroutine error_mesg(context,message,level)
character(len=*) :: context,message
integer :: level
error stop 'invalid native gravity-wave source'
end subroutine
logical function send_data(id,values,Time,is,js,k)
integer :: id,is,js
integer,optional :: k
real :: values(:,:,:)
type(time_type) :: Time
send_data=.false.
end function
"""
    adapter="""
subroutine select_source(n,nlat,pref,latb,parameters,levels,amplitudes) bind(C)
integer(c_int),value :: n,nlat
real(c_double),intent(in) :: pref(n),latb(1,nlat+1),parameters(4)
integer(c_int),intent(out) :: levels(nlat)
real(c_double),intent(out) :: amplitudes(nlat)
integer :: kmax,idf,jdf,k,i,j,klevel_of_source
real :: source_level_pressure,Bt_0,Bt_eq,Bt_eq_width,lat(1,nlat)
kmax=n-1;idf=1;jdf=nlat
source_level_pressure=parameters(1);Bt_0=parameters(2);Bt_eq=parameters(3);Bt_eq_width=parameters(4)
if (allocated(source_level)) deallocate(source_level,source_amp,gwd_u,gwd_v,c0,kwv,k2)
allocate(source_level(1,nlat),source_amp(1,nlat),gwd_u(1,nlat,kmax),gwd_v(1,nlat,kmax))
allocate(c0(0),kwv(0),k2(0))
"""+selection+"""
levels=source_level(1,:);amplitudes=source_amp(1,:)
end subroutine
subroutine configure(n_levels,src,amplitude,parameters,options)
integer :: n_levels,src,options(4)
real :: amplitude,parameters(5)
integer :: n
cmax=parameters(1);dc=parameters(2);cw=parameters(3);Bw=parameters(4);Bn=parameters(5)
nk=options(1);flag=options(2);dump_flux=options(3)/=0;do_conserve_energy=options(4)/=0
if (allocated(c0)) deallocate(c0,kwv,k2,source_amp,source_level,gwd_u,gwd_v)
allocate(source_amp(1,1),source_level(1,1),gwd_u(1,1,n_levels),gwd_v(1,1,n_levels))
source_amp=amplitude;source_level=src;gwd_u=0;gwd_v=0;cgdrag_alarm=0
"""+initialization+"""
end subroutine
subroutine column(n,src,amplitude,dt,p,z,t,u,v,parameters,options,du,dv,heat) bind(C)
integer(c_int),value :: n,src
real(c_double),value :: amplitude,dt
real(c_double),intent(in) :: p(n),z(n),t(n),u(n),v(n),parameters(5)
integer(c_int),intent(in) :: options(4)
real(c_double),intent(out) :: du(n),dv(n),heat(n)
real :: pp(1,1,n),zz(1,1,n),tt(1,1,n),uu(1,1,n),vv(1,1,n),lat(1,1)
real :: outu(1,1,n),outv(1,1,n),outt(1,1,n)
type(time_type) :: Time
call configure(n,src,amplitude,parameters,options)
pp(1,1,:)=p;zz(1,1,:)=z;tt(1,1,:)=t;uu(1,1,:)=u;vv(1,1,:)=v;lat=0
call cg_drag_calc(1,1,lat,pp,zz,tt,uu,vv,Time,dt,outu,outv,outt)
du=outu(1,1,:);dv=outv(1,1,:);heat=outt(1,1,:)
end subroutine
subroutine timed_column(n,src,amplitude,dt,frequency,alarm,p,z,t,u,v,parameters,options,du,dv,heat,next_alarm) bind(C)
integer(c_int),value :: n,src,frequency,alarm
real(c_double),value :: amplitude,dt
real(c_double),intent(in) :: p(n),z(n),t(n),u(n),v(n),parameters(5)
integer(c_int),intent(in) :: options(4)
real(c_double),intent(inout) :: du(n),dv(n)
real(c_double),intent(out) :: heat(n)
integer(c_int),intent(out) :: next_alarm
real :: pp(1,1,n),zz(1,1,n),tt(1,1,n),uu(1,1,n),vv(1,1,n),lat(1,1)
real :: outu(1,1,n),outv(1,1,n),outt(1,1,n)
type(time_type) :: Time
call configure(n,src,amplitude,parameters,options)
cg_drag_freq=frequency;cgdrag_alarm=alarm;gwd_u(1,1,:)=du;gwd_v(1,1,:)=dv
pp(1,1,:)=p;zz(1,1,:)=z;tt(1,1,:)=t;uu(1,1,:)=u;vv(1,1,:)=v;lat=0
call cg_drag_time_vary(dt)
call cg_drag_calc(1,1,lat,pp,zz,tt,uu,vv,Time,dt,outu,outv,outt)
call cg_drag_endts
du=outu(1,1,:);dv=outv(1,1,:);heat=outt(1,1,:);next_alarm=cgdrag_alarm
end subroutine
subroutine waves(n,src,amplitude,rho,u,bf,z,parameters,options,out,eddy) bind(C)
integer(c_int),value :: n,src
real(c_double),value :: amplitude
real(c_double),intent(in) :: rho(0:n),u(0:n),bf(0:n),z(0:n),parameters(5)
integer(c_int),intent(in) :: options(4)
real(c_double),intent(out) :: out(0:n),eddy(0:n)
real :: rr(1,1,0:n),uu(1,1,0:n),bb(1,1,0:n),zz(1,1,0:n),ff(1,1,0:n),kk(1,1,0:n)
call configure(n,src,amplitude,parameters,options)
rr(1,1,:)=rho;uu(1,1,:)=u;bb(1,1,:)=bf;zz(1,1,:)=z
call gwfc(1,1,1,1,source_level,source_amp,rr,uu,bb,zz,ff,kk)
out=ff(1,1,:);eddy=kk(1,1,:)
end subroutine
subroutine phases(n,values) bind(C)
integer(c_int),value :: n
real(c_double),intent(out) :: values(n)
values=c0
end subroutine
end module
"""
    destination=Path(destination);code=destination/'reference.f90'
    code.write_text(header+'\n'.join(routines)+adapter)
    library=destination/'reference.so'
    flags=['-O3','-cpp','-fdefault-real-8','-fdefault-double-8','-ffp-contract=off','-fPIC','-shared']
    result=subprocess.run([compiler,*flags,str(code),'-o',str(library)],cwd=destination,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stderr)
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    ints=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.column.argtypes=[ctypes.c_int]*2+[ctypes.c_double]*2+[array]*6+[ints]+[array]*3
    native.waves.argtypes=[ctypes.c_int]*2+[ctypes.c_double]+[array]*5+[ints]+[array]*2
    native.phases.argtypes=[ctypes.c_int,array]
    native.select_source.argtypes=[ctypes.c_int]*2+[array]*3+[ints,array]
    native.timed_column.argtypes=[ctypes.c_int]*2+[ctypes.c_double]*2+[ctypes.c_int]*2+[array]*6+[ints]+[array]*3+[ctypes.POINTER(ctypes.c_int)]
    for name in ('column','waves','phases','timed_column','select_source'): getattr(native,name).restype=None
    return native,dict(source_git_blob=blob,mars_patch_git_blob=PATCH_BLOB,
        compiler_flags=flags,real_precision_bits=64,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        routine_sha256={name:hashlib.sha256(body.encode()).hexdigest()
            for name,body in zip(names,routines,strict=True)},
        spectrum_initialization_sha256=hashlib.sha256(initialization.encode()).hexdigest(),
        source_selection_sha256=hashlib.sha256(selection.encode()).hexdigest(),
        adapter_sha256=hashlib.sha256(adapter.encode()).hexdigest(),
        adaptation='unchanged GFDL gwfc/cg_drag_calc and spectrum initialization; unchanged NASA-patched source selection; '
            'no FP contraction, so the zero phase speed remains exactly zero; diagnostic I/O stubbed')


def validate(source,mars_source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='nonorographic-native-') as tmp:
        native,provenance=compile_reference(source,mars_source,Path(tmp),compiler)
        latb=f(np.linspace(-np.pi/2.,np.pi/2.,10)[None,:]);lat=.5*(latb[0,1:]+latb[0,:-1])
        for n in supported_layers():
            for reference in (610.,705.,50000.):
                pressure=initialization_reference_interfaces(sigma_boundaries(n),reference)
                boundary=pressure[len(pressure)//2]
                for source_pressure in (150.,np.nextafter(boundary,0.),boundary,np.nextafter(boundary,np.inf),
                                        np.nextafter(pressure[-2],0.)):
                    if not np.any(pressure[:-1]>source_pressure): continue
                    for base,equatorial,width in ((0.,.08,45.),(.003,.04,.5)):
                        levels=np.empty(len(lat),dtype=np.int32);amplitudes=f(np.zeros(len(lat)))
                        native.select_source(n+1,len(lat),f(pressure),latb,
                            f([source_pressure,base,equatorial,width]),levels,amplitudes)
                        actual_level,actual_amp=source_parameters(pressure,lat,source_pressure_pa=source_pressure,
                            base_amplitude=base,equatorial_amplitude=equatorial,equatorial_width=width)
                        np.testing.assert_array_equal(np.full(len(lat),actual_level),levels)
                        np.testing.assert_allclose(actual_amp,amplitudes,rtol=2e-14,atol=1e-17)
                        cases.append(dict(case='source-selection',layers=n,reference_pressure_pa=reference,
                            source_pressure_pa=source_pressure,source_index=actual_level,
                            base_amplitude=base,equatorial_amplitude=equatorial,equatorial_width=width))
        for n in (4,12,40):
            p=np.geomspace(.03,610.,n);z=9000.*np.log(610./p);t=np.linspace(150.,220.,n)
            for src in (2,n-1):
                for wind in ('calm','shear','reversal','critical'):
                    u=(np.zeros(n) if wind=='calm' else np.linspace(10.,70.,n) if wind=='shear' else
                        np.linspace(-40.,20.,n) if wind=='reversal' else np.full(n,12.))
                    v=np.zeros(n) if wind=='calm' else -.3*u+1.7
                    for dump in (False,True):
                        for conserve in (False,True):
                            parameters=f([72.,1.2,25.,.4,0.]);options=np.array([1,1,dump,conserve],dtype=np.int32)
                            expected=[f(np.zeros(n)) for _ in range(3)]
                            native.column(n,src,.08,300.,f(p),f(z),f(t),f(u),f(v),parameters,options,*expected)
                            actual=tendencies(t,p,z,u,v,src,.08,300.,conserve_energy=conserve,dump_remaining_flux=dump)
                            errors=[]
                            for name,value,reference in zip(('u','v','heat'),actual[:3],expected,strict=True):
                                assert np.all(np.isfinite(value)) and np.all(np.isfinite(reference))
                                np.testing.assert_allclose(value,reference,rtol=3e-10,atol=2e-13,
                                    err_msg=str((n,src,wind,dump,conserve,name)))
                                errors.append(float(np.max(abs(value-reference))))
                            phase=np.empty(121);native.phases(121,phase)
                            np.testing.assert_array_equal(spectrum()[0],phase)
                            cases.append(dict(case='driver',layers=n,source_index=src,wind=wind,
                                dump_remaining_flux=dump,conserve_energy=conserve,max_errors=errors))
            rho=p/(189.02*t)
            extended_rho=np.r_[rho[0]**2/rho[1],rho];extended_z=np.r_[2*z[0]-z[1],z]
            extended_u=np.linspace(-23.7,51.3,n+1);bf=np.full(n+1,.01)
            for count in (1,2):
                for stationary in (False,True):
                    for dump in (False,True):
                        parameters=f([60.,2.,25.,.4,.1]);options=np.array([count,stationary,dump,False],dtype=np.int32)
                        expected=[f(np.zeros(n+1)) for _ in range(2)]
                        native.waves(n,n-1,.08,f(extended_rho),f(extended_u),f(bf),f(extended_z),parameters,options,*expected)
                        actual=wave_forcing(extended_rho,extended_u,bf,extended_z,n-1,.08,
                            maximum_speed=60.,speed_interval=2.,wavelength_count=count,narrow_amplitude=.1,
                            stationary_spectrum=stationary,dump_remaining_flux=dump)
                        for value,reference in zip(actual[:2],expected,strict=True):
                            np.testing.assert_allclose(value,reference,rtol=3e-10,atol=2e-13)
                        cases.append(dict(case='wave-spectrum',layers=n,wavelength_count=count,
                            stationary=stationary,dump_remaining_flux=dump))
        n=12;p=np.geomspace(.03,610.,n);z=9000.*np.log(610./p);t=np.linspace(150.,220.,n)
        for frequency,offset,dt in ((0,0,300.),(900,0,300.),(900,450,300.),(17,3,5.75),(2,0,5.75)):
            state=initial_drag_state(t,update_seconds=frequency,offset_seconds=offset)
            expected=[f(np.zeros(n)) for _ in range(3)];alarm=int(state.alarm_seconds)
            for step in range(8):
                u=np.linspace(-31.7,20.3,n)+step*.17;v=-.3*u+1.7
                next_alarm=ctypes.c_int()
                native.timed_column(n,10,.08,dt,frequency,alarm,f(p),f(z),f(t),f(u),f(v),
                    f([72.,1.2,25.,.4,0.]),np.array([1,1,False,True],dtype=np.int32),
                    *expected,ctypes.byref(next_alarm))
                actual,state=driver_step(t,p,z,u,v,10,.08,dt,state,update_seconds=frequency,
                    offset_seconds=offset,conserve_energy=True)
                for value,reference in zip(actual[:3],expected,strict=True):
                    np.testing.assert_allclose(value,reference,rtol=3e-10,atol=2e-13)
                assert state.alarm_seconds==next_alarm.value
                alarm=next_alarm.value
                cases.append(dict(case='timer-held-forcing',frequency_seconds=frequency,
                    offset_seconds=offset,dt_seconds=dt,step=step,alarm_seconds=alarm))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_nonorographic_drag.py','scripts/native_reference.py',
        'package/src/framework/physics/nonorographic_drag.py','package/src/framework/physics/vertical_grid.py',
        'package/src/framework/physics/vertical_grid_coefficients.npz']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Native kernel/driver plus patched source selection on released reference interfaces; no radiation/dynamics coupling. '
            'Source index 1 requires a finite auxiliary-wind extension; '
            'the coupled application interface requires separate checks.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gfdl-source',type=Path,required=True)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.gfdl_source,args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native gravity-wave cases passed")
