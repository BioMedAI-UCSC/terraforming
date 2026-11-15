"""Compare physical dust-mode coagulation with the unchanged native driver."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.dust_coagulation import coagulation_step
from native_reference import compile_routines,extract_routines

BLOB='e0c1818849461fa252ff1980dd264a93c57d79ab'


def compile_reference(source,destination,compiler,direct):
    path=source/'atmos_param_mars/coagulation.F90'
    extract_routines(path,BLOB,['coagul_main'])
    raw=path.read_text()
    findval=raw[raw.index('integer function findval('):raw.index('end function findval')+len('end function findval')]
    names=['coagul_main','dynvis','kinvis','thervel','mfp','cun','diff','lambda','delta','fallv',
        'reyn','stokes','schmidt','coal','betab','betag','betade','betade_tabfunc','betati','make_tables','frac','boxinterp']
    header=('#define FULLCOAG\n' if direct else '')+"""module reference_mod
use iso_c_binding
implicit none
real,parameter :: pi=acos(-1.d0),grav=3.72d0,rdgas=189.02d0,kbz=1.38064852d-23
real,parameter :: dpden_dt=2500.d0,dev_dt=.63676d0,mco20=.04401d0/6.02d23,athird=1.d0/3.d0
integer,parameter :: nres_coag=25,nt_nontag=100,table_numt=15,table_nump=25,table_numm=25
integer :: ndust_mass=1,dust_mass_indx(2)=(/1,3/),coal_kg=1
real :: rads_coag(25),vols_coag(25),deltar_coag(25),vrat_coag
real :: table_temp(15),table_pres(25),table_mfp(25)
real :: table_b(15,25,25,25),table_g(15,25,25,25),table_de(15,25,25,25)
real :: effb=1.,coal_fac=1.
logical :: kernel_b=.true.,kernel_g=.true.,kernel_de=.true.,kernel_ti=.false.,mcpu0,initialized=.false.
contains
integer function mpp_pe()
mpp_pe=0
end function
integer function mpp_root_pe()
mpp_root_pe=0
end function
"""
    adapter='\n'+findval+"""
subroutine run(n,modes,dt,temp,pres,mass,number,flags,coalescence,efficiency,outm,outn) bind(C)
integer(c_int),value :: n,modes,coalescence
real(c_double),value :: dt,efficiency
real(c_double),intent(in) :: temp(n),pres(n),mass(n,modes),number(n,modes)
integer(c_int),intent(in) :: flags(3)
real(c_double),intent(out) :: outm(n,modes),outn(n,modes)
real :: lat(1,1),lon(1,1),half(1,1,n+1),p(1,1,n),t(1,1,n)
real :: q(1,1,n,2*modes),dq(1,1,n,2*modes)
integer :: i,j,k
if (.not.initialized.or.coal_kg/=coalescence.or.effb/=efficiency) then
 coal_kg=coalescence;effb=efficiency
 vrat_coag=(40.d-6/1.d-8)**(3.d0/(nres_coag-1))
 do i=1,nres_coag
  rads_coag(i)=1.d-8*vrat_coag**((i-1)/3.d0)
  vols_coag(i)=4.d0/3.d0*pi*(1.d-8)**3*vrat_coag**(i-1)
 enddo
 deltar_coag=2.d0*rads_coag*2.d0**(1.d0/3.d0)*(vrat_coag**(1.d0/3.d0)-1)/(1+vrat_coag)**(1.d0/3.d0)
 call make_tables()
 initialized=.true.
endif
ndust_mass=modes
kernel_b=flags(1)/=0;kernel_g=flags(2)/=0;kernel_de=flags(3)/=0
lat=0;lon=0;half=0;dq=0
do k=1,n
 t(1,1,k)=temp(k);p(1,1,k)=pres(k)
 do j=1,modes
  q(1,1,k,2*j-1)=mass(k,j);q(1,1,k,2*j)=number(k,j)
 enddo
enddo
call coagul_main(1,1,1,1,n,2*modes,lon,lat,dt,t,half,p,q,dq)
do k=1,n
 do j=1,modes
  outm(k,j)=dq(1,1,k,2*j-1);outn(k,j)=dq(1,1,k,2*j)
 enddo
enddo
end subroutine
end module
"""
    library,provenance=compile_routines(path,BLOB,names,header,adapter,destination,compiler)
    provenance['findval_sha256']=hashlib.sha256(findval.encode()).hexdigest()
    provenance['direct_kernels']=direct
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    integers=np.ctypeslib.ndpointer(dtype=np.int32,flags='F_CONTIGUOUS')
    native.run.argtypes=[ctypes.c_int]*2+[ctypes.c_double]+[array]*4+[integers,ctypes.c_int,ctypes.c_double]+[array]*2
    native.run.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[];provenance=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='dust-coagulation-native-') as tmp:
        for direct in (False,True):
            dest=Path(tmp)/str(direct);dest.mkdir()
            native,info=compile_reference(source,dest,compiler,direct);provenance.append(info)
            t=f([150.,190.,250.,300.]);p=f([10.,610.,5e4,1000.])
            for modes in (1,2):
                for flags in ((1,0,0),(0,1,0),(0,0,1),(1,1,1)):
                    for coalescence in (0,1,2):
                        for dt in (5.,300.,3e4):
                            mass=f(np.broadcast_to([1e-5,2e-5][:modes],(4,modes)))
                            number=f(np.broadcast_to([1e8,1.5e8][:modes],(4,modes)))
                            outm=f(np.empty_like(mass));outn=f(np.empty_like(number))
                            native.run(4,modes,dt,t,p,mass,number,np.array(flags,dtype=np.int32),coalescence,1.,outm,outn)
                            actual=coagulation_step(t,p,mass,number,dt,brownian=bool(flags[0]),
                                gravitational=bool(flags[1]),diffusion_enhancement=bool(flags[2]),
                                coalescence=coalescence,direct=direct)
                            assert np.all(np.isfinite(actual.mass_tendency)) and np.all(np.isfinite(outm))
                            np.testing.assert_allclose(actual.mass_tendency,outm,rtol=3e-9,atol=1e-19,
                                err_msg=str((direct,modes,flags,coalescence,dt,'mass')))
                            np.testing.assert_allclose(actual.number_tendency,outn,rtol=3e-9,atol=1e-6,
                                err_msg=str((direct,modes,flags,coalescence,dt,'number')))
                            cases.append(dict(direct=direct,modes=modes,kernels=flags,coalescence=coalescence,
                                dt_seconds=dt,max_mass_tendency_error=float(np.max(abs(actual.mass_tendency-outm))),
                                max_number_tendency_error=float(np.max(abs(actual.number_tendency-outn)))))
                for empty in (False,True):
                    mass=f(np.full((4,modes),0. if empty else 1e-12))
                    number=f(np.full((4,modes),0. if empty else 10.))
                    outm=f(np.empty_like(mass));outn=f(np.empty_like(number))
                    native.run(4,modes,300.,t,p,mass,number,np.ones(3,dtype=np.int32),1,1.,outm,outn)
                    actual=coagulation_step(t,p,mass,number,300.,direct=direct)
                    # The source tests raw+dt*tendency <= 0 after cancellation.
                    # A one-ulp divide/multiply change selects its floor repair.
                    floor_bound=1e-15*(1.+16.*np.finfo(float).eps)
                    np.testing.assert_allclose(actual.mass_tendency,outm,rtol=3e-9,atol=floor_bound)
                    np.testing.assert_allclose(actual.number_tendency,outn,rtol=3e-9,atol=1e-14*(1.+16.*np.finfo(float).eps))
                    cases.append(dict(direct=direct,modes=modes,empty=empty,case='inactive-column',
                        floor_switch_tendency_bound_kg_kg_s=floor_bound,
                        max_mass_tendency_error=float(np.max(abs(actual.mass_tendency-outm)))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_dust_coagulation.py','scripts/native_reference.py','package/src/framework/physics/dust_coagulation.py']
    return dict(status='pass',reference=provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Physical modes in table and FULLCOAG paths. Source table corner order, pressure fallback, '
            'number reconstruction and 1e-15 mass regularization are retained. '
            'Empty-bin ratios use zero weight. Equal radii have explicit zero differential settling '
            'to preserve the source identity in compiled reverse derivatives. '
            'Inactive columns retain the source floor repair and mass loss, recorded by diagnostics. '
            'Their raw+dt*tendency<=0 cancellation can select different floor branches; '
            'that comparison records the exact source 1e-15 mass-tendency floor bound separately. '
            'Turbulent inertia has uninitialized epsilon in the source.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native dust-coagulation cases passed")
