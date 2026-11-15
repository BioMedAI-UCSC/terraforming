"""Compare combined dust lifting/control with unchanged native source blocks."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.dust_lifting import interactive_step,background_step
from src.framework.physics.dust_control import LocalTimeState,local_time_step
from validate_dust_lifting import compile_reference,BLOB
from native_reference import compile_routines


def compile_local_reference(source,destination,compiler):
    path=source/'atmos_param_mars/dust_update.F90';text=path.read_text()
    start=text.index('if (ltfrac) then',text.index('!****** data for Local Time'))
    stop=text.index('!************ Custom sources',start)
    block=text[start:stop]
    header='''module reference_mod
use iso_c_binding
implicit none
real,parameter :: pi=acos(-1.d0)
contains
logical function send_data(id,values,clock,istart,jstart)
integer :: id,clock,istart,jstart
real :: values(:,:)
send_data=.false.
end function
'''
    adapter='''subroutine local_control(ns,nb,axis,bins,maps,fractions,ls,longitude,modelsol,oldtau,oldbin,output) bind(C)
integer(c_int),value :: ns,nb,oldbin
real(c_double),value :: ls,longitude,modelsol,oldtau
real(c_double),intent(in) :: axis(ns),bins(nb+1),maps(ns),fractions(ns,nb)
real(c_double),intent(out) :: output(4)
integer,parameter :: is=1,ie=1,js=1,je=1,Time=0
integer,parameter :: id_ltfrac=0,id_tau_old=0,dgdm_type=1
logical,parameter :: ltfrac=.true.,opac_from_aerosol=.false.
logical :: used
integer :: iaa,iaam,i,bins_length,bin_curr(1,1),bin_ltfrac_old(1,1)
real :: areolat,areo_ltfrac(ns),bins_ltfrac(nb+1),ltfrac_scenario_ls(1,1,ns,nb)
real :: tau_scenario_ls(1,1,ns),ltfrac_scenario(1,1,nb),dtau(1,1),tau_scenario(1,1)
real :: tau_ltfrac_old(1,1),ltfrac_curr(1,1),localtime(1,1),lon(1,1),taudust_fix(1,1,1)
real :: sols,delareo,fac1
areo_ltfrac=axis;bins_ltfrac=bins;bins_length=nb+1;areolat=ls;lon=longitude;sols=modelsol
ltfrac_scenario_ls(1,1,:,:)=fractions;tau_scenario_ls(1,1,:)=maps
tau_ltfrac_old=oldtau;bin_ltfrac_old=oldbin+1;taudust_fix=0
do iaa=1,ns
 if (axis(iaa)>ls) exit
enddo
iaam=iaa-1
if (iaam<1) iaam=iaam+ns
dtau=maps(iaa)-maps(iaam)
'''+block+'''
output(1)=tau_scenario(1,1);output(2)=ltfrac_curr(1,1)
output(3)=tau_ltfrac_old(1,1);output(4)=bin_ltfrac_old(1,1)-1
end subroutine
end module
'''
    library,provenance=compile_routines(path,BLOB,[],header,adapter,destination,compiler)
    provenance.update(local_time_block_sha256=hashlib.sha256(block.encode()).hexdigest(),
        adaptation='Unchanged local-time history block; single-column array adapter; diagnostic I/O stubbed')
    native=ctypes.CDLL(str(library));array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.local_control.argtypes=[ctypes.c_int]*2+[array]*4+[ctypes.c_double]*4+[ctypes.c_int,array]
    native.local_control.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='dust-control-native-') as tmp:
        combined=Path(tmp)/'combined';combined.mkdir()
        native,provenance=compile_reference(source,combined,compiler,combined=True)
        half=f([0.,50.,200.,400.,610.]);full=f(.5*(half[:-1]+half[1:]))
        t=f([160.,180.,210.,230.]);q=f(np.broadcast_to([1e-5,1e8,0.,0.,0.,1e-6],(4,6)))
        for mode in (1,2,21):
            for scale in (.5,2.):
                for current in (.05,.1,.3):
                    for injection in (1,7):
                        for pbl in (False,True):
                            scenario=.2;dt=300.
                            native.dual_config(mode,scale)
                            native.background_config(f([scenario,1.,1.,1.,2.45,2e-6,injection,1.]),
                                np.array([False,False,True,pbl],dtype=np.int32))
                            settings=f([dt,260.,.05,20.,200.,.5,0.,0.,current,30.])
                            out=f(np.empty_like(q));state=np.empty(4)
                            native.interactive(4,1,1,settings,half,full,t,q,
                                np.array([1,0,0,1,0,0,1,1],dtype=np.int32),out,state)
                            lifted=interactive_step(q[:,0],q[:,1],t,half,full,260.,.05,20.,1,200.,
                                .5,0.,0.,current,30.,dt,dual_mode=mode,dual_scale=scale,
                                scenario_opacity=scenario)
                            actual=background_step(lifted.mass_ratio,lifted.number_ratio,half,1,260.,0.,
                                current,scenario,1.,lifted.surface_dust_kg_m2,dt,extinction_efficiency=2.45,
                                injection_type=injection,inject_through_pbl=pbl)
                            for value,reference in ((actual.mass_ratio,out[:,0]),(actual.number_ratio,out[:,1]),
                                (actual.surface_dust_kg_m2,state[0])):
                                np.testing.assert_allclose(value,reference,rtol=3e-12,atol=1e-20)
                            cases.append(dict(mode=mode,dual_scale=scale,current_opacity=current,
                                injection_type=injection,inject_pbl=pbl,
                                max_mass_ratio_error=float(np.max(abs(actual.mass_ratio-out[:,0])))))
        local=Path(tmp)/'local';local.mkdir()
        native,local_provenance=compile_local_reference(source,local,compiler)
        axis=f([0.,90.,180.,270.,360.]);bins=f([0.,6.,12.,18.,24.])
        maps=f([.1,.5,.3,.05,.1]);fractions=f(np.arange(20).reshape(5,4)*.01)
        for ls in (0.,45.,90.,220.,310.,359.99):
            for longitude in (-np.pi,-.5*np.pi,0.,.5*np.pi,2.*np.pi):
                history=LocalTimeState(np.array([[.15]]),np.array([[0]]))
                for sol in (0.,.125,.25,.5,.75,1.,1.125):
                    output=np.empty(4)
                    native.local_control(5,4,axis,bins,maps,fractions,ls,longitude,sol,
                        float(history.opacity[0,0]),int(history.bin_index[0,0]),output)
                    opacity,fraction,updated=local_time_step(maps[:,None,None],fractions[:,None,None,:],
                        axis,bins,ls,longitude,sol,history)
                    actual=np.array([opacity[0,0],fraction[0,0],updated.opacity[0,0],updated.bin_index[0,0]])
                    np.testing.assert_allclose(actual,output,rtol=3e-12,atol=1e-15)
                    history=updated
                    cases.append(dict(case='local-time-history',solar_longitude_deg=ls,
                        longitude_rad=longitude,sol=sol))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_dust_control.py','scripts/validate_dust_lifting.py',
        'scripts/native_reference.py','package/src/framework/physics/dust_lifting.py',
        'package/src/framework/physics/dust_control.py']
    return dict(status='pass',reference=provenance,local_time_reference=local_provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='One physical dust mode with interactive lifting then background injection, '
            'old-opacity gating, and no removal. Sink repairs have independent conservation checks. '
            'Local-time history uses supplied mapped fractions and matching seasonal axes. '
            'Global multimode driver state remains a separate check.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native combined dust-control cases passed")
