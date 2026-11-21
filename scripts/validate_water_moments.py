"""Check moment-water kinetics against unchanged original native routines."""
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
from src.framework.physics.water_moments import (nucleation_rates,growth_coefficient,shape_factor,size_grid,
    nucleation_condensation,subcycled_cloud_step,EffectiveState)
from native_reference import extract_routines

BLOB='4897f72860b59d7effeb6650dbf3d4484b3aca71'


def compile_reference(source,destination,compiler,precision):
    names=['nuclea','growthrate2','fshape2','scrit_tdep','nucleacond','findQ','newton','watsat']
    routines,blob=extract_routines(source/'atmos_param_mars/micromod_mgcm.F90',BLOB,names)
    header="""module constants_mod
implicit none
real*8,parameter :: grav=3.72d0,cp_air=735.94d0,rdgas=189.02d0,pi=acos(-1.d0)
end module
module initracer_mod
implicit none
integer,parameter :: nbin=4,ntrace_mom=6,iMa_dt=1,iNb_dt=2,iMa_cld=3,iNb_cld=4,iMa_cor=5,iMa_vap=6
integer :: ndust_mass=1,nt_nontag=100,dust_mass_indx(2)=(/1,7/)
real*8,parameter :: athird=1.d0/3.d0,dpden_ice=917.d0
real*8 :: aerdens(6)=(/2500.d0,2500.d0,917.d0,917.d0,917.d0,0.d0/)
real*8 :: stdv(6)=(/.63676d0,.63676d0,.3087d0,.3087d0,.3087d0,0.d0/),aerad(4),rb(5)
end module
module reference_mod
use initracer_mod
use constants_mod
implicit none
integer :: mteta_case=0
logical :: latent_heat=.false.
real*8 :: mteta=.965
real*8,parameter :: kbz=1.38064852d-23,mh2o=18.01d-3,rgp=8.3143d0
real*8,parameter :: vo1=mh2o/917.d0,nav=6.02d23,desorp=.288d-19
real*8,parameter :: surfdif=desorp/10.d0,nus=1.d13,m0=mh2o/nav
contains
"""
    kind=16 if precision==128 else 8
    adapter=f"""module bridge
use iso_c_binding
use reference_mod
implicit none
contains
subroutine column(t,p,q,dm,dn,dt,modes,latent,radii,edges,heat,out,outm,outn,sat,contact) bind(C)
real(c_double),value :: t,p,dt
integer(c_int),value :: modes,latent
real(c_double),intent(in) :: q(6),dm(modes),dn(modes),radii(4),edges(5)
real(c_double),intent(out) :: heat(1),out(6),outm(modes),outn(modes),sat(1),contact(1)
real(kind={kind}) :: pl(5),tl(5),rho(5),tracer(5,6),mass(5,modes),number(5,modes)
real(kind={kind}) :: dh(5),dq(5,6),dqm(5,modes),dqn(5,modes),s(1),m(1)
ndust_mass=modes
mteta_case=0
latent_heat=latent/=0
aerad=real(radii,kind={kind});rb=real(edges,kind={kind})
pl=p;tl=t;rho=1;tracer=0;mass=0;number=0
tracer(4,:)=q;mass(4,:)=dm;number(4,:)=dn
call nucleacond(1,real(dt,kind={kind}),pl,tl,rho,tracer,mass,number,modes>1,dh,dq,dqm,dqn,s,m)
heat(1)=dh(4);out=dq(4,:);outm=dqm(4,:);outn=dqn(4,:);sat=s;contact=m
! Single-mode driver rebuilds dust bins from tracer slots on the next call.
if(modes==1) then
  outm(1)=out(1);outn(1)=out(2)
endif
end subroutine
subroutine nucleation(t,p,s,counts,radii,model,out,m) bind(C)
real(c_double),value :: t,p,s
integer(c_int),value :: model
real(c_double),intent(in) :: counts(4),radii(4)
real(c_double),intent(out) :: out(4),m(1)
real(kind={kind}) :: rate(4),contact
aerad=real(radii,kind={kind})
mteta_case=model
call nuclea(real(p,kind={kind}),real(t,kind={kind}),real(s,kind={kind}),real(counts,kind={kind}),rate,contact)
out=real(rate,kind=c_double)
m(1)=real(contact,kind=c_double)
end subroutine
subroutine growth(t,p,psat,seq,r,out) bind(C)
real(c_double),value :: t,p,psat,seq,r
real(c_double),intent(out) :: out(1)
real(kind={kind}) :: result
call growthrate2(real(300,kind={kind}),real(t,kind={kind}),real(p,kind={kind}),real(0,kind={kind}), &
 real(psat,kind={kind}),real(seq,kind={kind}),real(r,kind={kind}),result)
out(1)=real(result,kind=c_double)
end subroutine
subroutine shape(m,x,out) bind(C)
real(c_double),value :: m,x
real(c_double),intent(out) :: out(1)
real(kind={kind}) :: result
call fshape2(real(m,kind={kind}),real(x,kind={kind}),result)
out(1)=real(result,kind=c_double)
end subroutine
end module
"""
    destination=Path(destination);destination.mkdir()
    code=destination/'reference.f90';bridge=destination/'bridge.f90'
    code.write_text(header+'\n'.join(routines)+'\nend module\n');bridge.write_text(adapter)
    common=['-O3','-cpp','-fPIC'];flags=[*common,'-fdefault-real-8','-fdefault-double-8']
    if precision==128:
        flags.remove('-fdefault-double-8')
        flags+=['-freal-8-real-16']
    for command in ([compiler,*flags,'-c',str(code),'-o',str(destination/'reference.o')],
                    [compiler,*common,'-c',str(bridge),'-o',str(destination/'bridge.o')],
                    [compiler,'-shared',str(destination/'reference.o'),str(destination/'bridge.o'),'-o',str(destination/'reference.so')]):
        result=subprocess.run(command,cwd=destination,capture_output=True,text=True)
        if result.returncode: raise RuntimeError(result.stderr)
    native=ctypes.CDLL(str(destination/'reference.so'))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.nucleation.argtypes=[ctypes.c_double]*3+[array,array,ctypes.c_int,array,array]
    native.growth.argtypes=[ctypes.c_double]*5+[array]
    native.shape.argtypes=[ctypes.c_double]*2+[array]
    native.column.argtypes=[ctypes.c_double]*2+[array]*3+[ctypes.c_double]+[ctypes.c_int]*2+[array]*8
    for name in ('nucleation','growth','shape','column'): getattr(native,name).restype=None
    return native,dict(source_git_blob=blob,real_precision_bits=precision,compiler_flags=flags,
        routine_sha256={n:hashlib.sha256(r.encode()).hexdigest() for n,r in zip(names,routines,strict=True)},
        adapter_sha256=hashlib.sha256(bridge.read_bytes()).hexdigest(),
        adaptation='unchanged original routine bodies; intended source precision or binary128; separate C conversions; '
            'single-mode adapter synchronizes auxiliary dust tendencies with native tracer slots, as the driver does')


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    radii=np.asfortranarray(size_grid()[0]);counts=np.ones(4)
    with tempfile.TemporaryDirectory(prefix='water-moment-native-') as tmp:
        native,provenance=compile_reference(source,Path(tmp)/'quad',compiler,128)
        fp64,fp_provenance=compile_reference(source,Path(tmp)/'double',compiler,64)
        for m in (.5,.9,.965,.99):
            for x in (.01,1.,10.,100.,1000.,3000.,3000.01):
                expected=np.empty(1);old=np.empty(1);native.shape(m,x,expected);fp64.shape(m,x,old)
                actual=shape_factor(m,x)
                np.testing.assert_allclose(actual,expected[0],rtol=3e-11,atol=2e-15)
                cases.append(dict(case='embryo-geometry',contact=m,radius_ratio=x,
                    absolute_error=float(abs(actual-expected[0])),native_fp64_error=float(abs(old[0]-expected[0]))))
        for model in range(4):
            for t in (150.,170.,190.):
                for sat in (.8,1.,2.,4.,10.):
                    ph=611.*np.exp(22.5*(1.-273.16/t))*sat
                    expected=np.empty(4);contact=np.empty(1);old=np.empty(4);old_contact=np.empty(1)
                    native.nucleation(t,ph,sat,counts,radii,model,expected,contact)
                    fp64.nucleation(t,ph,sat,counts,radii,model,old,old_contact)
                    rate,m=nucleation_rates(ph,t,sat,counts,contact_case=model)
                    np.testing.assert_array_equal(np.isnan(rate),np.isnan(expected))
                    np.testing.assert_array_equal(np.isinf(rate),np.isinf(expected))
                    np.testing.assert_allclose(rate,expected,rtol=3e-11,atol=1e-20)
                    np.testing.assert_allclose(m,contact[0],rtol=3e-15,atol=1e-15)
                    cases.append(dict(case='nucleation',contact_case=model,temperature_k=t,saturation=sat,
                        nonfinite_count=int(np.sum(~np.isfinite(expected))),
                        max_relative_error=(float(np.max(np.abs(rate-expected)/np.maximum(np.abs(expected),1e-100)))
                            if np.all(np.isfinite(expected)) else None),
                        native_fp64_nonfinite_count=int(np.sum(~np.isfinite(old))),
                        native_fp64_relative_error=(float(np.max(np.abs(old-expected)/np.maximum(np.abs(expected),1e-100)))
                            if np.all(np.isfinite(old)) else None)))
        for t in (150.,190.,250.):
            for p in (10.,610.,5e4):
                for r in (1e-8,1e-6,1e-4):
                    psat=611.*np.exp(22.5*(1.-273.16/t));expected=np.empty(1)
                    native.growth(t,p,psat,1.1,r,expected)
                    actual=growth_coefficient(t,p,psat,1.1,r)
                    np.testing.assert_allclose(actual,expected[0],rtol=3e-13,atol=1e-25)
                    cases.append(dict(case='growth-resistance',temperature_k=t,pressure_pa=p,radius_m=r,
                        absolute_error=float(abs(actual-expected[0]))))
        for modes in (1,2):
            for latent in (False,True):
                for t in (170.,190.,220.):
                    for sat in (.1,1.,2.,10.):
                        p=610.;dt=100.
                        qs=611.*np.exp(22.5*(1.-273.16/t))*18./(44.*p)
                        q=np.array([1e-5,1e8,1e-6,1e6,1e-8,qs*sat])
                        dm=np.array([1e-5,2e-5])[:modes].copy()
                        dn=np.array([1e8,1.5e8])[:modes].copy()
                        heat=np.empty(1);out=np.empty(6);outm=np.empty(modes);outn=np.empty(modes)
                        saturation=np.empty(1);contact=np.empty(1)
                        fp64.column(t,p,q,dm,dn,dt,modes,int(latent),radii,np.asfortranarray(size_grid()[1]),
                            heat,out,outm,outn,saturation,contact)
                        result=nucleation_condensation(t,p,q,dt,dust_mass=dm,dust_number=dn,latent_heat=latent)
                        assert all(np.all(np.isfinite(x)) for x in result)
                        # Native root has an absolute 1e-8 stopping rule. Report
                        # that bound separately from the strict no-root cases.
                        allowance=1e-8/dt if latent else 1e-20
                        for index,tolerance in enumerate((1e-20,1e-8,allowance,1e-8,1e-20,allowance)):
                            np.testing.assert_allclose(result.tracer_tendency[index],out[index],rtol=3e-10,
                                atol=tolerance,err_msg=str((modes,latent,t,sat,index)))
                        np.testing.assert_allclose(result.dust_mass_tendency,outm,rtol=3e-10,atol=1e-20)
                        np.testing.assert_allclose(result.dust_number_tendency,outn,rtol=3e-10,atol=1e-8)
                        np.testing.assert_allclose(result.temperature_tendency,heat[0],rtol=3e-10,
                            atol=allowance*2.8343e6/735.94)
                        cases.append(dict(case='coupled-nucleation-condensation',modes=modes,latent_heat=latent,
                            temperature_k=t,saturation=sat,max_tracer_error=float(np.max(abs(result.tracer_tendency-out))),
                            temperature_tendency_error=float(abs(result.temperature_tendency-heat[0]))))
        for substeps in (1,5,10):
            for modes in (1,2):
                for latent in (False,True):
                    t0=190.;t=191.;p=610.;dt=100.
                    qs=611.*np.exp(22.5*(1.-273.16/t0))*18./(44.*p)
                    q0=np.array([1e-5,1e8,1e-6,1e6,1e-8,qs*3.])
                    q=q0*np.array([.9,1.,1.,1.,1.,1.2])
                    dm0=np.array([1e-5,2e-5])[:modes];dm=dm0*.9
                    dn0=np.array([1e8,1.5e8])[:modes];dn=dn0.copy()
                    current_t=t0;current_q=np.minimum(q0,q);current_dm=np.minimum(dm0,dm)
                    current_dn=np.minimum(dn0,dn)
                    ramp_q=(q-current_q)/substeps;ramp_dm=(dm-current_dm)/substeps
                    ramp_dn=(dn-current_dn)/substeps;microdt=dt/substeps
                    for _ in range(substeps):
                        current_t+=(t-t0)/substeps
                        current_q=np.maximum(current_q+ramp_q,0.)
                        current_dm=np.maximum(current_dm+ramp_dm,0.)
                        current_dn=np.maximum(current_dn+ramp_dn,0.)
                        heat=np.empty(1);out=np.empty(6);outm=np.empty(modes);outn=np.empty(modes)
                        native.column(current_t,p,current_q,current_dm,current_dn,microdt,modes,int(latent),radii,
                            np.asfortranarray(size_grid()[1]),heat,out,outm,outn,np.empty(1),np.empty(1))
                        current_t+=microdt*heat[0];current_q=np.maximum(current_q+microdt*out,0.)
                        current_dm=np.maximum(current_dm+microdt*outm,0.)
                        current_dn=np.maximum(current_dn+microdt*outn,0.)
                    actual=subcycled_cloud_step(t0,t,p,q0,q,dt,microtimestep=microdt,
                        initial_dust_mass=dm0,dust_mass=dm,initial_dust_number=dn0,dust_number=dn,
                        latent_heat=latent)
                    allowance=substeps*1e-8/dt if latent else 1e-20
                    for index,tolerance in enumerate((1e-20,1e-8,allowance,1e-8,1e-20,allowance)):
                        np.testing.assert_allclose(actual.tracer_tendency[index],(current_q[index]-q[index])/dt,
                            rtol=3e-10,atol=tolerance,err_msg=str((substeps,modes,latent,index)))
                    np.testing.assert_allclose(actual.temperature_tendency,(current_t-t)/dt,rtol=3e-10,
                        atol=allowance*2.8343e6/735.94+1e-15)
                    cases.append(dict(case='native-substep-composition',substeps=substeps,modes=modes,
                        latent_heat=latent,max_tracer_error=float(np.max(abs(actual.tracer_tendency-(current_q-q)/dt)))))
                    # Seed all history fields: the released driver never fills its dust-history arrays.
                    historical=subcycled_cloud_step(t0-4.,t,p,q0*.5,q,dt,microtimestep=microdt,
                        initial_dust_mass=dm0*.5,dust_mass=dm,initial_dust_number=dn0*.5,dust_number=dn,
                        effective_input_mode=1,previous_state=EffectiveState(t0,q0,dm0,dn0),latent_heat=latent)
                    for value,expected in zip(historical[:4],actual[:4],strict=True):
                        np.testing.assert_array_equal(value,expected)
                    cases.append(dict(case='seeded-previous-effective-substeps',substeps=substeps,modes=modes,
                        latent_heat=latent))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=provenance,original_fp64_reference=fp_provenance,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_water_moments.py','package/src/framework/physics/water_moments.py',
             'package/src/framework/physics/water_bulk.py')},
        limits='Native nucleacond and kinetic primitives; source subeffmode=0 ramp composed with native calls. '
            'Original FP64 geometry cancellation and contact-case-1 subsaturated NaNs are recorded. '
            'Binary128 only resolves the geometric cancellation; it does not change the source equations. '
            'Empty bins use zero weight; dry vapor uses saturation pressure directly. '
            'The shared implicit root removes the native 1e-8 stopping noise. '
            'Previous effective input uses seeded history with native substeps; the released driver never '
            'fills its dust-history arrays. The port saves pre-cloud dust alongside source pre-cloud Q '
            'and post-cloud T. Outer sedimentation and long climate validation remain separate.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} moment-water native cases passed")


if __name__=='__main__': main()
