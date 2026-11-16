"""Compare size-dependent particle opacity with unchanged released Fortran kernels."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.particle_optics import dust_optics,co2_cloud_optics,water_bulk_optics,water_moment_optics,load_tables
from src.framework.physics.prescribed_dust import fixed_optics
from native_reference import compile_routines

RADIATION_BLOB='40d8879c3735bbec6cabc63817cad5220f281532'


def compile_reference(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
integer,parameter :: nbin_rt=20,nbin_rtblk=29,nbin_rtco2=29,nratioblk=1,nlonv=7,l_nspectv=7,l_nrefv=6
integer,parameter :: ndust_mass=2,nt_nontag=4
integer,parameter :: nratio=15,ima_cld=1,inb_cld=2,ima_cor=3
integer :: dust_mass_indx(2)=[1,3]
integer :: ntrace=4,l_layers,l_levels,nloni,l_nspecti,l_nrefi
real*8,parameter :: grav=3.72d0,pi=acos(-1.d0),athird=1.d0/3.d0
real*8,parameter :: dev_dt=.63676d0,dev_ice=.3087d0,dev_co2ice=.3087d0
real*8,parameter :: dpden_dt=2500.d0,dpden_ice=917.d0,dpden_co2ice=1620.d0
real*8 :: bulk_ccn=1.d5,bulk_co2_ccn=1.d5,radb_rt(21),radb_rtco2(30),rad_rtblk(29)
real*4 :: qextv_dst(20,7),qscatv_dst(20,7),gv_dst(20,7)
real*4 :: qextv_co2cld(1,29,7),qscatv_co2cld(1,29,7),gv_co2cld(1,29,7)
real*4 :: qextv_bcld(1,29,7),qscatv_bcld(1,29,7),gv_bcld(1,29,7)
real*4 :: qextv_blcld(1,29,7),qscatv_blcld(1,29,7),gv_blcld(1,29,7)
real*4 :: qextv_cld(15,20,7),qscatv_cld(15,20,7),gv_cld(15,20,7),fcorrect(20,15)
real*4 :: cor_ratio(15)=[.1,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.7,.8,.9,.99]
real*4,allocatable :: qexti_dst(:,:),qscati_dst(:,:),gi_dst(:,:)
real*4,allocatable :: qexti_co2cld(:,:,:),qscati_co2cld(:,:,:),gi_co2cld(:,:,:)
real*4,allocatable :: qexti_bcld(:,:,:),qscati_bcld(:,:,:),gi_bcld(:,:,:)
real*4,allocatable :: qexti_blcld(:,:,:),qscati_blcld(:,:,:),gi_blcld(:,:,:)
real*4,allocatable :: qexti_cld(:,:,:),qscati_cld(:,:,:),gi_cld(:,:,:)
logical :: user_fixed_dust_opts=.false.
real*8 :: reff_fixed=1.5e-6
real*8 :: qextv(7)=0.,qscatv(7)=0.,ggv(7)=0.,qexti(8)=0.,qscati(8)=0.,ggi(8)=0.
real*8 :: qxv_read(7)=0.,qsv_read(7)=0.,gv_read(7)=0.,qxi_read(8)=0.,qsi_read(8)=0.,gi_read(8)=0.
contains
"""
    adapter="""
subroutine particles(kind,n,nir,nb,nr,ps,q,t,vin,iin,vl,il,edges,radii,fc,extv,scatv,momv,exti,scati,momi) bind(C)
integer(c_int),value :: kind,n,nir,nb,nr
real(c_double),intent(in) :: ps(2*n+3),q(n,4),t(2*n+3)
real(c_double),intent(in) :: vin(nr,nb,3,7),iin(nr,nb,3,nir),vl(nr,nb,3,7),il(nr,nb,3,nir)
real(c_double),intent(in) :: edges(nb+1),radii(nb),fc(20,15)
real(c_double),intent(out) :: extv(2*n+2,7),scatv(2*n+2,7),momv(2*n+2,7)
real(c_double),intent(out) :: exti(2*n+2,nir),scati(2*n+2,nir),momi(2*n+2,nir)
real*8 :: qxv(2*n+4,7),qsv(2*n+4,7),gv(2*n+4,7)
real*8 :: qxi(2*n+4,nir),qsi(2*n+4,nir),gi(2*n+4,nir)
real*8 :: ref(2*n+4),tau(2*n+4),uv(2*n+4),diag(4),split(4,2),surf(20,n),p(2*n+3)
integer :: k
l_layers=n
l_levels=2*n+3
l_nspecti=nir
nloni=nir
l_nrefi=4
if(nir==8) l_nrefi=7
p=ps/100.d0
if(allocated(qexti_dst)) then
 deallocate(qexti_dst,qscati_dst,gi_dst,qexti_co2cld,qscati_co2cld,gi_co2cld)
 deallocate(qexti_bcld,qscati_bcld,gi_bcld,qexti_blcld,qscati_blcld,gi_blcld)
 deallocate(qexti_cld,qscati_cld,gi_cld)
endif
allocate(qexti_dst(20,nir),qscati_dst(20,nir),gi_dst(20,nir))
allocate(qexti_co2cld(1,29,nir),qscati_co2cld(1,29,nir),gi_co2cld(1,29,nir))
allocate(qexti_bcld(1,29,nir),qscati_bcld(1,29,nir),gi_bcld(1,29,nir))
allocate(qexti_blcld(1,29,nir),qscati_blcld(1,29,nir),gi_blcld(1,29,nir))
allocate(qexti_cld(15,20,nir),qscati_cld(15,20,nir),gi_cld(15,20,nir))
if(kind==3) then
 qextv_cld=real(vin(:,:,1,:),4)
 qscatv_cld=real(vin(:,:,2,:),4)
 gv_cld=real(vin(:,:,3,:),4)
 qexti_cld=real(iin(:,:,1,:),4)
 qscati_cld=real(iin(:,:,2,:),4)
 gi_cld=real(iin(:,:,3,:),4)
 fcorrect=real(fc,4)
 radb_rt=edges
 call opt_cld(q,p,qxv,qxi,qsv,qsi,gv,gi,ref,tau,diag,uv)
else &
if(kind==0) then
 qextv_dst=real(vin(1,:,1,:),4)
 qscatv_dst=real(vin(1,:,2,:),4)
 gv_dst=real(vin(1,:,3,:),4)
 qexti_dst=real(iin(1,:,1,:),4)
 qscati_dst=real(iin(1,:,2,:),4)
 gi_dst=real(iin(1,:,3,:),4)
 radb_rt=edges
 call opt_dst(q,p,qxv,qxi,qsv,qsi,gv,gi,ref,tau,surf,diag(1:2),split)
else if(kind==1) then
 qextv_co2cld(1,:,:)=real(vin(1,:,1,:),4)
 qscatv_co2cld(1,:,:)=real(vin(1,:,2,:),4)
 gv_co2cld(1,:,:)=real(vin(1,:,3,:),4)
 qexti_co2cld(1,:,:)=real(iin(1,:,1,:),4)
 qscati_co2cld(1,:,:)=real(iin(1,:,2,:),4)
 gi_co2cld(1,:,:)=real(iin(1,:,3,:),4)
 radb_rtco2=edges
 call opt_co2cld(q,p,qxv,qxi,qsv,qsi,gv,gi,ref,tau,diag(1:2),1)
else
 qextv_bcld(1,:,:)=real(vin(1,:,1,:),4)
 qscatv_bcld(1,:,:)=real(vin(1,:,2,:),4)
 gv_bcld(1,:,:)=real(vin(1,:,3,:),4)
 qexti_bcld(1,:,:)=real(iin(1,:,1,:),4)
 qscati_bcld(1,:,:)=real(iin(1,:,2,:),4)
 gi_bcld(1,:,:)=real(iin(1,:,3,:),4)
 qextv_blcld(1,:,:)=real(vl(1,:,1,:),4)
 qscatv_blcld(1,:,:)=real(vl(1,:,2,:),4)
 gv_blcld(1,:,:)=real(vl(1,:,3,:),4)
 qexti_blcld(1,:,:)=real(il(1,:,1,:),4)
 qscati_blcld(1,:,:)=real(il(1,:,2,:),4)
 gi_blcld(1,:,:)=real(il(1,:,3,:),4)
 rad_rtblk=radii
 call opt_cld_blk(q,p,t,qxv,qxi,qsv,qsi,gv,gi,ref,tau,diag,1,.false.,uv)
endif
do k=2,l_levels
 extv(k-1,:)=tau(k)/ref(k)*qxv(k,:)
 scatv(k-1,:)=tau(k)/ref(k)*qsv(k,:)
 momv(k-1,:)=tau(k)/ref(k)*qsv(k,:)*gv(k,:)
 exti(k-1,:)=tau(k)/ref(k)*qxi(k,:)
 scati(k-1,:)=tau(k)/ref(k)*qsi(k,:)
 momi(k-1,:)=tau(k)/ref(k)*qsi(k,:)*gi(k,:)
enddo
end subroutine
end module
"""
    adapter=adapter.replace("end module",'subroutine fixed_dust(n,nir,use_user,radius,p,tau,vin,iin,edges,user_v,user_i,ev,sv,mv,ei,si,mi) bind(C)\ninteger(c_int),value :: n,nir,use_user\nreal(c_double),value :: radius\nreal(c_double),intent(in) :: p(2*n+3),tau(n),vin(1,20,3,7),iin(1,20,3,nir),edges(21),user_v(3,7),user_i(3,nir)\nreal(c_double),intent(out) :: ev(2*n+2,7),sv(2*n+2,7),mv(2*n+2,7)\nreal(c_double),intent(out) :: ei(2*n+2,nir),si(2*n+2,nir),mi(2*n+2,nir)\nreal*8 :: qxv(2*n+4,7),qsv(2*n+4,7),gv(2*n+4,7),ref(2*n+4),amount(2*n+4)\nreal*8 :: qxi(2*n+4,nir),qsi(2*n+4,nir),gi(2*n+4,nir),dummyv(7),dummyi(nir),surf(20,n)\ninteger l,k\nl_layers=n\nl_levels=2*n+3\nnloni=nir\nl_nspecti=nir\nreff_fixed=radius\nqextv_dst=real(vin(1,:,1,:),4)\nqscatv_dst=real(vin(1,:,2,:),4)\ngv_dst=real(vin(1,:,3,:),4)\nif(allocated(qexti_dst)) deallocate(qexti_dst,qscati_dst,gi_dst)\nallocate(qexti_dst(20,nir),qscati_dst(20,nir),gi_dst(20,nir))\nqexti_dst=real(iin(1,:,1,:),4)\nqscati_dst=real(iin(1,:,2,:),4)\ngi_dst=real(iin(1,:,3,:),4)\nradb_rt=edges\ndummyv=0.\ndummyi=0.\nuser_fixed_dust_opts=use_user/=0\nqxv_read=user_v(1,:)\nqsv_read=user_v(2,:)\ngv_read=user_v(3,:)\nqxi_read(:nir)=user_i(1,:)\nqsi_read(:nir)=user_i(2,:)\ngi_read(:nir)=user_i(3,:)\ncall ini_optdst(dummyv,dummyi,qxv,qxi,qsv,qsi,gv,gi,ref,surf,.true.)\namount=0.\ndo l=1,n\n k=2*l+2\n amount(k)=tau(l)*(p(k)-p(k-1))/(p(k+1)-p(k-1))\n amount(k+1)=tau(l)*(p(k+1)-p(k))/(p(k+1)-p(k-1))\nenddo\ndo k=2,l_levels\n ev(k-1,:)=amount(k)/ref(k)*qxv(k,:)\n sv(k-1,:)=amount(k)/ref(k)*qsv(k,:)\n mv(k-1,:)=amount(k)/ref(k)*qsv(k,:)*gv(k,:)\n ei(k-1,:)=amount(k)/ref(k)*qxi(k,:)\n si(k-1,:)=amount(k)/ref(k)*qsi(k,:)\n mi(k-1,:)=amount(k)/ref(k)*qsi(k,:)*gi(k,:)\nenddo\nend subroutine\n'+"end module")
    library,provenance=compile_routines(source/'atmos_param_mars/Rad/rtmod_mgcm.F90',RADIATION_BLOB,
        ['opt_dst','opt_co2cld','opt_cld_blk','opt_cld','calc_opts','ini_optdst'],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    native.particles.argtypes=[ctypes.c_int]*5+[array]*16
    native.particles.restype=None
    native.fixed_dust.argtypes=[ctypes.c_int]*3+[ctypes.c_double]+[array]*13
    native.fixed_dust.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True)
    data=load_tables();cases=[];f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='particle-optics-') as temporary:
        native,provenance=compile_reference(source,Path(temporary),compiler)
        for n in (2,12,40):
            half=np.linspace(0.,610.,n+1);full=.5*(half[:-1]+half[1:])
            p=np.zeros(2*n+3);p[2::2]=half;p[3::2]=full
            t=np.linspace(230.,300.,2*n+3)
            for bands in (12,15):
                nir=5 if bands==12 else 8
                for kind in (0,1,2,3):
                    if kind==1 and bands==12: continue
                    name={0:f'dust_{bands}',1:'co2_15',2:'water_ice_15',3:f'water_moment_{bands}'}[kind]
                    grid={0:'moment',1:'co2',2:'bulk',3:'moment'}[kind]
                    v=data[f'{name}_sw'];ir=data[f'{name}_ir'][...,:nir]
                    vl=data['water_liquid_15_sw'] if kind==2 else v
                    il=data['water_liquid_15_ir'][...,:nir] if kind==2 else ir
                    for mode in ('clear','threshold','mixed','large'):
                        q=np.zeros((n,4))
                        if kind==0:
                            q[:,0]=np.geomspace(1e-12,1e-3,n);q[:,1]=np.geomspace(1.,1e10,n)
                            q[:,2]=q[::-1,0]*.5;q[:,3]=q[::-1,1]*.5
                            if mode=='large': q[:,0::2]*=1000.
                            if mode=='clear': q[:]=0.
                            if mode=='threshold': q[:,0::2]=1e-12
                            actual=dust_optics(q[:,[0,2]],q[:,[1,3]],p,bands=bands)
                        else:
                            q[:,0]=np.geomspace(1e-15,.1,n)
                            if mode=='large': q[:,0]*=100.
                            if mode=='clear': q[:]=0.
                            if mode=='threshold': q[:,0]=1e-18
                            if kind==3:
                                q[:,1]=1e8;q[:,2]=q[:,0]*.03
                                actual=water_moment_optics(q[:,0],q[:,2],q[:,1],p,bands=bands)
                            else:
                                actual=(co2_cloud_optics(q[:,0],p) if kind==1 else
                                    water_bulk_optics(q[:,0],p,t,bands=bands))
                        outputs=[f(np.zeros((2*n+2,nb))) for nb in (7,7,7,nir,nir,nir)]
                        native.particles(kind,n,nir,v.shape[1],v.shape[0],f(p),f(q),f(t),f(v),f(ir),f(vl),f(il),
                            f(data[f'{grid}_edges_m']),f(data[f'{grid}_radii_m']),f(data['water_absorption_correction']),*outputs)
                        errors=[]
                        for a,b in zip((*actual.visible,*actual.infrared),outputs,strict=True):
                            np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-13,
                                err_msg=f'{kind=} {n=} {bands=} {mode=}')
                            errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                        cases.append(dict(layers=n,bands=bands,kind=kind,case=mode,max_absolute_errors=errors))
        for n in (2,12):
            half=np.linspace(0.,610.,n+1);p=np.zeros(2*n+3)
            p[2::2]=half;p[3::2]=.5*(half[:-1]+half[1:])
            tau=np.geomspace(.001,.1,n)
            for bands in (12,15):
                nir=5 if bands==12 else 8
                for radius in (1e-8,1.5e-6,100e-6):
                    actual=fixed_optics(tau,p,effective_radius_m=radius,bands=bands)
                    outputs=[f(np.zeros((2*n+2,b))) for b in (7,7,7,nir,nir,nir)]
                    native.fixed_dust(n,nir,0,radius,f(p),f(tau),f(data[f'dust_{bands}_sw']),
                        f(data[f'dust_{bands}_ir']),f(data['moment_edges_m']),f(np.zeros((3,7))),f(np.zeros((3,nir))),*outputs)
                    errors=[]
                    for a,b in zip((*actual.visible,*actual.infrared),outputs,strict=True):
                        np.testing.assert_allclose(a,b,rtol=2e-11,atol=2e-13)
                        errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                    cases.append(dict(layers=n,bands=bands,kind='fixed-dust',radius_m=radius,max_absolute_errors=errors))
                for loading in (0.,1.):
                    # Deliberately exceed the averaging cap: native user inputs bypass it.
                    visible=f([np.linspace(1.,3.,7),np.linspace(.2,3.,7),np.linspace(-.2,.8,7)])
                    infrared=f([np.linspace(.1,2.,nir),np.linspace(.01,2.,nir),np.linspace(-.1,.7,nir)])
                    actual=fixed_optics(loading*tau,p,bands=bands,optical_properties=(visible,infrared))
                    outputs=[f(np.zeros((2*n+2,b))) for b in (7,7,7,nir,nir,nir)]
                    native.fixed_dust(n,nir,1,1.5e-6,f(p),f(loading*tau),f(data[f'dust_{bands}_sw']),
                        f(data[f'dust_{bands}_ir']),f(data['moment_edges_m']),visible,infrared,*outputs)
                    errors=[]
                    for a,b in zip((*actual.visible,*actual.infrared),outputs,strict=True):
                        np.testing.assert_allclose(a,b,rtol=2e-14,atol=2e-15)
                        errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                    np.testing.assert_array_equal(actual.diagnostics['size_distribution'],0.)
                    cases.append(dict(layers=n,bands=bands,kind='user-fixed-dust',loading=loading,
                        max_absolute_errors=errors))
    repo=Path(__file__).parents[1]
    paths=['scripts/validate_particle_optics.py','scripts/stage_particle_optics.py',
        'package/src/framework/physics/particle_optics.py','package/src/framework/physics/particle_optical_tables.npz',
        'package/src/framework/physics/prescribed_dust.py']
    return dict(status='pass',reference=provenance,cases=cases,
        optical_inputs=json.loads(str(data['source_provenance_json'])),
        sha256={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in paths})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=report['status'],cases=len(report['cases']))))
