"""Compare explicit CO2/argon mass exchange with unchanged native update loops."""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.dinosaur_physics import remap_gas
from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY

SOURCE_BLOB='26f594e50fb4c6be6645cbd54d027ba72706c619'


def compile_reference(source,destination,compiler):
    raw=(Path(source)/'atmos_cubed_sphere_mars/mars_physics_update.F90').read_bytes()
    blob=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
    if blob!=SOURCE_BLOB: raise ValueError('gas-mass update differs from the pinned release')
    text=raw.decode()
    start=text.index('    q1(:,jsw:jew,:,nco2)')
    species=text[start:text.index('  else  ! total mass',start)]
    start=text.index('   do k=1,nlev',start)
    pressure=text[start:text.index('!-----------------------------------------------------------------------------',start)]
    code='''module reference_mod
use iso_c_binding
implicit none
contains
subroutine update(n,q,delp,dmass,grav,qout) bind(C)
integer(c_int),value :: n
real(c_double),value :: grav
real(c_double),intent(in) :: q(1,1,n,3),dmass(1,1,n)
real(c_double),intent(inout) :: delp(1,1,n)
real(c_double),intent(out) :: qout(1,1,n,3)
real :: q1(1,1,n,3),q_dt(1,1,n,3),dt_delp(1,1,n),qratio(1,1,n)
real,parameter :: dt=7.
integer,parameter :: is=1,ie=1,jsw=1,jew=1,js=1,je=1,nco2=1,nar=2
integer :: i,j,k,nlev
nlev=n
q1=q
q_dt=0.
'''+species+pressure+'''
qout=(q+dt*q_dt)/spread(qratio,4,3)
end subroutine
end module
'''
    destination=Path(destination);(destination/'reference.f90').write_text(code)
    flags=['-O3','-fdefault-real-8','-fdefault-double-8','-fPIC','-shared']
    result=subprocess.run([compiler,*flags,'reference.f90','-o','reference.so'],
        cwd=destination,capture_output=True,text=True)
    if result.returncode: raise RuntimeError(result.stderr)
    library=ctypes.CDLL(str(destination/'reference.so'))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags='F_CONTIGUOUS')
    library.update.argtypes=[ctypes.c_int,array,array,array,ctypes.c_double,array]
    library.update.restype=None
    def evaluate(q,dp,loss):
        q=np.asfortranarray(q[None,None]);dp=np.asfortranarray(dp[None,None])
        loss=np.asfortranarray(loss[None,None]);out=np.empty_like(q)
        library.update(q.shape[2],q,dp,loss,BODY.gravity_m_s2,out)
        return out[0,0],dp[0,0]
    return evaluate,dict(source_git_blob=blob,
        species_loop_sha256=hashlib.sha256(species.encode()).hexdigest(),
        pressure_loop_sha256=hashlib.sha256(pressure.encode()).hexdigest(),
        compiled_source_sha256=hashlib.sha256(code.encode()).hexdigest(),compiler_flags=flags,
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
        adaptation='unchanged do_qmass species and qratio pressure loops; declarations, C adapter and fv_phys ratio normalization')


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    with tempfile.TemporaryDirectory(prefix='gas-mass-native-') as temporary:
        native,reference=compile_reference(source,temporary,compiler)
        for layers in (1,4,12,56):
            weights=np.arange(1,layers+1,dtype=float);weights/=weights.sum()
            for ps in (61.,705.,5e4):
                gas=ps*weights/BODY.gravity_m_s2
                for fraction in (.1,.8,.999):
                    co2=fraction*np.linspace(.8,1.,layers)
                    q=np.stack([co2,1.-co2,np.full(layers,1e-5)],axis=-1)
                    for loss_fraction in (-.01,0.,.01):
                        loss=gas*co2*loss_fraction
                        native_q,native_dp=native(q,ps*weights,loss)
                        *_,actual,new_pressure=remap_gas(jnp.full(layers,200.),jnp.zeros(layers),
                            jnp.zeros(layers),q,jnp.asarray(ps),loss,weights,BODY,co2_index=0)
                        target=float(new_pressure)*weights/BODY.gravity_m_s2
                        actual_mass=np.sum(target[:,None]*actual,axis=0)
                        expected_mass=np.sum(native_dp[:,None]/BODY.gravity_m_s2*native_q,axis=0)
                        np.testing.assert_allclose(new_pressure,native_dp.sum(),rtol=3e-14)
                        np.testing.assert_allclose(actual_mass,expected_mass,rtol=3e-14,atol=1e-16)
                        if layers==1: np.testing.assert_allclose(actual,native_q,rtol=3e-14)
                        cases.append(dict(layers=layers,pressure_pa=ps,co2_fraction=fraction,
                            loss_fraction=loss_fraction,
                            max_species_mass_error_kg_m2=float(np.max(abs(actual_mass-expected_mass)))))
    root=Path(__file__).parents[1]
    paths=('scripts/validate_gas_mass.py','package/src/framework/physics/dinosaur_physics.py',
        'package/src/framework/physics/co2_clouds.py',
        'package/src/celestials/planets/mars/reference_constants.py')
    return dict(status='pass',reference=reference,cases=cases,
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
        limits='Native mass update and species inventories agree; multilevel results compare integrated species mass after the separate Dinosaur sigma remap. This does not validate mixture thermodynamics or climate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native gas-mass cases passed")
