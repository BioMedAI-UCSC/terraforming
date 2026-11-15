"""Compare a coupled Ames-order column with native PBL/surface/CO2/convection."""
import argparse
import ctypes
import dataclasses
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
from src.framework.gcm._dinosaur import jax
from src.celestials.planets.mars.reference_constants import AMES_BODY_3D
from src.framework.physics.column_physics import column_physics
from src.framework.physics.soil_frost import soil_layer_thickness
from native_reference import compile_routines
from validate_dry_convection import UPDATE_BLOB
from validate_boundary_layer import compile_reference as compile_pbl
from validate_soil_frost import compile_reference as compile_surface


def compile_tail(source,destination,compiler):
    header="""module reference_mod
use iso_c_binding
implicit none
real :: grav,cp_air,kappa,co2_lheat
logical :: mix_tracers=.true.,mix_momentum=.true.
type time_type
integer :: unused=0
end type
contains
"""
    adapter="""
subroutine ames_tail(n,ntr,kap,cp,g,latent,t,u,v,r,ph,pf,tout,uout,vout,rout,mass,precip) bind(C)
integer(c_int), value :: n,ntr
real(c_double), value :: kap,cp,g,latent
real(c_double), intent(in) :: t(1,1,n),u(1,1,n),v(1,1,n),r(1,1,n,ntr),ph(1,1,n+1),pf(1,1,n)
real(c_double), intent(out) :: tout(1,1,n),uout(1,1,n),vout(1,1,n),rout(1,1,n,ntr)
real(c_double), intent(out) :: mass(1,1,n),precip(1,1)
real :: zeros(1,1,n),zero_r(1,1,n,ntr),co2heat(1,1,n),adj_t(1,1,n)
type(time_type) :: clock
grav=g
cp_air=cp
kappa=kap
co2_lheat=latent
zeros=0.
zero_r=0.
call co2_condense(1,1,1,1,n,1.,clock,t,zeros,ph,pf,precip,mass,co2heat)
call pass2(1,1,1,1,n,ntr,1.,clock,u,v,t,r,zeros,zeros,co2heat,zero_r, &
    uout,vout,adj_t,rout,ph(:,:,2:n+1)-ph(:,:,1:n),ph,pf)
tout=t+co2heat+adj_t
uout=u+uout
vout=v+vout
rout=r+rout
end subroutine
end module
"""
    library,provenance=compile_routines(source/"atmos_param_mars/update_mars_atmos.F90",
        UPDATE_BLOB,["co2_condense","pass2"],header,adapter,destination,compiler)
    native=ctypes.CDLL(str(library))
    array=np.ctypeslib.ndpointer(dtype=np.float64,flags="F_CONTIGUOUS")
    native.ames_tail.argtypes=[ctypes.c_int]*2+[ctypes.c_double]*4+[array]*12
    native.ames_tail.restype=None
    return native,provenance


def validate(source,compiler):
    jax.config.update("jax_enable_x64",True)
    body=AMES_BODY_3D
    cases=[]
    with tempfile.TemporaryDirectory(prefix="ames-column-") as temporary:
        root=Path(temporary)
        for name in ('pbl','surface','tail'):
            (root/name).mkdir()
        pbl,pbl_prov=compile_pbl(source,root/'pbl',compiler)
        surface,surface_prov=compile_surface(source,root/'surface',compiler)
        tail,tail_prov=compile_tail(source,root/'tail',compiler)
        for n in (4,12,40):
            ph=np.asfortranarray(np.linspace(0.,1.,n+1)*610.)
            pf=np.asfortranarray((ph[1:]+ph[:-1])*.5)
            u=np.asfortranarray(np.linspace(5.,30.,n));v=np.asfortranarray(-u*.3)
            r=np.asfortranarray(np.linspace(0.,1e-4,n*2).reshape(n,2))
            soil_n=16;dz=np.asfortranarray(soil_layer_thickness(soil_n))
            for name,skin,snow,down in (("clear",200.,0.,150.),("winter",148.,100.,0.),
                    ("thaw",148.,.001,200.),("convective",240.,0.,200.)):
                t=np.asfortranarray(np.linspace(110.,150.,n) if name=='winter' else np.linspace(180.,220.,n))
                if name=='convective':
                    t=np.asfortranarray(np.linspace(180.,300.,n)*(pf/610.)**body.kappa)
                rad=np.asfortranarray(np.linspace(-1e-5,1e-5,n))
                predicted=np.asfortranarray(t+rad*300.)
                pbl_t,pbl_u,pbl_v=(np.empty(n) for _ in range(3));pbl_r=np.empty_like(r)
                km,kh=(np.empty(2*n+1) for _ in range(2));fluxes=np.empty(9)
                pbl.column_physics(n,2,body.gravity_m_s2,body.gas_constant_j_kg_k,body.cp_j_kg_k,
                    300.,predicted,u,v,r,ph,pf,skin,snow,0.,pbl_t,pbl_u,pbl_v,pbl_r,km,kh,fluxes)
                soil=np.full(soil_n,skin);soil_new=np.empty(soil_n);ice_new=np.empty(1);phase=np.empty(1)
                surface.soil_frost(soil_n,300.,250.,610.,snow,down,fluxes[0],fluxes[8],.95,1.,
                    soil,dz,soil_new,ice_new,phase)
                out_t,out_u,out_v,mass=(np.empty(n) for _ in range(4));out_r=np.empty_like(r);precip=np.empty(1)
                tail.ames_tail(n,2,body.kappa,body.cp_j_kg_k,body.gravity_m_s2,590000.,
                    pbl_t,pbl_u,pbl_v,pbl_r,ph,pf,out_t,out_u,out_v,out_r,mass,precip)
                start=max(0,n-4)
                weights=np.diff(ph).copy();weights[:start]=0.;weights/=weights.sum()
                mass+=phase[0]*weights
                expected=(out_t,out_u,out_v,out_r,soil_new,ice_new[0]+precip[0],mass)
                actual=column_physics(t,u,v,r,soil,ph,pf,snow,rad,down,300.,body,
                    soil_thickness=dz,soil_diffusivity=(250./(1500.*627.9))**2,
                    soil_volumetric_heat_capacity=1500.*627.9,emissivity=.81 if snow>2. else .95,
                    surface_exchange_start=start)
                errors=[]
                for a,b in zip(actual[:7],expected,strict=True):
                    np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-12)
                    errors.append(float(np.max(np.abs(np.asarray(a)-b))))
                np.testing.assert_allclose(float(actual.co2_ice_kg_m2)-snow,
                    np.sum(actual.gas_mass_loss_kg_m2),rtol=2e-12,atol=2e-12)
                assert not np.any(actual.diagnostics['co2_mass_guard'])
                cases.append(dict(layers=n,case=name,max_absolute_errors=dict(zip(
                    ("temperature_k","u_ms","v_ms","tracer_ratio","soil_k","co2_ice_kg_m2","gas_mass_loss_kg_m2"),
                    errors,strict=True)),co2_mass_guard=bool(np.any(actual.diagnostics['co2_mass_guard']))))
    physics=Path(__file__).parents[1]/'package/src/framework/physics'
    return dict(status="pass",reference=dict(pbl=pbl_prov,surface=surface_prov,tail=tail_prov),cases=cases,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        body=dataclasses.asdict(body),
        physics_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in physics.glob('*.py')},
        jax_version=jax.__version__,devices=[str(x) for x in jax.devices()],
        limits="coupled column with prescribed radiative heating/downward flux; Dinosaur gas-mass remap and full radiation/global validation remain pending")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    compiler=shutil.which('gfortran')
    if compiler is None:
        parser.error('gfortran is required for the independent reference')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
