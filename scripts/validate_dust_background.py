"""Check scenario-driven dust updates on native-defined source branches."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import numpy as np
from src.framework.gcm._dinosaur import jax
from src.framework.physics.dust_lifting import background_step,reference_efficiencies
from src.framework.physics.particle_optics import load_tables
from validate_dust_lifting import compile_reference
from validate_particle_optics import compile_reference as compile_optics_reference


def validate(source,compiler):
    jax.config.update('jax_enable_x64',True);cases=[]
    f=lambda x:np.asfortranarray(x,dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix='background-dust-native-') as tmp:
        directory=Path(tmp)/'background';directory.mkdir()
        native,provenance=compile_reference(source,directory,compiler,background=True)
        half=f([0.,50.,200.,400.,610.]);full=f(.5*(half[:-1]+half[1:]))
        t=f([160.,180.,210.,230.]);q=f(np.broadcast_to([1e-5,1e8,0.,0.,0.,1e-6],(4,6)))
        for injection_type in range(1,8):
            for inject_pbl,remove_excess in ((False,False),(True,False),(False,True)):
                for current,scenario,ts,snow,change in ((.1,.2,230.,0.,1.),(.2,.1,230.,0.,-1.),
                    (.2,.2,230.,0.,0.),(.1,.2,100.,0.,1.),(.1,.2,230.,1.,1.),
                    (.1,.2,230.,0.,-1000.)):
                    settings=f([300.,ts,0.,0.,200.,.5,snow,0.,current,30.])
                    options=f([scenario,change,1.,1.,2.45,2e-6,injection_type,1.])
                    native.background_config(options,np.array([remove_excess,False,True,inject_pbl],dtype=np.int32))
                    out=f(np.empty_like(q));state=np.empty(4)
                    native.interactive(4,1,1,settings,half,full,t,q,np.zeros(8,dtype=np.int32),out,state)
                    actual=background_step(q[:,0],q[:,1],half,1,ts,snow,current,scenario,change,30.,300.,
                        extinction_efficiency=2.45,injection_type=injection_type,
                        inject_through_pbl=inject_pbl,remove_excess=remove_excess)
                    np.testing.assert_allclose(actual.mass_ratio,out[:,0],rtol=3e-12,atol=1e-20)
                    np.testing.assert_allclose(actual.number_ratio,out[:,1],rtol=3e-12,atol=1e-8)
                    np.testing.assert_allclose(actual.surface_dust_kg_m2,state[0],rtol=3e-14,atol=1e-14)
                    np.testing.assert_allclose(actual.diagnostics['injected_kg_m2'],state[1],rtol=3e-12,atol=1e-20)
                    np.testing.assert_allclose(actual.diagnostics['deposited_kg_m2'],state[3],rtol=3e-12,atol=1e-20)
                    cases.append(dict(injection_type=injection_type,inject_pbl=inject_pbl,
                        remove_excess=remove_excess,current_opacity=current,scenario_opacity=scenario,
                        temperature_surface_k=ts,snow_kg_m2=snow,seasonal_change=change,
                        max_mass_ratio_error=float(np.max(abs(actual.mass_ratio-out[:,0])))))
        directory=Path(tmp)/'optics';directory.mkdir()
        optics,optics_provenance=compile_optics_reference(source,directory,compiler)
        data=load_tables()
        half=np.array([0.,305.,610.]);pressure=np.zeros(7)
        pressure[2::2]=half;pressure[3::2]=.5*(half[:-1]+half[1:])
        for bands in (12,15):
            nir=5 if bands==12 else 8
            for effective in (1e-8,2e-6,100e-6):
                radius=np.clip(effective,1e-7,50e-6)
                number=1e-5*.75/np.pi/2500./(radius*np.exp(-.63676**2))**3
                q=np.zeros((2,4));q[:,0]=1e-5;q[:,1]=number
                outputs=[f(np.zeros((6,nb))) for nb in (7,7,7,nir,nir,nir)]
                v=data[f'dust_{bands}_sw'];ir=data[f'dust_{bands}_ir']
                optics.particles(0,2,nir,v.shape[1],v.shape[0],f(pressure),f(q),f(np.full(7,200.)),
                    f(v),f(ir),f(v),f(ir),f(data['moment_edges_m']),f(data['moment_radii_m']),
                    f(data['water_absorption_correction']),*outputs)
                area=number*np.pi*(radius*np.exp(-1.5*.63676**2))**2
                ir_index=3 if bands==12 else 6
                expected=np.array([np.sum(outputs[0][:,5]),
                    np.sum(outputs[3][:,ir_index]-outputs[4][:,ir_index])])/(area*610./3.72)
                actual=reference_efficiencies(effective,bands=bands)
                np.testing.assert_allclose(actual,expected,rtol=2e-11,atol=2e-13)
                cases.append(dict(case='fixed-radius-injection-efficiency',bands=bands,radius_m=effective,
                    max_efficiency_error=float(np.max(abs(actual-expected)))))
    root=Path(__file__).parents[1]
    paths=['scripts/validate_dust_background.py','scripts/validate_dust_lifting.py','scripts/native_reference.py',
        'scripts/validate_particle_optics.py','package/src/framework/physics/dust_lifting.py',
        'package/src/framework/physics/particle_optics.py','package/src/framework/physics/particle_optical_tables.npz']
    return dict(status='pass',reference=provenance,efficiency_reference=optics_provenance,cases=cases,
        sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths},
        limits='Native default injection and uniform sink with fixed-layer injection. '
            'The latter leaves i,j defined; the original PBL-injection sink uses stale out-of-bounds indices. '
            'Bottom sinks and the negative 1e-32 sink allowance use explicit conservative repairs, '
            'checked by independent material tests. Seasonal/local-time history and global integration are separate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} native background dust cases passed")
