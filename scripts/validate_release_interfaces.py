"""Audit real and unavailable physics interfaces in the pinned public release."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
from native_reference import extract_routines

BLOBS={
    'atmos_param_mars/mars_physics.F90':'ca0b7c0774b1b23cafaee219062c3bffa3d0c4cc',
    'atmos_param_mars/null_physics_mod.F90':'4ca7a58d46e31dd8fb281469bbbcd8d50071bba2',
    'build_run/compile.bash':'3a43a2fdb4e26dc0980e3e5bfaa5efbe0861a597',
}


def validate(source,compiler):
    for name,expected in BLOBS.items():
        raw=(source/name).read_bytes()
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected,name
    build=(source/'build_run/compile.bash').read_text()
    definitions=re.search(r'^cppDefs="([^"]+)"',build,re.M)
    assert definitions is not None
    definitions=definitions[1].split()
    assert definitions==['-DMARS_GCM','-DMARS_SURFACE','-DRELEASE']
    path=source/'atmos_param_mars/mars_physics.F90'
    flags=['-cpp','-E','-P',*definitions]
    release=subprocess.check_output([compiler,*flags,str(path)],text=True)
    assert 'use null_physics_mod, only: cloud_physics' in release
    assert 'photochem_driver' in release
    assert not re.search(r'^use\s+(cloud_physics_mod|photochem_mod|dust_source_mod|tagging_method_mod)',release,re.M)
    alternate=subprocess.check_output([compiler,*flags,'-Dfv3_turb',str(path)],text=True)
    calls=lambda text:set(re.findall(r'^\s*call\s+(\w+)',text,re.I|re.M))
    excluded=sorted(calls(alternate)-calls(release))
    assert {'surface_flux_2d','vert_turb_driver'}<=set(excluded)
    assert not re.search(r'^use\s+vert_(turb|diff)_driver_mod',release,re.M)
    names=['cloud_physics','cloud_physics_init','photochem_driver','dust_source_sink','dust_source_init']
    routines,_=extract_routines(source/'atmos_param_mars/null_physics_mod.F90',BLOBS['atmos_param_mars/null_physics_mod.F90'],names)
    cases=[]
    for name,body in zip(names,routines,strict=True):
        assert re.search(r'call\s+error_mesg\([^\n]*FATAL\)',body,re.I),name
        calls=re.findall(r'^\s*call\s+(\w+)',body,re.I|re.M)
        assert calls==['error_mesg'],(name,calls)
        cases.append(dict(interface=name,released_implementation='fatal placeholder',
            routine_sha256=hashlib.sha256(body.encode()).hexdigest()))
    root=Path(__file__).parents[1]
    return dict(status='pass',reference=dict(commit='ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26',
        source_git_blobs=BLOBS,preprocessor_flags=flags,
        preprocessed_sha256=hashlib.sha256(release.encode()).hexdigest(),
        compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0]),
        cases=cases,excluded_build_interfaces=dict(required_definition='fv3_turb',calls=excluded,
            alternate_preprocessed_sha256=hashlib.sha256(alternate.encode()).hexdigest()),
        sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('scripts/validate_release_interfaces.py','scripts/native_reference.py')},
        limits='This verifies unavailable release interfaces, not functioning physics. '
            'Bin-cloud microphysics, photochemistry and legacy binned dust-source equations are not in '
            'these released implementations. Tagging imports are excluded by RELEASE. '
            'Moment water, bulk water and moment dust have separate real modules and validations. '
            'Alternate FV3 turbulence calls require fv3_turb, which the pinned build does not define; '
            'this exclusion does not implement or validate those alternate schemes.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ames-source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();compiler=shutil.which('gfortran')
    if compiler is None: parser.error('gfortran is required')
    report=validate(args.ames_source,compiler)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(f"{len(report['cases'])} unavailable release interfaces verified")
