"""Prepare pinned runtimes on CPUs, then run the matched 100-sol comparison."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

PINS={
    'AmesGCM':('nasa','ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26'),
    'FMS':('NOAA-GFDL','71957e9d79fbd832c4e25ee452eb211669e0df1d'),
    'GFDL_atmos_cubed_sphere':('NOAA-GFDL','29654ea167575fa73b65f7f2ceb85f1682884ddd'),
    'atmos_drivers':('NOAA-GFDL','03d6c84128b3ea94cf4fcde992957f68722bff1c'),
    'atmos_phys':('NOAA-GFDL','46817a12232ed54d1a5d5a4394a695621f575812'),
    'mkmf':('NOAA-GFDL','76aa558474bbbf5ad8fa792b84efbb6ad0145980'),
}
INPUTS={
    'mars_thermal_inertia8_2011.nc':'43756c80ff8052a9123b71530942ed267b8c8af24e137bc3ba27b47b69c53476',
    'mars_TES_albedo8_new.nc':'4e4cc150a58be27bef0c3713d432789154b94e64c804ddeaf07eee7119e34fd7',
    'mars_sfc_emissivity8.nc':'23b53ed48a530679606f6fdf7b6fafbd0d92d022c507eb94dadb109b74dbab5d',
    'GRS_interp16.nc':'7c718ccec796fe1937b37f9ee9606cbf7720703c17eb0031076e9ef4fbda576f',
    'npcflag8.nc':'8acd88316770617d5f720dc8f57b96e845937ae1135fa4b84aa817f5d4711c7c',
    'DustScenario_MY30.nc':'1fefe32b06c8ae0b550a2aae0da44a83643b6cdcee377f15f5f53f4ed8177421',
    'mars_topo_mola16.nc':'59873fba00f78cc96a62c9f26127c789dccb1107504ad4d3fe6adb79e9e458f8',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle,'sha256').hexdigest()


def execute(arguments,**kwargs):
    print(' '.join(map(str,arguments)),flush=True)
    return subprocess.run(list(map(str,arguments)),check=True,**kwargs)


def source_hashes(directory):
    return {str(path.relative_to(directory)):digest(path) for path in sorted(directory.rglob('*'))
        if path.is_file() and not {'.git','__pycache__'}&set(path.relative_to(directory).parts)}


def checkout_repository(target,repository,pin):
    if not (target/'.git').exists(): execute(['git','init',target])
    head=subprocess.run(['git','-C',target,'rev-parse','--verify','HEAD'],capture_output=True,text=True)
    if head.returncode:
        execute(['git','-C',target,'fetch','--depth=1',repository,pin])
        execute(['git','-C',target,'checkout','--detach','FETCH_HEAD'])
    actual=subprocess.check_output(['git','-C',target,'rev-parse','HEAD'],text=True).strip()
    if actual!=pin: raise ValueError(f'{target.name}: cached source revision differs')


def fetch_sources(directory):
    directory.mkdir(parents=True,exist_ok=True)
    for name,(owner,pin) in PINS.items():
        target=directory/name
        checkout_repository(target,f'https://github.com/{owner}/{name}.git',pin)
    nasa=directory/'AmesGCM'
    for name,patch in (('FMS','srcFMS.patch'),('GFDL_atmos_cubed_sphere','srcGFDL_acs.patch'),
                        ('atmos_drivers','srcatmos_drivers.patch'),('atmos_phys','srcatmos_phys.patch')):
        target=directory/name;path=nasa/'patches'/patch
        check=subprocess.run(['git','apply','--check',str(path)],cwd=target,capture_output=True)
        if check.returncode==0: execute(['git','apply',path],cwd=target)
        else: execute(['git','apply','--reverse','--check',path],cwd=target)
    path=nasa/'atmos_param_mars/Rad/rtmod_mgcm.F90'
    before=path.read_text()
    after=re.sub(r'(?m)^(\s*#)(IFNDEF|ENDIF)\b',lambda match:match[1]+match[2].lower(),before)
    path.write_text(after)
    return nasa


def prepare_native(cache):
    nasa=fetch_sources(cache/'src');execution=cache/'exec';execution.mkdir(exist_ok=True)
    path=cache/'src/FMS/diag_manager/diag_manager.F90'
    before=path.read_text()
    old='if (output_fields(out_num)%last_output > time) CYCLE'
    guarded='if (.not.output_fields(out_num)%static) then\n'+old+'\nendif'
    # Static fields have no initialized dynamic output clock.
    if guarded not in before:
        if before.count(old)!=1: raise ValueError('native static diagnostic time gate differs')
        path.write_text(before.replace(old,guarded))
    for name,expected in INPUTS.items():
        if digest(nasa/'data'/name)!=expected: raise ValueError(f'{name}: native release input differs')
    query=lambda tool,option:subprocess.check_output([tool,option],text=True).strip()
    template=(nasa/'build_run/bin/mkmf.template.gnu').read_text()
    template=template.replace('CPPFLAGS = -I$(NETCDF)/include -I$(MPI_INC)',
        'CPPFLAGS = '+query('nf-config','--fflags')+' '+query('nc-config','--cflags'))
    template=template.replace('CC = gcc','CC = mpicc').replace('FC = gfortran','FC = mpifort').replace('LD = gfortran','LD = mpifort')
    template=template.replace('LDFLAGS = -L$(NETCDF)/lib -L$(MPI_LIB) -lnetcdff -lnetcdf -fopenmp -lmpi',
        'LDFLAGS = '+query('nf-config','--flibs')+' '+query('nc-config','--libs')+' -fopenmp')
    template=template.replace('CFLAGS = -D__GFORTRAN $(CPPFLAGS) $(MARS_CPP) $(FFLAGS)',
        'CFLAGS = -D__GFORTRAN $(CPPFLAGS) $(MARS_CPP) -O2'+
        (' -DHAVE_GETTID' if sys.platform=='linux' else ''))
    (execution/'mkmf.template.gnu').write_text(template)
    script=(nasa/'build_run/compile.bash').read_text().replace('platform="intel"','platform="gnu"')
    script=script.replace('source "$cwd/loadmods.bash"','')
    script=script.replace('/nasa/netcdf/4.4.1.1_mpt',query('nf-config','--prefix'))
    for before,after in (('*cmip*','cmip'),('*fv_cmp*','fv_cmp'),('*/docs/*','/docs/'),
                         ('*/monin_obukhov/monin_obukhov*','/monin_obukhov/monin_obukhov')):
        script=script.replace(before,after)
    # list_paths deduplicates basenames before filtering; docs contain another fv_eta.F90.
    script=script.replace('"$execdir/atmos_cubed/pathnames_atmos_cubed" "$src_fv3"',
        '"$execdir/atmos_cubed/pathnames_atmos_cubed" "$src_fv3/model" "$src_fv3/tools" "$src_fv3/GFDL_tools"')
    script=script.replace('make "${executable##*/}"','make -j4 "${executable##*/}"')
    (execution/'compile-local.bash').write_text(script)
    execute(['/bin/bash',execution/'compile-local.bash'],cwd=execution)
    binary=execution/'exec.gnu.mars3.2/FMS_MARS.x'
    return dict(executable=str(binary),executable_sha256=digest(binary),
        source_sha256=source_hashes(cache/'src'),
        compiler=query('gfortran','--version'),mpi=query('mpirun','--version'),
        adaptations='GNU compiler and paths, MPI/netCDF linkage, source-list regular expressions, lowercase four CPP directives; static diagnostic fields bypass the undefined dynamic output-time gate; Fortran physics statements unchanged')


def prepare_dinosaur(cache,source):
    environment=cache/'venv';python=environment/'bin/python'
    if not python.exists(): execute(['uv','venv','--python','3.12',environment])
    package=cache/'framework'
    if not package.exists(): shutil.copytree(source/'package',package,
        ignore=shutil.ignore_patterns('__pycache__','*.egg-info','build','.pytest_cache'))
    execute(['uv','pip','install','--python',python,'-c',source/'apps/mars-calibration/constraints.txt',
        f'{package}[gcm3d]','jax[cuda12]==0.11.0','xarray','h5netcdf'])
    execute(['uv','pip','check','--python',python])
    installed=Path(subprocess.check_output([python,'-c',
        'import src;from pathlib import Path;print(Path(src.__file__).parent)'],text=True).strip())
    return dict(python=str(python),installed_source=str(installed),
        installed_source_sha256=source_hashes(installed),
        packages=subprocess.check_output(['uv','pip','freeze','--python',python],text=True))


def native_inputs(nasa,output,resolution,sols):
    """Resolve the released cloud run's tables without executing its PBS script."""
    if resolution not in ('C24','C48'): raise ValueError('native resolution must be C24 or C48')
    cells=int(resolution[1:]);layout='1,1' if cells==24 else '1,2'
    values=dict(name='matched-reference',TILE_LAYOUT=layout,PREF='705.',NPX=str(cells+1),
        NPZ='56',NPZ_RST='0',NKS=str(3*cells//24),NNS='8',NORD='3' if cells==24 else '2',
        NCX=str(cells),days=str(sols),DTA='231')
    text=(nasa/'build_run/fms_mars_clouds_v3.2').read_text()
    written=set()
    for name,body in re.findall(r'^cat > (\S+) <<\\?EOF\n(.*?)^\\?EOF\s*$',text,re.M|re.S):
        if name not in ('diag_table','field_table','input.nml.unexpanded'): continue
        body=re.sub(r'\$\{?([A-Za-z_]\w*)\}?',lambda match:values[match[1]],body)
        if name=='input.nml.unexpanded':
            name='input.nml';body=body.replace(' &test_case_nml',' /\n\n &test_case_nml')
        (output/name).write_text(body)
        written.add(name)
    if written!={'diag_table','field_table','input.nml'}:
        raise ValueError('released cloud script does not contain all three input tables')
    for name in ('INPUT','RESTART'): (output/name).mkdir()
    for path in (nasa/'data').iterdir():
        if path.is_file(): (output/'INPUT'/path.name).symlink_to(path)
    aliases={'mars_topo.nc':'mars_topo_mola16.nc','thermal_inertia.nc':'mars_thermal_inertia8_2011.nc',
        'albedo.nc':'mars_TES_albedo8_new.nc','sfc_emiss.nc':'mars_sfc_emissivity8.nc',
        'dust_cycle.nc':'DustScenario_MY30.nc'}
    for name,target in aliases.items(): (output/'INPUT'/name).symlink_to(nasa/'data'/target)
    return 6 if cells==24 else 12


def validate_prepared(cache,details,framework,revision):
    if details['revision']!=revision or details['framework']!=framework or details['inputs_sha256']!=INPUTS:
        raise ValueError('prepared runtime identity differs from the requested comparison')
    for name,expected in INPUTS.items():
        if digest(cache/'inputs'/name)!=expected: raise ValueError(f'{name}: cached input changed')
    if framework=='native':
        if digest(details['executable'])!=details['executable_sha256'] or source_hashes(cache/'src')!=details['source_sha256']:
            raise ValueError('cached native executable or source changed')
    elif source_hashes(Path(details['installed_source']))!=details['installed_source_sha256']:
        raise ValueError('cached installed framework source changed')


def main(framework,resolution):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--inputs',type=Path,default=Path('/inputs/reference'))
    parser.add_argument('--sols',type=int,default=100)
    args=parser.parse_args()
    supported={'native':('C24','C48'),'dinosaur':('T21','T42')}
    if framework not in supported or resolution not in supported[framework]:
        parser.error('unsupported framework or resolution')
    if args.sols<1: parser.error('sols must be positive')
    gpu=os.environ.get('EXPERIMENT_GPU')=='1'
    if gpu and (args.prepare_only or framework=='native'):
        parser.error('runtime preparation and native Fortran experiments require CPU-only jobs')
    source=Path(os.environ['EXPERIMENT_SOURCE_DIR']);output=Path(os.environ['EXPERIMENT_OUTPUT_DIR'])
    source_metadata=json.loads((Path(os.environ['EXPERIMENT_RUN_DIR'])/'source.json').read_text())
    cache=Path('/work/reference-runtime')/source_metadata['revision']/framework
    cache.mkdir(parents=True,exist_ok=True);ready=cache/'ready.json'
    with (cache/'.setup.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not ready.exists():
            if not args.prepare_only: raise RuntimeError('prepare this exact source on a CPU job first')
            inputs=cache/'inputs';inputs.mkdir(exist_ok=True)
            for name,expected in INPUTS.items():
                path=args.inputs/name
                if digest(path)!=expected: raise ValueError(f'{name}: input hash differs')
                shutil.copy2(path,inputs/name)
            details=prepare_native(cache) if framework=='native' else prepare_dinosaur(cache,source)
            temporary=ready.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(framework=framework,revision=source_metadata['revision'],
                inputs_sha256=INPUTS,**details),indent=2)+'\n')
            temporary.replace(ready)
        details=json.loads(ready.read_text())
        validate_prepared(cache,details,framework,source_metadata['revision'])
    (output/'runtime.json').write_text(json.dumps(details,indent=2)+'\n')
    if args.prepare_only: return
    env=dict(os.environ,OMP_NUM_THREADS='1' if framework=='native' else '4',OPENBLAS_NUM_THREADS='1',
        MPLCONFIGDIR=str(cache/'matplotlib'),JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if framework=='native':
        nasa=cache/'src/AmesGCM'
        for name in INPUTS:
            if digest(nasa/'data'/name)!=INPUTS[name]: raise ValueError(f'{name}: native release input differs')
        ranks=native_inputs(nasa,output,resolution,args.sols)
        invocation=['mpirun','--host',f'localhost:{ranks}','--bind-to','none',
            '--mca','btl_vader_single_copy_mechanism','none','-np',str(ranks),details['executable']]
    else:
        if os.environ.get('EXPERIMENT_GPU')!='1': raise RuntimeError('Dinosaur climate comparison requires a GPU job')
        env.update(JAX_PLATFORMS='cuda',XLA_PYTHON_CLIENT_MEM_FRACTION='.45')
        execute([details['python'],'-c','import jax; assert any(d.platform=="gpu" for d in jax.devices()), jax.devices()'],env=env)
        invocation=[details['python'],source/'scripts/run_gcm3d_ablation.py',cache/'inputs',
            '--config','reference','--mola-path',cache/'inputs/mars_topo_mola16.nc',
            '--truncation',resolution,'--layers','56','--dt','231','--sols',str(args.sols),
            '--chunk-sols','5','--cooldown-seconds','0','--output-dir',output/'climate']
    started=time.perf_counter();execute(invocation,cwd=output if framework=='native' else source,env=env)
    (output/'completed.json').write_text(json.dumps(dict(framework=framework,resolution=resolution,
        sols=args.sols,dt_seconds=231,timing_scope='concurrent experiment throughput; not an isolated speed measurement',wall_seconds=time.perf_counter()-started,
        limits='Same released physical setup and input hashes; FV3 hybrid/cubed-sphere and Dinosaur sigma/spectral discretizations differ.'),indent=2)+'\n')
