"""Select module CI jobs from changed paths and run each test exactly once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
GCM = 'package/tests/gcm3d/'
MODULES = {
    'dynamics': 'architecture benchmarks body co2_projection '
        'coordinates dynamics hydrostatic_initialization maps mars_clocks orbit physics reference_constants '
        'restart specs submission topography vertical_grid',
    'radiation': 'collision_absorption correlated_k extended_radiation frost_radiation '
        'nonlte_cooling optical_depth particle_optics radiation_grid radiative_transfer scattering_transfer '
        'simple_radiation solar_geometry spectral_flux',
    'surface': 'atmospheric_condensation boundary_layer co2_clouds co2_exchange co2_microphysics column_physics dry_convection '
        'legacy_convection moist_convection nonorographic_drag pbl_regressions rayleigh_drag '
        'saturation_pressure soil_frost surface_maps terrain_drag thermal_relaxation',
    'particles': 'bulk_water dust dust_background dust_coagulation dust_control dust_lifting '
        'dust_scenario particle_sedimentation prescribed_dust tracer_transport water_moments water_surface',
    'neural': 'control_metrics neural_framework neural_temp radiation_training_budget',
}
DIRECTORIES = {
    'legacy': ('package/tests/engine', 'package/tests/interventions', 'package/tests/mars'),
    'cli': ('cli/tests',),
    'calibration': ('apps/mars-calibration/tests',),
    'runtime': ('apps/nautalis',),
}
SUITES = (*MODULES, 'aerosols', 'coupled', *DIRECTORIES)
SHARDS = {'coupled': 4}
RADIATION = set('correlated_k scattering_transfer simple_radiation radiation_grid optical_depth '
    'spectral_flux radiative_transfer collision_absorption nonlte_cooling particle_optics solar_geometry twostream'.split())
SURFACE = set('boundary_layer co2_condensation co2_microphysics co2_clouds soil_frost dry_convection '
    'legacy_convection moist_convection nonorographic_drag rayleigh_drag saturation_pressure '
    'terrain_drag thermal_relaxation hydrostatic_initialization'.split())
PARTICLES = set('particle_sedimentation tracer_transport prescribed_dust dust_lifting dust_control '
    'dust_coagulation water_bulk water_moments water_surface'.split())


def test_files(root=ROOT):
    files = {suite: [] for suite in SUITES}
    for suite, names in MODULES.items():
        files[suite] = [GCM + 'test_' + name + '.py' for name in names.split()]
    files['aerosols'] = sorted(str(p.relative_to(root)) for p in (root / GCM).glob('test_aerosol*.py'))
    files['coupled'] = [GCM + 'test_dinosaur_physics.py']
    files['legacy'] = [GCM + 'test_terraforming_ode.py', GCM + 'test_terraforming_ode_seasonal.py']
    for suite, directories in DIRECTORIES.items():
        files[suite].extend(sorted(str(p.relative_to(root)) for directory in directories
            for p in (root / directory).rglob('test_*.py')))
    assigned = [path for paths in files.values() for path in paths]
    discovered = {str(p.relative_to(root)) for directory in
        ('package/tests', 'cli/tests', 'apps/mars-calibration/tests', 'apps/nautalis')
        for p in (root / directory).rglob('test_*.py')}
    if len(assigned) != len(set(assigned)) or set(assigned) != discovered:
        raise ValueError(f'Test inventory mismatch: unassigned={sorted(discovered-set(assigned))}; '
            f'missing={sorted(set(assigned)-discovered)}; duplicates={len(assigned)-len(set(assigned))}')
    return files


def affected_suites(paths, files):
    selected = set()
    owners = {path: suite for suite, members in files.items() for path in members}
    numerical = set(SUITES) - {'runtime'}
    for path in paths:
        if path in owners:
            selected.add(owners[path])
        elif path.startswith(('scripts/ci.py', 'scripts/tests/', '.github/workflows/', '.github/actions/')) or \
                Path(path).name in {'uv.lock', 'pyproject.toml', 'pytest.ini', 'conftest.py', '.python-version'}:
            return set(SUITES)
        elif '/tests/' in path or path.startswith('apps/nautalis/test_'):
            return set(SUITES)
        elif path.startswith(('apps/nautalis/', 'experiments/', 'scripts/run_reference_comparison.py',
                              'scripts/reconstruct_native_grid.py')):
            selected.add('runtime')
        elif path.startswith('ui/'):
            continue
        elif path.startswith('cli/'):
            selected.add('cli')
        elif path.startswith('package/src/interventions/aerosols/'):
            selected.update(('aerosols', 'cli'))
        elif path.startswith('package/src/framework/neural/') or path.startswith('examples/neural/'):
            selected.update(('neural', 'calibration'))
        elif path.startswith('package/src/framework/physics/'):
            stem = Path(path).stem
            if stem in RADIATION or Path(path).suffix == '.npz' and stem.startswith(('co2_', 'collision_', 'particle_optical', 'nonlte_')):
                selected.update(('radiation', 'particles', 'coupled', 'neural', 'aerosols', 'calibration', 'cli'))
            elif stem in SURFACE:
                selected.update(('surface', 'particles', 'coupled', 'calibration', 'cli'))
            elif stem in PARTICLES:
                selected.update(('particles', 'radiation', 'coupled', 'aerosols', 'calibration', 'cli'))
            else:
                selected.update(numerical)
        elif path.startswith('package/src/'):
            selected.update(numerical)
        elif path.startswith(('apps/mars-calibration/', 'paper/arxiv-2610.04081/')):
            selected.add('calibration')
        elif path.startswith('scripts/'):
            # Script tests span runners, maps, calibration and native adapters.
            selected.update(numerical | {'runtime'})
        elif path.startswith(('docs/', 'iclr-results/', 'licenses/')) or \
                Path(path).suffix.lower() in {'.md', '.rst', '.png', '.svg', '.jpg'}:
            continue
        else:
            return set(SUITES)
    return selected


def plan(paths, full=False, suite='all'):
    files = test_files()
    if suite != 'all' and suite not in SUITES:
        raise ValueError(f'Unknown suite: {suite}')
    selected = {suite} if suite != 'all' else set(SUITES) if full else affected_suites(paths, files)
    matrix = [dict(suite=name, shard=index, shards=SHARDS.get(name, 1),
        name=name if name not in SHARDS else f'{name} ({index+1}/{SHARDS[name]})')
        for name in SUITES if name in selected for index in range(SHARDS.get(name, 1))]
    ui = full or any(path.startswith(('ui/', '.github/workflows/')) or path == 'scripts/ci.py' for path in paths)
    return dict(matrix=matrix, ui=ui if suite == 'all' else False,
        selected=sorted(selected), skipped=sorted(set(SUITES)-selected))


def run_suite(suite, shard=0, shards=1, slow=False, collect_only=False):
    files = test_files()[suite]
    if shards != SHARDS.get(suite, 1) or not 0 <= shard < shards:
        raise ValueError('Invalid suite shard')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    if suite != 'runtime':
        # Check dependencies without retaining a JAX runtime beside the test process.
        subprocess.run([sys.executable, '-c', 'import dinosaur, jax, h5netcdf'], cwd=ROOT, check=True)
    report_dir = ROOT / 'test-results'
    report_dir.mkdir(exist_ok=True)
    command = [sys.executable, '-m', 'pytest', '-c', 'package/pyproject.toml', '--rootdir=.', '-p', 'no:cacheprovider']
    if not slow:
        command += ['-m', 'not slow']
    if suite == 'coupled':
        result = subprocess.run([*command, '--collect-only', '-q', *files], cwd=ROOT,
            capture_output=True, text=True, check=True)
        nodes = [line for line in result.stdout.splitlines() if line.startswith(GCM) and '::' in line]
        if not nodes or len(nodes) != len(set(nodes)):
            raise RuntimeError(f'Invalid coupled test collection:\n{result.stdout}')
        batches = [[node] for node in nodes[shard::shards]]
        if collect_only:
            print('\n'.join(batch[0] for batch in batches))
            return
    else:
        # Release compiled JAX executables between modules, rather than retain them for the full suite.
        batches = [[path] for path in files]
    if not batches:
        raise RuntimeError(f'Empty test suite: {suite}')
    failed = False
    for index, batch in enumerate(batches):
        args = ['--collect-only', '-q'] if collect_only else ['-v', '--durations=10',
            f'--junitxml={report_dir / (suite+"-"+str(shard)+"-"+str(index)+".xml")}']
        result = subprocess.run([*command, *args, *batch], cwd=ROOT)
        failed |= result.returncode != 0
    if failed:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--base')
    parser.add_argument('--head', default='HEAD')
    parser.add_argument('--full', action='store_true')
    parser.add_argument('--suite', choices=('all', *SUITES), default='all')
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--slow', action='store_true')
    parser.add_argument('--collect-only', action='store_true')
    args = parser.parse_args()
    if not args.plan:
        if args.suite == 'all':
            parser.error('choose one --suite to run; use --plan --full to list all jobs')
        run_suite(args.suite, args.shard, args.shards, args.slow, args.collect_only)
        return
    paths = []
    full = args.full or not args.base
    if not full:
        diff = subprocess.run(['git', 'diff', '--no-renames', '--name-only', '-z', args.base, args.head],
            cwd=ROOT, capture_output=True, check=True)
        paths = diff.stdout.decode().rstrip('\0').split('\0') if diff.stdout else []
    result = plan(paths, full, args.suite)
    print(json.dumps(result, indent=2))
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as output:
            output.write('matrix='+json.dumps({'include': result['matrix']})+'\n')
            output.write('run_tests='+str(bool(result['matrix'])).lower()+'\n')
            output.write('ui='+str(result['ui']).lower()+'\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as summary:
            summary.write('## Test selection\n\nSelected: '+', '.join(result['selected'])+'\n\n')
            summary.write('Skipped: '+', '.join(result['skipped'])+'\n\nUI build: '+str(result['ui'])+'\n')


if __name__ == '__main__':
    main()
