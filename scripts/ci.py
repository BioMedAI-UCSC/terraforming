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
SUITES = (*MODULES, 'coupled', *DIRECTORIES)
INTEGRATION_GROUPS = {
    'Integration: radiation and temperature': '''
        thermal_relaxation_reaches_global_rollout_and_reverse_derivative
        simple_radiation_reaches_global_rollout_and_reverse_derivative
        scaled_surface_ir_preserves_restart_and_three_step_gradient
        fixed_spectral_date_routes_geometry_and_optical_clock_but_not_startup_ramp
        fixed_spectral_calendar_preserves_restart_and_three_step_clock_derivative
        spectral_ramp_keeps_interface_fluxes_and_stratosphere_unscaled
        zero_startup_ramp_updates_air_and_soil_like_zero_solar_forcing
        aerosol_control_requires_matching_provider_switch_and_cache
        aerosol_opacity_control_holds_native_channels_and_preserves_restart_gradient
        multistep_dinosaur_reverse_derivative_reaches_surface_state
        nonlte_callback_keeps_interface_fluxes_and_reports_heating_separately
        radiation_schedule_holds_kelvin_rate_and_stratosphere_until_source_clock_refresh
        held_radiation_retains_multistep_reverse_derivative
        prescribed_dust_scenario_and_size_optics_reach_multistep_reverse_gradient
    ''',
    'Integration: dust transport and cloud coupling': '''
        released_aerosol_mode_validity_applies_to_initialization_and_step
        independent_dust_transport_radiation_material_and_restart
        frozen_dust_retains_dry_transport_but_keeps_surface_sources_and_restart
        disabled_particle_scavenging_keeps_cloud_cores_material_restart_and_gradient
        independent_dust_lifting_coagulation_radiation_three_step_gradient
        multimode_cloud_dust_feedback_material_and_restart
        multimode_cloud_dust_multistep_reverse_derivative
        seasonal_spatial_dust_shares_preserve_material_restart_and_three_step_gradient
    ''',
    'Integration: water clouds': '''
        disabled_moment_clouds_keep_settling_and_water_budget_without_advancing_cloud_history
        moist_adjustment_global_water_budget_and_restart
        moist_adjustment_global_reverse_derivative
        reference_cloud_backend_forms_suspended_ice_and_closes_raw_mass
        reference_cloud_backend_multistep_reverse_derivative
        suspended_cloud_radiation_feedback_and_rollout_gradient
        bulk_water_backend_closes_material_and_restarts_full_cloud_feedback
        surface_water_exchange_uses_previous_step_coefficient
        bulk_water_backend_multistep_reverse_derivative
        moment_water_cloud_feedback_material_and_restart
        moment_cloud_multistep_reverse_derivative
    ''',
    'Integration: surface and CO2': '''
        disabled_co2_feedback_is_not_undone_by_global_inventory_projection
        disabled_co2_condensation_keeps_suspended_tracer_without_forming_clouds
        co2_controls_keep_three_step_reverse_derivatives
        surface_controls_keep_global_reverse_derivatives
        recalculated_pbl_heat_flux_changes_soil_and_preserves_restart_and_three_step_gradient
        dinosaur_backend_runs_cold_column_and_preserves_total_inventory
        convection_source_switch_runs_globally_and_restarts
        soil_weight_and_surface_phase_controls_preserve_three_step_derivative_and_restart
        native_cold_atmosphere_keeps_three_step_gradient_and_exact_restart
    ''',
    'Integration: transport and budgets': '''
        sigma_remap_conserves_energy_momentum_and_passive_material
        impossible_donor_mass_is_nonfinite_instead_of_clipped
        explicit_co2_remap_changes_only_co2_material_and_preserves_gas_closure
        sigma_remap_reverse_derivative
        sigma_remap_carries_suspended_mass_momentum_and_energy
        no_forcing_advances_dynamics_without_radiation_or_surface_updates
        layer_divergence_diffusion_preserves_material_and_other_fields
        layer_divergence_diffusion_rejects_invalid_profile
        layer_divergence_diffusion_keeps_full_physics_gradient_and_restart
        explicit_gas_global_species_budget_restart_and_reverse_derivative
        nonorographic_drag_global_timer_and_restart
        nonorographic_drag_global_multistep_reverse_derivative
        passive_transport_keeps_material_through_dynamics_and_column_physics
        stretched_sigma_global_gradient_and_invalid_geometry
        surface_drag_transfers_axial_momentum_to_ground_and_reports_projection
        exchange_energy_surface_derivative_uses_soil_node_capacity
        water_latent_energy_is_explicit_differentiable_and_used_at_all_repair_stages
    ''',
}
SHARDS = {'coupled': len(INTEGRATION_GROUPS)}


def test_files(root=ROOT):
    files = {suite: [] for suite in SUITES}
    for suite, names in MODULES.items():
        files[suite] = [GCM + 'test_' + name + '.py' for name in names.split()]
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
        elif path.startswith('examples/neural/'):
            selected.update(('neural', 'calibration'))
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
        name=f'Module: {name}' if name != 'coupled' else tuple(INTEGRATION_GROUPS)[index])
        for name in SUITES if name in selected for index in range(SHARDS.get(name, 1))]
    ui = full or any(path.startswith(('ui/', '.github/workflows/')) or path == 'scripts/ci.py' for path in paths)
    return dict(matrix=matrix, ui=ui if suite == 'all' else False,
        selected=sorted(selected), skipped=sorted(set(SUITES)-selected))


def integration_groups(nodes):
    groups = {name: [] for name in INTEGRATION_GROUPS}
    owners = {}
    for group, tests in INTEGRATION_GROUPS.items():
        for test in tests.split():
            name = 'test_' + test
            if name in owners:
                raise ValueError(f'Duplicate integration test: {name}')
            owners[name] = group
    for node in nodes:
        name = node.split('::', 1)[1].split('[', 1)[0]
        if name not in owners:
            raise ValueError(f'Unassigned integration test: {node}')
        groups[owners[name]].append(node)
    return groups


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
        groups = integration_groups(nodes)
        group = tuple(groups)[shard]
        batches = [[node] for node in groups[group]]
        if not batches:
            raise RuntimeError(f'Empty integration group: {group}')
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
            summary.write('## Test selection\n\nSelected: '+', '.join(row['name'] for row in result['matrix'])+'\n\n')
            summary.write('Skipped: '+', '.join(result['skipped'])+'\n\nUI build: '+str(result['ui'])+'\n')


if __name__ == '__main__':
    main()
