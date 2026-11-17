"""Guard test ownership, safe change selection and runner failure propagation."""
import importlib.util
import ast
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('ci', Path(__file__).resolve().parents[1] / 'ci.py')
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class SelectionTests(unittest.TestCase):
    def test_full_inventory_has_one_owner_per_file(self):
        files = ci.test_files()
        flat = [p for members in files.values() for p in members]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertTrue(all(files.values()))

    def test_unassigned_test_fails_instead_of_silently_skipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for paths in ci.test_files().values():
                for name in paths:
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.touch()
            (root / ci.GCM / 'test_new_physics.py').touch()
            with self.assertRaisesRegex(ValueError, 'unassigned=.*test_new_physics'):
                ci.test_files(root)

    def test_narrow_changes_select_relevant_modules(self):
        for path, expected in [
            ('docs/physics-parity.md', set()),
            ('ui/src/App.tsx', set()),
            ('apps/nautalis/jobs.py', {'runtime'}),
            ('experiments/reference-comparison/deploy.py', {'runtime'}),
            ('cli/models.py', {'cli'}),
            ('package/tests/gcm3d/test_spectral_flux.py', {'radiation'}),
        ]:
            with self.subTest(path=path):
                self.assertEqual(set(ci.plan([path])['selected']), expected)
        self.assertTrue(ci.plan(['ui/src/App.tsx'])['ui'])
        self.assertFalse(ci.plan(['docs/physics-parity.md'])['ui'])

    def test_physics_changes_include_downstream_consumers(self):
        for module, required in [
            ('spectral_flux', {'radiation', 'coupled', 'neural', 'cli', 'calibration'}),
            ('water_moments', {'particles', 'coupled', 'radiation', 'calibration'}),
            ('soil_frost', {'surface', 'coupled', 'particles', 'calibration'}),
        ]:
            with self.subTest(module=module):
                result = ci.plan([f'package/src/framework/physics/{module}.py'])
                self.assertTrue(required <= set(result['selected']))
                self.assertNotIn('runtime', result['selected'])

    def test_shared_or_unknown_changes_run_every_python_suite(self):
        for path in ('uv.lock', 'package/pyproject.toml', 'package/src/framework/gcm/restart.py',
                     'package/src/framework/physics/new_scheme.py', 'scripts/new_runner.py',
                     'package/tests/conftest.py', '.github/workflows/tests.yml', 'new-runtime-config'):
            with self.subTest(path=path):
                selected = set(ci.plan([path])['selected'])
                expected = set(ci.SUITES) - ({'runtime'} if path.startswith('package/src/') else set())
                self.assertEqual(selected, expected)

    def test_full_and_manual_selection(self):
        result = ci.plan([], full=True)
        self.assertEqual(set(result['selected']), set(ci.SUITES))
        self.assertTrue(result['ui'])
        for name in ci.SUITES:
            rows = ci.plan([], full=True, suite=name)['matrix']
            self.assertEqual({row['suite'] for row in rows}, {name})
            self.assertEqual([row['shard'] for row in rows], list(range(ci.SHARDS.get(name, 1))))
        with self.assertRaises(ValueError):
            ci.plan([], suite='typo')

    def test_integration_names_match_all_real_test_functions(self):
        source = ast.parse((ci.ROOT / ci.GCM / 'test_dinosaur_physics.py').read_text())
        functions = {node.name for node in source.body
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')}
        nodes = [ci.GCM + 'test_dinosaur_physics.py::' + name for name in sorted(functions)]
        groups = ci.integration_groups(nodes)
        self.assertTrue(all(groups.values()))
        self.assertCountEqual([node for members in groups.values() for node in members], nodes)
        configured = ['test_'+name for names in ci.INTEGRATION_GROUPS.values() for name in names.split()]
        self.assertCountEqual(configured, functions)
        rows = ci.plan([], full=True, suite='coupled')['matrix']
        self.assertEqual([row['name'] for row in rows], list(groups))

    def test_new_integration_case_requires_a_group(self):
        with self.assertRaisesRegex(ValueError, 'Unassigned integration test'):
            ci.integration_groups([ci.GCM+'test_dinosaur_physics.py::test_new_physics'])


class RunnerTests(unittest.TestCase):
    def test_runtime_keeps_reports_and_propagates_failure_after_other_files(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, 'ROOT', Path(temporary)), \
                patch.object(ci, 'test_files', return_value={'runtime': ['a.py', 'b.py']}), \
                patch.object(ci.subprocess, 'run', side_effect=[subprocess.CompletedProcess([], 1),
                    subprocess.CompletedProcess([], 0)]) as run:
            with self.assertRaises(SystemExit):
                ci.run_suite('runtime')
            self.assertEqual(run.call_count, 2)
            self.assertTrue(all(any(arg.startswith('--junitxml=') for arg in call.args[0])
                for call in run.call_args_list))

    def test_named_integration_groups_cover_every_node_once_in_isolated_processes(self):
        nodes = [ci.GCM + f'test_dinosaur_physics.py::test_{name}[{index}]'
            for names in ci.INTEGRATION_GROUPS.values() for name in names.split() for index in range(2)]
        seen = []
        def execute(command, **kwargs):
            if '-c' in command and 'import dinosaur, jax, h5netcdf' in command:
                return subprocess.CompletedProcess(command, 0)
            if '--collect-only' in command:
                return subprocess.CompletedProcess(command, 0, '\n'.join(nodes), '')
            selected = [arg for arg in command if arg in nodes]
            self.assertEqual(len(selected), 1)
            seen.extend(selected)
            return subprocess.CompletedProcess(command, 0)
        with tempfile.TemporaryDirectory() as temporary, patch.object(ci, 'ROOT', Path(temporary)), \
                patch.object(ci, 'test_files', return_value={'coupled': [ci.GCM+'test_dinosaur_physics.py']}), \
                patch.object(ci.subprocess, 'run', side_effect=execute):
            for index in range(ci.SHARDS['coupled']):
                ci.run_suite('coupled', index, ci.SHARDS['coupled'])
        self.assertCountEqual(seen, nodes)
        self.assertEqual(len(seen), len(set(seen)))


if __name__ == '__main__':
    unittest.main()
