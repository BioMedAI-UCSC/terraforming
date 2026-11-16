"""Checks for frozen inputs, native setup recovery and matched run configuration."""
import importlib.util
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('comparison',ROOT/'scripts/run_reference_comparison.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)
spec=importlib.util.spec_from_file_location('comparison_deploy',ROOT/'experiments/reference-comparison/deploy.py')
deploy=importlib.util.module_from_spec(spec);spec.loader.exec_module(deploy)


class ComparisonTests(unittest.TestCase):
    def test_selected_framework_keeps_parallel_resolutions_without_relaunching_other_model(self):
        for framework in ('native','dinosaur'):
            args=SimpleNamespace(name='matched-corrected',ref='a'*40,phase='prepare',namespace='razvanlab',
                input_node='input-node',input_pvc='input-data',context='nautilus',framework=framework)
            for phase in ('prepare','run'):
                args.phase=phase
                body=deploy.comparison_manifests(args)
                jobs=[item for item in body['items'] if item['kind']=='Job']
                self.assertEqual(len(jobs),1)
                job=jobs[0]
                self.assertIn('-'+framework+'-',job['metadata']['name'])
                arguments=job['spec']['template']['spec']['containers'][0]['args']
                self.assertEqual(arguments[arguments.index('--parallel')+1],'2')
                self.assertEqual(sum(arg.startswith('experiments/reference-comparison/') for arg in arguments),2)
                self.assertNotIn('activeDeadlineSeconds',job['spec'])

    def test_native_build_search_excludes_documentation_before_basename_deduplication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);nasa=root/'src/AmesGCM';(nasa/'build_run/bin').mkdir(parents=True)
            (nasa/'build_run/bin/mkmf.template.gnu').write_text('CFLAGS = -D__GFORTRAN $(CPPFLAGS) $(MARS_CPP) $(FFLAGS)\n')
            line='${listpaths} -o "$execdir/atmos_cubed/pathnames_atmos_cubed" "$src_fv3"'
            (nasa/'build_run/compile.bash').write_text(line+'\n')
            def build(arguments,**kwargs):
                script=Path(arguments[1]).read_text()
                self.assertIn('"$src_fv3/model" "$src_fv3/tools" "$src_fv3/GFDL_tools"',script)
                self.assertNotIn('"$src_fv3"',script)
                # Directory order must not let a duplicate documentation example hide model code.
                tree=root/'src/GFDL_atmos_cubed_sphere'
                for directory in ('docs/example','model','tools','GFDL_tools'):
                    (tree/directory).mkdir(parents=True,exist_ok=True)
                (tree/'docs/example/fv_eta.F90').write_text('documentation example')
                (tree/'tools/fv_eta.F90').write_text('model implementation')
                listing=root/'listpaths'
                listing.write_text('#!'+sys.executable+'\n'
                    'import sys\nfrom pathlib import Path\npaths={}\n'
                    'for directory in sys.argv[3:]:\n'
                    ' for p in sorted(Path(directory).rglob("*.F90")): paths.setdefault(p.name,p)\n'
                    'Path(sys.argv[2]).write_text("\\n".join(map(str,paths.values())))\n')
                listing.chmod(0o755)
                target=root/'exec/atmos_cubed';target.mkdir(parents=True)
                subprocess.run(['/bin/bash',arguments[1]],check=True,env=dict(os.environ,
                    listpaths=str(listing),execdir=str(root/'exec'),src_fv3=str(tree)))
                selected=(target/'pathnames_atmos_cubed').read_text()
                self.assertEqual(Path(selected).read_text(),'model implementation')
                binary=root/'exec/exec.gnu.mars3.2/FMS_MARS.x';binary.parent.mkdir(parents=True)
                binary.write_bytes(b'fixture')
            with patch.object(comparison,'fetch_sources',return_value=nasa),patch.object(comparison,'INPUTS',{}),\
                 patch.object(comparison.subprocess,'check_output',return_value='fixture'),\
                 patch.object(comparison,'execute',side_effect=build):
                comparison.prepare_native(root)

    def test_deployment_has_parallel_cpu_preparation_and_separate_cpu_gpu_compute(self):
        args=SimpleNamespace(name='matched-v1',ref='a'*40,phase='prepare',namespace='razvanlab',
            input_node='input-node',input_pvc='input-data',context='nautilus')
        prepared=deploy.comparison_manifests(args)
        jobs=[item for item in prepared['items'] if item['kind']=='Job']
        self.assertEqual(len(jobs),2)
        for job in jobs:
            pod=job['spec']['template']['spec'];container=pod['containers'][0]
            self.assertEqual(pod['nodeSelector'],{'kubernetes.io/hostname':'input-node'})
            self.assertNotIn('nvidia.com/a100',container['resources']['requests'])
            self.assertTrue(container['volumeMounts'][-1]['readOnly'])
            self.assertEqual(container['args'][container['args'].index('--parallel')+1],'2')
            self.assertEqual(container['args'][-1],'--prepare-only')
        args.phase='run';running=deploy.comparison_manifests(args)
        self.assertEqual(len(running['items']),2)
        cpu,gpu=[job['spec']['template']['spec'] for job in running['items']]
        self.assertNotIn('nvidia.com/a100',cpu['containers'][0]['resources']['requests'])
        native_env={item['name']:item['value'] for item in cpu['containers'][0]['env']}
        self.assertEqual(native_env['OMPI_MCA_btl_vader_single_copy_mechanism'],'none')
        self.assertEqual(native_env['OMPI_MCA_rmaps_base_oversubscribe'],'1')
        self.assertEqual(gpu['containers'][0]['resources']['requests']['nvidia.com/a100'],'1')
        self.assertEqual(gpu['priorityClassName'],'opportunistic')
        for job in running['items']:
            self.assertNotIn('activeDeadlineSeconds',job['spec'])
            self.assertNotIn('activeDeadlineSeconds',job['spec']['template']['spec'])
            self.assertNotIn('ttlSecondsAfterFinished',job['spec'])
            self.assertEqual(job['spec']['template']['spec']['containers'][0]['args'][-2:],['--sols','100'])
        args.phase='prepare';args.pvc_prefix='previous-volumes'
        reused=deploy.comparison_manifests(args)
        self.assertEqual([item['metadata']['name'] for item in reused['items'] if item['kind']=='PersistentVolumeClaim'],
                         ['previous-volumes-native','previous-volumes-dinosaur'])
        args.ref='main'
        with self.assertRaisesRegex(ValueError,'immutable'): deploy.comparison_manifests(args)

    def test_compute_submission_requires_successful_preparation_with_matching_identity(self):
        args=SimpleNamespace(name='mars-reference-100sol-v3',ref='a'*40,phase='run',namespace='razvanlab',
            input_node='input-node',input_pvc='input-data',context='nautilus')
        body=deploy.comparison_manifests(args);saved=copy.deepcopy(body['items'])
        for job in saved: job['status']={'succeeded':1}
        def check():
            with patch.object(deploy.subprocess,'check_output',side_effect=[json.dumps(job) for job in saved]) as query:
                deploy.require_prepared(args,body)
                self.assertEqual([call.args[0][-3] for call in query.call_args_list],
                    ['mars-reference-100sol-v3-native-prepare','mars-reference-100sol-v3-dinosaur-prepare'])
        check();saved[0]['status']['succeeded']=0
        with self.assertRaisesRegex(RuntimeError,'has not succeeded'): check()
        saved[0]['status']['succeeded']=1;saved[0]['metadata']['annotations']['terraforming/source-revision']='b'*40
        with self.assertRaisesRegex(RuntimeError,'source or image differs'): check()

    def test_partial_checkout_resumes_without_replacing_a_different_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'repository';source.mkdir()
            def git(*arguments):
                return subprocess.check_output(['git',*map(str,arguments)],text=True,stderr=subprocess.DEVNULL).strip()
            git('init',source);git('-C',source,'config','user.name','Test')
            git('-C',source,'config','user.email','test@example.invalid')
            (source/'physics.f90').write_text('! original physics\n')
            git('-C',source,'add','.');git('-C',source,'commit','-m','reference')
            pin=git('-C',source,'rev-parse','HEAD')
            target=root/'cache';git('init',target)
            comparison.checkout_repository(target,str(source),pin)
            self.assertEqual((target/'physics.f90').read_text(),'! original physics\n')
            comparison.checkout_repository(target,str(source),pin)
            with self.assertRaisesRegex(ValueError,'revision differs'):
                comparison.checkout_repository(target,str(source),'0'*40)

    def test_native_configuration_changes_resolution_and_keeps_the_physical_protocol(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);nasa=root/'release';(nasa/'build_run').mkdir(parents=True)
            (nasa/'data').mkdir()
            for name in comparison.INPUTS: (nasa/'data'/name).write_bytes(b'fixture')
            (nasa/'build_run/fms_mars_clouds_v3.2').write_text('''exit 91
cat > diag_table <<EOF
$name
EOF
cat > field_table <<EOF
"TRACER", "atmos_mod", "vap_mass_mom"
EOF
cat > input.nml.unexpanded <<\\EOF
 &fv_core_nml
 layout=$TILE_LAYOUT, npx=$NPX, npy=$NPX, npz=$NPZ,
 p_ref=$PREF, dry_mass=$PREF, k_split=$NKS, n_split=$NNS, nord=$NORD,
 /
 &main_nml
 days=$days, dt_atmos=$DTA,
 /
 &mars_physics_nml
 do_co2_condensation_cycle=T,
 /
 &diag_manager_nml
 issue_oor_warnings=F,
 &test_case_nml
 test_case=11
 /
\\EOF
''')
            for resolution,ranks,cells,layout,split in (('C24',6,25,'1,1',3),('C48',12,49,'1,2',6)):
                output=root/resolution;output.mkdir()
                self.assertEqual(comparison.native_inputs(nasa,output,resolution,100),ranks)
                text=(output/'input.nml').read_text()
                self.assertIn(f'npx={cells}',text);self.assertIn(f'layout={layout}',text)
                self.assertIn(f'k_split={split}',text);self.assertIn('npz=56',text)
                self.assertIn('days=100, dt_atmos=231',text)
                self.assertIn('do_co2_condensation_cycle=T',text)
                self.assertIn(' /\n\n &test_case_nml',text)
                self.assertNotIn('$',text)
                self.assertEqual((output/'INPUT/mars_topo.nc').resolve(),(nasa/'data/mars_topo_mola16.nc').resolve())
                self.assertEqual((output/'INPUT/dust_cycle.nc').resolve(),(nasa/'data/DustScenario_MY30.nc').resolve())
            with self.assertRaisesRegex(ValueError,'C24 or C48'):
                comparison.native_inputs(nasa,root/'unsupported','C36',100)

    def test_prepared_native_cache_rejects_changed_inputs_code_and_executable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'inputs').mkdir();(root/'src').mkdir()
            data=root/'inputs/map.nc';data.write_bytes(b'physical input')
            source=root/'src/physics.f90';source.write_text('! source\n')
            binary=root/'FMS_MARS.x';binary.write_bytes(b'executable fixture')
            inputs={'map.nc':comparison.digest(data)}
            details=dict(framework='native',revision='frozen',inputs_sha256=inputs,
                executable=str(binary),executable_sha256=comparison.digest(binary),
                source_sha256=comparison.source_hashes(root/'src'))
            with patch.object(comparison,'INPUTS',inputs):
                check=lambda:comparison.validate_prepared(root,details,'native','frozen')
                check();data.write_bytes(b'changed input')
                with self.assertRaisesRegex(ValueError,'cached input changed'): check()
                data.write_bytes(b'physical input');source.write_text('! changed physics\n')
                with self.assertRaisesRegex(ValueError,'executable or source changed'): check()
                source.write_text('! source\n');binary.write_bytes(b'changed executable')
                with self.assertRaisesRegex(ValueError,'executable or source changed'): check()

    def test_installed_framework_cache_ignores_bytecode_but_rejects_changed_physics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);installed=root/'installed';installed.mkdir()
            source=installed/'physics.py';source.write_text('value=1\n')
            details=dict(framework='dinosaur',revision='frozen',inputs_sha256={},
                installed_source=str(installed),installed_source_sha256=comparison.source_hashes(installed))
            with patch.object(comparison,'INPUTS',{}):
                check=lambda:comparison.validate_prepared(root,details,'dinosaur','frozen')
                check();(installed/'__pycache__').mkdir();(installed/'__pycache__/physics.pyc').touch();check()
                source.write_text('value=2\n')
                with self.assertRaisesRegex(ValueError,'framework source changed'): check()
                with self.assertRaisesRegex(ValueError,'identity differs'):
                    comparison.validate_prepared(root,details,'dinosaur','different')

    def test_gpu_jobs_reject_preparation_and_cpu_only_native_work(self):
        with patch.dict(os.environ,EXPERIMENT_GPU='1'):
            for framework,resolution,arguments in (('native','C24',[]),('dinosaur','T21',['--prepare-only'])):
                with patch.object(sys,'argv',['comparison',*arguments]):
                    with self.assertRaises(SystemExit): comparison.main(framework,resolution)


if __name__=='__main__': unittest.main()
