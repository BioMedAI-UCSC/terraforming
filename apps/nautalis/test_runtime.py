"""Small real-process tests for generic source selection, outputs and download safety."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).parent


def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'))
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


jobs=module('jobs');download=module('download')
SCRIPT='''import json,os,sys
from pathlib import Path
p=Path(os.environ['EXPERIMENT_OUTPUT_DIR'])/'result.json'
p.write_text(json.dumps(dict(arguments=sys.argv[1:],gpu=os.environ['EXPERIMENT_GPU'])))
print('experiment complete',flush=True)
sys.exit(7 if 'fail' in sys.argv else 0)
'''


class RuntimeTests(unittest.TestCase):
    def source(self,root):
        source=root/'source';(source/'sample').mkdir(parents=True)
        (source/'sample/run.py').write_text(SCRIPT)
        return source

    def invoke(self,source,work,*arguments,remote=False):
        selection=['--repository',str(source),'--ref','experiment-branch'] if remote else ['--source',str(source)]
        return subprocess.run([sys.executable,str(HERE/'run.py'),*selection,'--work',str(work),
            '--experiments','sample','--run-id','test','--',*arguments],capture_output=True,text=True)

    def test_mounted_folder_retries_preserve_outputs_and_arguments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=self.source(root);work=root/'work'
            first=self.invoke(source,work,'value with spaces')
            self.assertEqual(first.returncode,0,first.stderr)
            (source/'.venv').mkdir()
            (source/'.venv/ignored').touch()
            second=self.invoke(source,work,'fail')
            self.assertEqual(second.returncode,7,second.stderr)
            runs=list((work/'outputs/sample').glob('*/test'))
            self.assertEqual(len(runs),1)
            attempts=sorted((runs[0]/'attempts').iterdir())
            self.assertEqual(len(attempts),2)
            self.assertEqual(json.loads((attempts[0]/'outputs/result.json').read_text())['arguments'],['value with spaces'])
            self.assertEqual(json.loads((attempts[1]/'status.json').read_text())['returncode'],7)
            self.assertIn('experiment complete',(attempts[0]/'run.log').read_text())
            self.assertTrue((runs[0]/'source.tar').exists())

    def test_branch_is_fetched_and_resolved_revision_is_archived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=self.source(root)
            def git(*args):
                return subprocess.check_output(['git','-C',str(source),*args],text=True,stderr=subprocess.DEVNULL).strip()
            git('init','-b','experiment-branch');git('config','user.name','Test');git('config','user.email','test@example.invalid')
            git('add','.');git('commit','-m','test experiment')
            expected=git('rev-parse','HEAD')
            result=self.invoke(source,root/'work','argument',remote=True)
            self.assertEqual(result.returncode,0,result.stderr)
            saved=json.loads(next((root/'work/outputs/sample').glob('*/test/source.json')).read_text())
            self.assertEqual(saved['revision'],expected)
            self.assertEqual(saved['requested_ref'],'experiment-branch')

    def test_selected_folder_cannot_escape_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=self.source(root)
            (root/'outside').mkdir();(root/'outside/run.py').write_text(SCRIPT)
            result=subprocess.run([sys.executable,str(HERE/'run.py'),'--source',str(source),
                '--work',str(root/'work'),'--experiments','../outside','--run-id','test'],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertFalse((root/'work/outputs').exists())

    def test_multiple_folders_keep_independent_attempts_even_on_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=self.source(root)
            (source/'second').mkdir();(source/'second/run.py').write_text(SCRIPT)
            result=subprocess.run([sys.executable,str(HERE/'run.py'),'--source',str(source),
                '--work',str(root/'work'),'--experiments','sample','second','--run-id','test','--','fail'],capture_output=True)
            self.assertEqual(result.returncode,7)
            self.assertEqual(len(list((root/'work/outputs').glob('*/*/test/attempts/*/status.json'))),2)

    def test_jobs_are_protocol_independent_and_preserve_gpu_security_contract(self):
        options=['--name','custom','--ref','another-branch','--experiments','custom_folder','--run-id','v1']
        cpu=jobs.manifests(jobs.parser().parse_args(options))
        gpu=jobs.manifests(jobs.parser().parse_args(options+['--gpu','--','task','--seed','7']))
        for body in (cpu,gpu):
            self.assertNotIn('activeDeadlineSeconds',body['items'][1]['spec'])
            pod=body['items'][1]['spec']['template']['spec']
            self.assertNotIn('activeDeadlineSeconds',pod)
            self.assertFalse(pod['automountServiceAccountToken'])
            self.assertNotIn('initContainers',pod)
            self.assertIn('custom_folder',pod['containers'][0]['args'])
            self.assertEqual(body['items'][0]['spec']['accessModes'],['ReadWriteOnce'])
        container=gpu['items'][1]['spec']['template']['spec']['containers'][0]
        self.assertEqual(container['args'][-4:],['--','task','--seed','7'])
        self.assertEqual(container['resources']['requests']['nvidia.com/a100'],'1')
        self.assertNotIn('nvidia.com/a100',cpu['items'][1]['spec']['template']['spec']['containers'][0]['resources']['requests'])

    def test_completed_job_download_uses_readonly_reader_and_cleans_it_up(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary)/'snapshot';calls=[]
            job={'spec':{'template':{'spec':{'containers':[{'volumeMounts':[{'name':'work','mountPath':'/work'}]}],
                'volumes':[{'name':'work','persistentVolumeClaim':{'claimName':'saved-results'}}]}}}}
            pod={'metadata':{'name':'finished'},'spec':{'nodeName':'previous-node'},'status':{'phase':'Succeeded'}}
            def read(args,**kwargs):
                if 'pod' in args:
                    return json.dumps({'status':{'phase':'Running','conditions':[{'type':'Ready','status':'True'}]}})
                return json.dumps(job if 'job' in args else {'items':[pod]})
            def execute(args,**kwargs):
                calls.append((args,kwargs))
                if 'cp' in args:
                    target=Path(args[-1]);target.mkdir();(target/'result.json').write_text('{}')
                return SimpleNamespace(stdout='pod logs',stderr='',returncode=0)
            with patch.object(sys,'argv',['download','--job','custom','--output',str(output)]), \
                 patch.object(download.subprocess,'check_output',side_effect=read), \
                 patch.object(download.subprocess,'run',side_effect=execute):
                download.main()
            reader=json.loads(next(kwargs['input'] for args,kwargs in calls if 'apply' in args))
            self.assertTrue(reader['spec']['containers'][0]['volumeMounts'][0]['readOnly'])
            self.assertNotIn('activeDeadlineSeconds',reader['spec'])
            self.assertEqual(reader['spec']['containers'][0]['command'],['tail','-f','/dev/null'])
            self.assertNotIn('fsGroup',reader['spec']['securityContext'])
            self.assertTrue(any('delete' in args for args,_ in calls))
            self.assertTrue((output/'pods/finished.log').exists())
            self.assertIn('runs/result.json',json.loads((output/'sha256.json').read_text()))

    def test_downloader_refuses_to_overwrite_existing_snapshot(self):
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);(output/'saved').write_text('keep')
            with patch.object(sys,'argv',['download','--job','custom','--output',str(output)]), \
                 patch.object(download.subprocess,'check_output') as request:
                with self.assertRaises(SystemExit): download.main()
                request.assert_not_called()
            self.assertEqual((output/'saved').read_text(),'keep')

    def test_checked_in_nautilus_manifests_have_no_deadline(self):
        root=HERE.parents[1]
        for name in ('apps/mars-calibration/job.yaml','apps/mars-calibration/data-pod.yaml',
                     'experiments/reviewer-validation/job.json',
                     'experiments/reviewer-multiday-calibration/job.json'):
            self.assertNotIn('activeDeadlineSeconds',(root/name).read_text())


if __name__=='__main__': unittest.main()
