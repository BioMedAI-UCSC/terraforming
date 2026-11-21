"""Download experiment outputs and pod/job records from Nautilus without overwriting snapshots."""
import argparse
import datetime
import hashlib
import json
import inspect
from pathlib import Path
import subprocess
import time
import uuid


def hash_tree(root):
    root=Path(root)
    if not root.is_dir(): raise ValueError('output directory does not exist')
    hashes={}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError(f'output symlinks are not supported: {path}')
        if path.is_file():
            with path.open('rb') as handle:
                hashes[str(path.relative_to(root))]=hashlib.file_digest(handle,'sha256').hexdigest()
    return hashes


def main(default_job=None,default_path='/work/outputs',default_output='outputs'):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',default=default_job,required=default_job is None)
    parser.add_argument('--context',default='nautilus')
    parser.add_argument('--namespace',default='razvanlab')
    parser.add_argument('--path',default=default_path,help='Absolute directory inside the persistent work volume')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    remote=Path(args.path)
    if not remote.is_absolute() or '..' in remote.parts or not remote.is_relative_to('/work'):
        parser.error('remote path must be an absolute directory under /work')
    output=args.output or Path(default_output)/args.job/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    if output.exists() and any(output.iterdir()):
        parser.error('output directory is not empty; choose a new snapshot path')
    output.mkdir(parents=True,exist_ok=True)
    kube=['kubectl','--context',args.context,'-n',args.namespace]
    def read(*arguments):
        return subprocess.check_output([*kube,*arguments],text=True)
    job=json.loads(read('get','job',args.job,'-o','json'))
    pods=json.loads(read('get','pods','-l','job-name='+args.job,'-o','json'))['items']
    pods.sort(key=lambda pod:pod['metadata'].get('creationTimestamp',''))
    records=output/'pods';records.mkdir()
    (output/'job.json').write_text(json.dumps(job,indent=2))
    for pod in pods:
        name=pod['metadata']['name']
        (records/(name+'.json')).write_text(json.dumps(pod,indent=2))
        result=subprocess.run([*kube,'logs',name,'--all-containers'],text=True,capture_output=True)
        (records/(name+'.log')).write_text(result.stdout+result.stderr)
    active=next((p for p in pods if p['status']['phase']=='Running'),None)
    transfer=None
    reader_name=None
    try:
        if active:
            name=active['metadata']['name']
        else:
            spec=job['spec']['template']['spec']
            mounts=spec['containers'][0]['volumeMounts']
            mount=next(m for m in mounts if m['mountPath']=='/work')
            volume=next(v for v in spec['volumes'] if v['name']==mount['name'])
            claim=volume['persistentVolumeClaim']['claimName']
            transfer='experiment-download-'+uuid.uuid4().hex[:8]
            reader={'apiVersion':'batch/v1','kind':'Job','metadata':{'name':transfer,'namespace':args.namespace},
                'spec':{'backoffLimit':0,'template':{'metadata':{'labels':{'job-name':transfer}},'spec':{
                'restartPolicy':'Never','automountServiceAccountToken':False,
                'securityContext':{'runAsNonRoot':True,'runAsUser':10001,'runAsGroup':10001,'seccompProfile':{'type':'RuntimeDefault'}},
                'containers':[{'name':'results','image':spec['containers'][0]['image'],
                    'command':['sh','-c','while [ ! -f /control/complete ]; do sleep 1; done'],
                    'resources':{'requests':{'cpu':'1','memory':'1Gi'},'limits':{'cpu':'1','memory':'1Gi'}},
                    'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}},
                    'volumeMounts':[{'name':'work','mountPath':'/work','readOnly':True},
                        {'name':'control','mountPath':'/control'}]}],
                'volumes':[{'name':'work','persistentVolumeClaim':{'claimName':claim,'readOnly':True}},
                    {'name':'control','emptyDir':{}}]}}}}
            subprocess.run([*kube,'apply','-f','-'],input=json.dumps(reader),text=True,check=True)
            while True:
                candidates=json.loads(read('get','pods','-l','job-name='+transfer,'-o','json'))['items']
                if candidates:
                    reader_name=candidates[-1]['metadata']['name']
                    status=candidates[-1]['status']
                    if status.get('phase') in ('Failed','Succeeded'):
                        raise RuntimeError(f'Result reader exited before readiness: {status}')
                    failures=[c for c in status.get('containerStatuses',[]) if
                        c.get('state',{}).get('waiting',{}).get('reason') in
                        ('ImagePullBackOff','ErrImagePull','CreateContainerConfigError')]
                    if failures: raise RuntimeError(f'Result reader cannot start: {failures}')
                    if any(c['type']=='Ready' and c['status']=='True' for c in status.get('conditions',[])):
                        break
                time.sleep(2)
            name=reader_name
        script='import hashlib,json;from pathlib import Path\n'+inspect.getsource(hash_tree)+\
            '\nprint(json.dumps(hash_tree('+repr(args.path)+'),sort_keys=True))'
        def remote_hashes():
            return json.loads(read('exec',name,'--','python3','-c',script))
        before=remote_hashes()
        subprocess.run([*kube,'cp',name+':'+args.path,str(output/'runs')],check=True)
        after=remote_hashes()
        local=hash_tree(output/'runs')
        if before!=after or before!=local:
            raise RuntimeError('Output changed during transfer or download hashes differ; snapshot is unverified')
        (output/'remote-sha256.json').write_text(json.dumps(before,indent=2))
        hashes=hash_tree(output)
        (output/'sha256.json').write_text(json.dumps(hashes,indent=2))
        print(f'Saved outputs and pod logs to {output}')
    finally:
        if transfer:
            if reader_name:
                subprocess.run([*kube,'exec',reader_name,'--','touch','/control/complete'])
            cleanup=subprocess.run([*kube,'delete','job',transfer,'--wait=false'])
            if cleanup.returncode: print(f'Reader cleanup failed; delete job {transfer} after transfer')


if __name__=='__main__':
    main()
