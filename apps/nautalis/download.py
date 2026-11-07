"""Download experiment outputs and pod/job records from Nautilus without overwriting snapshots."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import uuid


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
    kube=['rtk','kubectl','--context',args.context,'-n',args.namespace]
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
            reader={'apiVersion':'v1','kind':'Pod','metadata':{'name':transfer,'namespace':args.namespace},'spec':{
                'restartPolicy':'Never','activeDeadlineSeconds':3600,'automountServiceAccountToken':False,
                'securityContext':{'runAsNonRoot':True,'runAsUser':10001,'runAsGroup':10001,'seccompProfile':{'type':'RuntimeDefault'}},
                'containers':[{'name':'results','image':'busybox:1.37.0','command':['sleep','3600'],
                    'resources':{'requests':{'cpu':'1','memory':'1Gi'},'limits':{'cpu':'1','memory':'1Gi'}},
                    'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}},
                    'volumeMounts':[{'name':'work','mountPath':'/work','readOnly':True}]}],
                'volumes':[{'name':'work','persistentVolumeClaim':{'claimName':claim,'readOnly':True}}]}}
            if pods and pods[-1]['spec'].get('nodeName'):
                reader['spec']['nodeSelector']={'kubernetes.io/hostname':pods[-1]['spec']['nodeName']}
            subprocess.run([*kube,'apply','-f','-'],input=json.dumps(reader),text=True,check=True)
            subprocess.run([*kube,'wait','--for=condition=Ready','pod/'+transfer,'--timeout=120s'],check=True)
            name=transfer
        subprocess.run([*kube,'cp',name+':'+args.path,str(output/'runs')],check=True)
        hashes={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in output.rglob('*') if p.is_file() and not p.is_symlink()}
        (output/'sha256.json').write_text(json.dumps(hashes,indent=2))
        print(f'Saved outputs and pod logs to {output}')
    finally:
        if transfer:
            cleanup=subprocess.run([*kube,'delete','pod',transfer,'--wait=false'])
            if cleanup.returncode: print(f'Reader cleanup failed; delete pod {transfer} after transfer')


if __name__=='__main__':
    main()
