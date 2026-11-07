"""Render or submit a generic Nautilus experiment Job and its persistent volume."""
import argparse
import json
import re
import subprocess

IMAGE='samkash/terraforming-paper@sha256:a04422421dbe06a2073d55259ba9ae9d54e2f22488e4bc2552fdbb5bc3f24fd2'
REPOSITORY='https://github.com/BioMedAI-UCSC/terraforming.git'


def manifests(args):
    for name in (args.name,args.namespace,args.pvc or args.name):
        if not re.fullmatch(r'[a-z0-9](?:[-a-z0-9]*[a-z0-9])?',name) or len(name)>63:
            raise ValueError('job, namespace and PVC names must be DNS labels of at most 63 characters')
    pvc=args.pvc or args.name
    invocation=['--repository',args.repository,'--ref',args.ref,'--run-id',args.run_id]
    if args.gpu: invocation.append('--gpu')
    invocation+=['--experiments',*args.experiments]
    invocation+=args.arguments if args.arguments[:1]==['--'] else ['--',*args.arguments]
    requests={'cpu':args.cpu,'memory':args.memory,'ephemeral-storage':'2Gi'}
    limits=dict(requests,**{'ephemeral-storage':'8Gi'})
    affinity=[{'key':'topology.kubernetes.io/region','operator':'In','values':['us-west']}]
    if args.gpu:
        requests[args.gpu_resource]=limits[args.gpu_resource]='1'
    pod={'restartPolicy':'Never','automountServiceAccountToken':False,'terminationGracePeriodSeconds':120,
        'securityContext':{'runAsNonRoot':True,'runAsUser':10001,'runAsGroup':10001,'fsGroup':10001,
            'fsGroupChangePolicy':'OnRootMismatch','seccompProfile':{'type':'RuntimeDefault'}},
        'affinity':{'nodeAffinity':{'requiredDuringSchedulingIgnoredDuringExecution':{'nodeSelectorTerms':[{'matchExpressions':affinity}]}}},
        'containers':[{'name':'experiment','image':args.image,'imagePullPolicy':'IfNotPresent','args':invocation,
            'resources':{'requests':requests,'limits':limits},
            'securityContext':{'allowPrivilegeEscalation':False,'readOnlyRootFilesystem':True,'capabilities':{'drop':['ALL']}},
            'volumeMounts':[{'name':'work','mountPath':'/work'},{'name':'tmp','mountPath':'/tmp'},{'name':'shm','mountPath':'/dev/shm'}]}],
        'volumes':[{'name':'work','persistentVolumeClaim':{'claimName':pvc}},
            {'name':'tmp','emptyDir':{'sizeLimit':'4Gi'}},{'name':'shm','emptyDir':{'medium':'Memory','sizeLimit':'8Gi'}}]}
    return {'apiVersion':'v1','kind':'List','items':[
        {'apiVersion':'v1','kind':'PersistentVolumeClaim','metadata':{'name':pvc,'namespace':args.namespace},
            'spec':{'storageClassName':'rook-ceph-block','accessModes':['ReadWriteOnce'],'resources':{'requests':{'storage':args.storage}}}},
        {'apiVersion':'batch/v1','kind':'Job','metadata':{'name':args.name,'namespace':args.namespace},
            'spec':{'backoffLimit':2,'activeDeadlineSeconds':604800,
                'template':{'metadata':{'labels':{'app':'nautalis-experiments'}},'spec':pod}}}]}


def parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--ref',required=True,help='Branch, tag or commit containing the experiment folders')
    parser.add_argument('--experiments','--experiment',nargs='+',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--repository',default=REPOSITORY)
    parser.add_argument('--image',default=IMAGE)
    parser.add_argument('--context',default='nautilus')
    parser.add_argument('--namespace',default='razvanlab')
    parser.add_argument('--pvc',help='Existing/shared claim; defaults to the job name')
    parser.add_argument('--storage',default='500Gi')
    parser.add_argument('--cpu',default='8')
    parser.add_argument('--memory',default='64Gi')
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--gpu-resource',default='nvidia.com/a100')
    parser.add_argument('--submit',action='store_true',help='Apply instead of printing JSON')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    return parser


if __name__=='__main__':
    args=parser().parse_args()
    body=json.dumps(manifests(args),indent=2)
    if args.submit:
        subprocess.run(['rtk','kubectl','--context',args.context,'-n',args.namespace,'apply','-f','-'],input=body,text=True,check=True)
    else:
        print(body)
