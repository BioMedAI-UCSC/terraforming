"""Render the CPU preparation or parallel 100-sol comparison jobs."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from apps.nautalis.jobs import manifests,parser as runtime_parser


def comparison_manifests(args):
    if not re.fullmatch('[0-9a-f]{40}',args.ref):
        raise ValueError('comparison source must be an immutable 40-character commit')
    items=[]
    for framework,resolutions in (('native',('c24','c48')),('dinosaur',('t21','t42'))):
        prepare=args.phase=='prepare'
        name=f'{args.name}-{framework}-'+('prepare' if prepare else '100sol')
        options=['--name',name,'--namespace',args.namespace,'--ref',args.ref,
            '--pvc',f'{getattr(args,"pvc_prefix",None) or args.name}-{framework}',
            '--storage','100Gi','--run-id','matched-100sol',
            '--cpu','4' if prepare else '18' if framework=='native' else '8',
            '--memory','16Gi' if prepare else '64Gi' if framework=='native' else '96Gi',
            '--parallel','2']
        gpu=not prepare and framework=='dinosaur'
        if gpu: options.append('--gpu')
        options+=['--experiments',*[f'experiments/reference-comparison/{framework}-{resolution}' for resolution in resolutions],
                  '--',*(['--prepare-only'] if prepare else ['--sols','100'])]
        volume,job=manifests(runtime_parser().parse_args(options))['items']
        job['metadata']['annotations']={'terraforming/source-revision':args.ref}
        pod=job['spec']['template']['spec']
        if prepare:
            pod['nodeSelector']={'kubernetes.io/hostname':args.input_node}
            pod['volumes'].append({'name':'reference-inputs','persistentVolumeClaim':{'claimName':args.input_pvc}})
            pod['containers'][0]['volumeMounts'].append({'name':'reference-inputs','mountPath':'/inputs',
                'subPath':'inputs','readOnly':True})
            items.append(volume)
        if gpu:
            pod['priorityClassName']='opportunistic'
            expression=pod['affinity']['nodeAffinity']['requiredDuringSchedulingIgnoredDuringExecution']['nodeSelectorTerms'][0]['matchExpressions']
            expression.extend([{'key':'nvidia.com/gpu.memory','operator':'Gt','values':['79000']},
                               {'key':'nvidia.com/cuda.driver.major','operator':'Gt','values':['524']}])
        items.append(job)
    return {'apiVersion':'v1','kind':'List','items':items}


def require_prepared(args,body):
    for job in body['items']:
        prepare_name=job['metadata']['name'].replace('-100sol','-prepare')
        saved=json.loads(subprocess.check_output(['kubectl','--context',args.context,'-n',args.namespace,
            'get','job',prepare_name,'-o','json'],text=True))
        if saved.get('status',{}).get('succeeded',0)<1:
            raise RuntimeError(f'{prepare_name}: CPU preparation has not succeeded')
        if (saved['metadata'].get('annotations',{}).get('terraforming/source-revision')!=args.ref or
            saved['spec']['template']['spec']['containers'][0]['image']!=job['spec']['template']['spec']['containers'][0]['image']):
            raise RuntimeError(f'{prepare_name}: prepared source or image differs')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True)
    parser.add_argument('--ref',required=True)
    parser.add_argument('--phase',choices=('prepare','run'),required=True)
    parser.add_argument('--input-pvc',default='mars-reference-physics-data')
    parser.add_argument('--pvc-prefix',help='Reuse the persistent framework volumes from an earlier preparation')
    parser.add_argument('--input-node',default='rci-tide-gpu-04.sdsu.edu')
    parser.add_argument('--context',default='nautilus')
    parser.add_argument('--namespace',default='razvanlab')
    parser.add_argument('--submit',action='store_true')
    args=parser.parse_args();body=comparison_manifests(args)
    if args.submit:
        if args.phase=='run': require_prepared(args,body)
        subprocess.run(['kubectl','--context',args.context,'-n',args.namespace,'apply','-f','-'],
            input=json.dumps(body),text=True,check=True)
    else: print(json.dumps(body,indent=2))


if __name__=='__main__': main()
