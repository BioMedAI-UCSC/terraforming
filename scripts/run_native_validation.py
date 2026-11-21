"""Run pinned native column validators in independent Linux CI shards."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from run_reference_comparison import PINS,checkout_repository


def cases(root):
    result=[(path.stem,[]) for path in sorted((root/'scripts').glob('validate_*.py'))
        if '--ames-source' in path.read_text()]
    result.extend([('validate_dust_lifting',['--dust-modes','2']),
        *[('validate_particle_sedimentation',args) for args in
            (['--dust-modes','2'],['--dust-modes','3'],['--dust-only'],
             ['--dust-only','--dust-modes','2'],['--dust-only','--dust-modes','3'])]])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    args=parser.parse_args()
    if not 0<=args.shard<args.shards: parser.error('shard must be in [0, shards)')
    root=Path(__file__).resolve().parents[1]
    args.sources.mkdir(parents=True,exist_ok=True);args.output.mkdir(parents=True,exist_ok=True)
    for repository in ('AmesGCM','FMS','atmos_phys'):
        owner,pin=PINS[repository]
        checkout_repository(args.sources/repository,f'https://github.com/{owner}/{repository}.git',pin)
    sources={'--ames-source':args.sources/'AmesGCM','--fms-source':args.sources/'FMS',
        '--fms-bilinear-source':args.sources/'FMS/horiz_interp/include/horiz_interp_bilinear.inc',
        '--gfdl-source':args.sources/'atmos_phys'}
    records=[]
    for index,(name,extra) in enumerate(cases(root)):
        if index%args.shards!=args.shard: continue
        script=root/'scripts'/(name+'.py');text=script.read_text()
        destination=args.output/f'{index:02d}-{name}.json'
        command=[sys.executable,str(script),'--output',str(destination),*extra]
        for flag,path in sources.items():
            if flag in text: command.extend([flag,str(path.resolve())])
        with destination.with_suffix('.log').open('w') as log:
            result=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(case=index,script=name,arguments=extra,returncode=result.returncode))
        print(json.dumps(records[-1]),flush=True)
    (args.output/f'summary-{args.shard}.json').write_text(json.dumps(records,indent=2)+'\n')
    if not records or any(row['returncode'] for row in records): raise SystemExit(1)


if __name__=='__main__': main()
