"""Fetch a source revision and run selected experiment folders; no science protocols."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import tempfile

REPOSITORY = 'https://github.com/BioMedAI-UCSC/terraforming.git'


def command(args, **kwargs):
    return subprocess.run(list(map(str,args)),check=True,**kwargs)


def git(source,*args):
    return command(['git','-C',source,*args],capture_output=True,text=True).stdout.strip()


def checkout(work,repository,ref):
    if not ref or ref.startswith('-'):
        raise ValueError('a branch, tag or commit is required')
    with tempfile.TemporaryDirectory(dir=work) as temporary:
        source=Path(temporary)/'checkout'
        command(['git','init',source],stdout=subprocess.DEVNULL)
        command(['git','-C',source,'fetch','--depth=1',repository,ref])
        command(['git','-C',source,'checkout','--detach','FETCH_HEAD'])
        revision=git(source,'rev-parse','HEAD')
        target=work/'source'/revision
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            source.rename(target)
    if git(target,'rev-parse','HEAD')!=revision or git(target,'status','--porcelain'):
        raise RuntimeError('cached checkout was modified; use a fresh work directory')
    return target,revision


def archive(source,path):
    """Capture mounted folders too, without following links outside the supplied source."""
    ignored={'.git','.venv','outputs','__pycache__','.pytest_cache'}
    def include(info):
        if any(part in ignored for part in Path(info.name).parts): return None
        info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
        return info
    with tarfile.open(path,'w') as handle:
        handle.add(source,arcname='source',filter=include)


def run_folder(source,folder,work,revision,run_id,arguments,gpu,repository,ref):
    directory=(source/folder).resolve()
    if not directory.is_relative_to(source.resolve()) or not directory.is_dir():
        raise ValueError('experiment folder must exist inside the supplied source')
    entry=next((directory/name for name in ('run.py','run.sh') if (directory/name).is_file()),None)
    if entry is None:
        raise ValueError(f'{folder}: expected run.py or run.sh')
    name=directory.relative_to(source.resolve()).as_posix()
    if name=='.': name=source.name
    output=work/'outputs'/name/revision/run_id
    attempt=output/'attempts'/datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    attempt.mkdir(parents=True)
    (attempt/'outputs').mkdir()
    (output/'source.json').write_text(json.dumps(dict(repository=repository,requested_ref=ref,
        revision=revision,experiment=folder,run_id=run_id),indent=2))
    if not (output/'source.tar').exists():
        archive(source,output/'source.tar')
    state=work/'experiments'/name/revision/run_id
    state.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,EXPERIMENT_SOURCE_DIR=str(source),EXPERIMENT_WORK_DIR=str(state),
        EXPERIMENT_RUN_DIR=str(output),EXPERIMENT_OUTPUT_DIR=str(attempt/'outputs'),
        EXPERIMENT_RUN_ID=run_id,EXPERIMENT_GPU=str(int(gpu)))
    env['PYTHONPATH']=os.pathsep.join([str(source),str(source/'package'),env.get('PYTHONPATH','')])
    invocation=[sys.executable if entry.suffix=='.py' else '/bin/sh',str(entry),*arguments]
    print(f'Experiment {folder}; source {revision}; outputs {output}',flush=True)
    with (attempt/'run.log').open('w') as log:
        process=subprocess.Popen(invocation,cwd=source,env=env,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,errors='replace',start_new_session=True)
        def forward(number,frame):
            try: os.killpg(process.pid,number)
            except ProcessLookupError: pass
        previous={sig:signal.signal(sig,forward)
                  for sig in (signal.SIGTERM,signal.SIGINT)}
        try:
            for line in process.stdout:
                log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        finally:
            for sig,handler in previous.items(): signal.signal(sig,handler)
    status=dict(returncode=code,command=invocation,attempt=str(attempt),revision=revision)
    (attempt/'status.json').write_text(json.dumps(status,indent=2))
    return 128-code if code<0 else code


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--ref',help='GitHub branch, tag or commit; resolved commit is archived')
    source.add_argument('--source',type=Path,help='Mounted source directory instead of a Git fetch')
    parser.add_argument('--repository',default=REPOSITORY)
    parser.add_argument('--experiments','--experiment',nargs='+',required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--work',type=Path,default=Path('/work'))
    parser.add_argument('--gpu',action='store_true')
    parser.add_argument('arguments',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if not args.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.run_id):
        parser.error('run-id must contain only letters, digits, dash or underscore')
    work=args.work.resolve();work.mkdir(parents=True,exist_ok=True)
    with (work/'.runtime.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.source:
            source=args.source.resolve()
            if not source.is_dir(): parser.error('mounted source must be a directory')
            if work.is_relative_to(source): parser.error('work directory must be outside mounted source')
            with tempfile.TemporaryDirectory(dir=work) as temporary:
                snapshot=Path(temporary)/'source.tar'
                archive(source,snapshot)
                revision=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        else:
            source,revision=checkout(work,args.repository,args.ref)
        arguments=args.arguments[1:] if args.arguments[:1]==['--'] else args.arguments
        failures=[]
        for folder in args.experiments:
            code=run_folder(source,folder,work,revision,args.run_id,arguments,args.gpu,
                            None if args.source else args.repository,args.ref)
            if code: failures.append(code)
        return failures[0] if failures else 0


if __name__=='__main__':
    sys.exit(main())
