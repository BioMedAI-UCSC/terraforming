"""Portable hardware identity and byte-exact source snapshots for experiment runs."""
from __future__ import annotations

import hashlib
import platform
from pathlib import Path
import shutil
import subprocess


def hardware(devices):
    model = platform.processor()
    if model.lower() in ('', 'arm', 'arm64', 'x86_64', 'amd64', 'i386'):
        model = None
    try:
        if platform.system() == 'Darwin':
            model = subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True,
                                            stderr=subprocess.DEVNULL,timeout=5).strip() or model
        elif platform.system() == 'Linux':
            for line in Path('/proc/cpuinfo').read_text().splitlines():
                if line.lower().startswith('model name'):
                    model=line.split(':',1)[1].strip()
                    break
    except (OSError,subprocess.SubprocessError):
        pass
    return dict(cpu_model=model,cpu_model_status='identified' if model else 'unavailable',
                architecture=platform.machine(),
                devices=[dict(id=str(d),platform=d.platform,kind=d.device_kind) for d in devices])


def snapshot_sources(root,output):
    """Keep source bytes, including uncommitted/new Python files, with each run.

    A commit ID and dirty=True do not identify the executed source. Restrict the
    snapshot to science-code directories; no environment variables or credentials
    are copied. Input data retains its separate hash/availability contract.
    """
    root,output=Path(root),Path(output)
    records=[]
    for folder in ('scripts','package/src','apps/mars-calibration'):
        for source in sorted((root/folder).rglob('*.py')):
            relative=source.relative_to(root)
            target=output/'source-snapshot'/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,target)
            records.append(dict(path=str(relative),snapshot=str(target.relative_to(output)),
                                sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    return records
