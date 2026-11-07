"""Create a blank experiment folder that follows the Nautalis runtime contract."""
import argparse
from pathlib import Path
import re


RUN_TEMPLATE = '''#!/usr/bin/env python3
"""Experiment entry point. Replace this stub before submitting a job."""
import os
from pathlib import Path


def main():
    output = Path(os.environ["EXPERIMENT_OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    raise SystemExit(f"Implement this experiment entry point: {output}")


if __name__ == "__main__":
    main()
'''

README_TEMPLATE = '''# {name}

Implement this experiment in `run.py`. The Nautalis runtime starts it from the
repository root and forwards arguments after `--` unchanged.

The entry point receives `EXPERIMENT_SOURCE_DIR`, `EXPERIMENT_WORK_DIR`,
`EXPERIMENT_RUN_DIR`, `EXPERIMENT_OUTPUT_DIR`, `EXPERIMENT_RUN_ID`, and
`EXPERIMENT_GPU`. Write result artifacts to `EXPERIMENT_OUTPUT_DIR`; use the
work/run directories only for dependencies, data, or validated checkpoints.
Install dependencies from this folder's own locked setup. Return a nonzero exit
code on failure. The generated `run.py` intentionally fails until implemented.

Submit after committing this folder:

```sh
python3 apps/nautalis/jobs.py --name {name} --ref YOUR_BRANCH \\
  --experiments {path} --run-id RUN_ID --submit -- ARGUMENTS
```
'''


def create_experiment(name: str, parent: Path = Path('experiments')) -> Path:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name):
        raise ValueError('name must start with a letter or digit and contain only letters, digits, _ or -')
    target = parent / name
    target.mkdir(parents=True, exist_ok=False)
    (target / 'run.py').write_text(RUN_TEMPLATE)
    (target / 'README.md').write_text(README_TEMPLATE.format(name=name, path=target.as_posix()))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('name', help='new experiment folder name')
    parser.add_argument('--parent', type=Path, default=Path('experiments'),
                        help='parent directory (default: experiments)')
    args = parser.parse_args()
    try:
        target = create_experiment(args.name, args.parent)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(f'Created {target}')


if __name__ == '__main__':
    main()
