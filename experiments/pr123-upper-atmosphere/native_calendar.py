"""Check the native executable's prepared source and Mars day build flags."""
import hashlib
import json
from pathlib import Path

root = Path('/native')
ready = next(root.glob('reference-runtime/*/native/ready.json'))
runtime = json.loads(ready.read_text())
cache = ready.parent
binary = root / Path(runtime['executable']).relative_to('/work')
assert hashlib.sha256(binary.read_bytes()).hexdigest() == runtime['executable_sha256']
for name, expected in runtime['inputs_sha256'].items():
    with (cache / 'src/AmesGCM/data' / name).open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected, name
records = {}
for path in (cache / 'exec').rglob('*'):
    if path.is_file() and (path.name == 'Makefile' or path.name.startswith('mkmf.template')):
        lines = [line for line in path.read_text(errors='replace').splitlines()
                 if line.startswith(('CPPDEFS', 'MARS_CPP', 'CPPFLAGS', 'FPPFLAGS', 'FFLAGS'))]
        if lines:
            records[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(), lines=lines)
            assert not any('-DSECDAY86400' in line for line in lines), path
assert records, 'Prepared native build flags are missing'
constants = {}
for path in (cache / 'src/FMS').rglob('*'):
    if not path.is_file() or '.git' in path.parts or path.stat().st_size > 2_000_000:
        continue
    lines = path.read_text(errors='replace').splitlines()
    matches = [index for index, line in enumerate(lines) if any(word in line for word in ('SECDAY86400', '8.8704', '88704'))]
    if matches:
        constants[str(path)] = dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            lines={str(index + 1): lines[max(0, index - 2):index + 4] for index in matches})
assert constants, 'Prepared solar-day constant source is missing'
record = dict(executable_sha256=runtime['executable_sha256'], native_revision=runtime['revision'],
              build_flags=records, constants_sources=constants, inputs_sha256=runtime['inputs_sha256'],
              prepared_manifest_sha256=hashlib.sha256(ready.read_bytes()).hexdigest(),
              prepared_source_hashes_sha256=hashlib.sha256(json.dumps(runtime['source_sha256'], sort_keys=True).encode()).hexdigest())
output = Path('/work/pr123/native-calendar')
output.mkdir(exist_ok=True)
(output / 'calendar.json').write_text(json.dumps(record, indent=2))
print(json.dumps(record), flush=True)
