"""Read native startup corner-area bounds and actual dynamics substeps."""
import hashlib
import json
from pathlib import Path
import re

baseline = json.loads(Path('/analysis/baseline.json').read_text())
records = {}
for case, metadata in baseline.items():
    outputs = Path(metadata['sampling']['path']).parent
    logs = []
    for path in outputs.parent.glob('*.log'):
        lines = [line for line in path.read_text(errors='replace').splitlines()
                 if any(key in line for key in ('da_min_c', 'da_max_c', 'Acoustic (small) dt', 'Remapping dt'))]
        logs.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), lines=lines))
    assert any('da_min_c = ' in line for log in logs for line in log['lines']), (case, logs)
    assert any('Acoustic (small) dt' in line for log in logs for line in log['lines']), (case, logs)
    nml = outputs / 'input.nml'
    substeps = [line for line in nml.read_text().splitlines()
                if re.match(r'\s*(n_split|k_split|p_split|nsplit|ksplit)\s*=', line, re.I)]
    assert len(substeps) in (2, 3), (case, substeps)
    records[case] = dict(startup_logs=logs, substeps=substeps,
                         input_nml_sha256=hashlib.sha256(nml.read_bytes()).hexdigest())
output = Path('/work/pr123/native-damping-metadata-v2')
output.mkdir(parents=True, exist_ok=True)
(output / 'metadata.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records), flush=True)
