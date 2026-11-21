import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mars_calibration.provenance import hardware,snapshot_sources


def test_unavailable_cpu_is_not_mislabeled_as_architecture(monkeypatch):
    import mars_calibration.provenance as p
    monkeypatch.setattr(p.platform,'processor',lambda:'arm')
    monkeypatch.setattr(p.platform,'system',lambda:'Other')
    result=hardware([SimpleNamespace(platform='gpu',device_kind='Named GPU')])
    assert result['cpu_model'] is None
    assert result['cpu_model_status']=='unavailable'
    assert result['devices'][0]['kind']=='Named GPU'


def test_snapshot_preserves_uncommitted_source_without_copying_other_files(tmp_path):
    root=tmp_path/'repo';(root/'scripts').mkdir(parents=True)
    source=root/'scripts/run.py';source.write_bytes(b'print("uncommitted")\r\n')
    (root/'scripts/private.env').write_text('not source')
    out=tmp_path/'run'
    records=snapshot_sources(root,out)
    assert len(records)==1
    assert records[0]['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert (out/records[0]['snapshot']).read_bytes()==source.read_bytes()
    source.write_text('changed')
    assert (out/records[0]['snapshot']).read_bytes()!=source.read_bytes()
