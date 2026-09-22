import json
from pathlib import Path
import sys
import time

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from neural_logging import configure, event, phase


def test_events_persist_heartbeats_and_failures(tmp_path):
    configure(tmp_path)
    with phase("download", heartbeat_seconds=.01, window=[0, 120]):
        time.sleep(.05)
    with pytest.raises(ValueError, match="bad data"):
        with phase("validation", heartbeat_seconds=.01):
            raise ValueError("bad data")
    event("training_progress", loss=1.25, step=1)
    records = [json.loads(line) for line in (tmp_path / "run.log").read_text().splitlines()]
    names = [r["event"] for r in records]
    assert names[0] == "download_start"
    assert "download_running" in names
    assert "download_complete" in names
    assert "validation_failed" in names
    assert "validation_complete" not in names
    assert records[-1]["loss"] == 1.25
    assert all("timestamp" in r for r in records)
    assert not any(r["event"] == "download_running" for r in records[names.index("download_complete") + 1:])


def test_reconfigure_does_not_duplicate_file_events(tmp_path):
    configure(tmp_path)
    configure(tmp_path)
    event("one_event")
    assert len((tmp_path / "run.log").read_text().splitlines()) == 1
