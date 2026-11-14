from pathlib import Path
import json
import sys
import tarfile

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from archive_historical_results import archive_run


def test_historical_archive_preserves_bytes_and_rejects_overwrite_and_nan(tmp_path):
    source, output = tmp_path / "run", tmp_path / "release"
    source.mkdir()
    (source / "environment.json").write_text(json.dumps(dict(git_commit="original", git_status="")))
    np.savez(source / "restart.npz", temperature=np.array([200.]))
    report = archive_run(source, output)
    assert report["recorded_git_commit"] == "original"
    assert report["original_gpu_model"] is None
    assert report["numeric_saved_arrays_checked"] == 1
    with tarfile.open(output / report["archive"]) as archive:
        assert archive.extractfile("run/restart.npz").read() == (source / "restart.npz").read_bytes()
    with pytest.raises(FileExistsError):
        archive_run(source, output)
    np.savez(source / "restart.npz", temperature=np.array([np.nan]))
    with pytest.raises(ValueError, match="Nonfinite"):
        archive_run(source, tmp_path / "bad-release")
