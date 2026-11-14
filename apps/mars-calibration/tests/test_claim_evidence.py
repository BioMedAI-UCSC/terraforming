from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "paper/arxiv-2610.04081"))
from regenerate import verify_claims


def test_claims_reject_missing_evidence_and_duplicate_seasonal_map(tmp_path):
    root = tmp_path / "paper" / "arxiv"
    assets = root / "revision/assets"
    assets.mkdir(parents=True)
    for season in (45, 135, 225, 315):
        (assets / f"ls{season:03d}_grid.png").write_bytes(str(season).encode())
    claim = dict(result="Maps", issues=[93], disposition="Historical", evidence=["revision/assets"])
    (root / "claims.json").write_text(json.dumps(dict(claims=[claim])))
    verify_claims(root)
    (assets / "ls225_grid.png").write_bytes((assets / "ls135_grid.png").read_bytes())
    with pytest.raises(AssertionError, match="Duplicate seasonal map"):
        verify_claims(root)
    claim["evidence"] = ["missing.json"]
    (root / "claims.json").write_text(json.dumps(dict(claims=[claim])))
    with pytest.raises(AssertionError, match="missing.json"):
        verify_claims(root)
