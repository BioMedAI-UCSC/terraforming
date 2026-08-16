#!/usr/bin/env python3
"""Stage the MOLA MEGDR global topography raster used by the gcm3d 3-D Mars maps.

Downloads the MGS MOLA MEGDR 4-pixel/degree topography (``megt90n000cb.img``)
from the NASA PDS Geosciences Node, verifies its SHA-256 and size against the
values pinned in ``src.gcm3d.topography``, and writes it to the configured path
(``$MOLA_PATH`` or the repo default ``data/mola/meg004/megt90n000cb.img``).

The download is integrity-checked, so a moved/incorrect mirror or a truncated
transfer fails loudly instead of silently staging corrupt data.

Usage:
    python scripts/stage_mola.py            # download + verify to the default path
    python scripts/stage_mola.py --verify   # only verify an already-staged file
    MOLA_PATH=/data/megt90n000cb.img python scripts/stage_mola.py

Provenance / license:
    Dataset  MGS-M-MOLA-5-MEGDR-L3-V1 (NASA PDS Geosciences Node), public domain.
    Cite     Smith, D. E., et al. (2001), JGR 106(E10), 23689-23722.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

# Import the pinned provenance so the script and the loader never disagree.
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "package" / "src"))
from src.gcm3d.topography import (  # noqa: E402
    MOLA_SHA256,
    MOLA_SIZE_BYTES,
    MOLA_SOURCE_URL,
    _default_mola_path,
)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path) -> bool:
    if not path.exists():
        print(f"✖  not found: {path}")
        return False
    size = path.stat().st_size
    if size != MOLA_SIZE_BYTES:
        print(f"✖  size {size} != expected {MOLA_SIZE_BYTES}: {path}")
        return False
    digest = _sha256(path)
    if digest != MOLA_SHA256:
        print(f"✖  sha256 {digest}\n   != {MOLA_SHA256}")
        return False
    print(f"✔  verified {path}\n   sha256 {digest}  ({size} bytes)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Stage the MOLA MEGDR raster.")
    ap.add_argument("--verify", action="store_true",
                    help="only verify an already-staged file (no download)")
    ap.add_argument("--url", default=MOLA_SOURCE_URL,
                    help="override the source URL (PDS mirrors move)")
    args = ap.parse_args()

    dest = _default_mola_path()

    if args.verify:
        return 0 if _verify(dest) else 1

    if dest.exists() and _verify(dest):
        print("already staged and valid — nothing to do.")
        return 0

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"downloading {args.url}\n         -> {dest}")
    try:
        urllib.request.urlretrieve(args.url, tmp)  # noqa: S310 (trusted PDS host)
    except Exception as exc:  # noqa: BLE001
        print(f"✖  download failed: {exc}\n"
              f"   The PDS mirror may have moved; pass --url with the current "
              f"location of megt90n000cb.img (integrity is still checked).")
        tmp.unlink(missing_ok=True)
        return 1

    tmp.replace(dest)
    if not _verify(dest):
        print("✖  staged file failed integrity check — removing.")
        dest.unlink(missing_ok=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
