#!/usr/bin/env python3
"""Download MOLA MEGDR topography grids from the PDS Geosciences Node.

The MOLA MEGDR product is the gridded Mars topography.  It is the ``orog`` field
for the JCM terrain (see docs/ideas/mars-neural-gcm-plan.md, Step 9).

This script downloads the topography grid (the ``megt`` files) at one resolution.
Each product is a pair: a ``.img`` binary file and a ``.lbl`` PDS label.  The
label describes the binary format.  So the script downloads both.

Resolutions (pixels per degree):
    4    small global grid (1440 x 720).  Enough for a GCM.  The default.
    16   larger global grid (5760 x 2880).
    32, 64, 128   high resolution.  128 is tiled into many files.

The topography height is in metres, relative to the Mars areoid.

Data set: MGS-M-MOLA-5-MEGDR-L3-V1.0 (DOI 10.17189/1519460).
Source:   https://pds-geosciences.wustl.edu/missions/mgs/megdr.html

Usage:
    python scripts/download_mola_topography.py                 # 4 px/deg
    python scripts/download_mola_topography.py --resolution 16
    python scripts/download_mola_topography.py --type megr     # radius, not height
"""

from __future__ import annotations

import argparse
import ssl
import urllib.request
from html.parser import HTMLParser
from pathlib import Path

# The PDS server has certificate problems, so relax verification (same as the
# other PDS download scripts in this folder).
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

BASE_URL = "https://pds-geosciences.wustl.edu/mgs/mgs-m-mola-5-megdr-l3-v1/mgsl_300x"
OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "mola"

# Map a pixels-per-degree choice to its PDS subdirectory.
_RES_DIRS = {4: "meg004", 16: "meg016", 32: "meg032", 64: "meg064", 128: "meg128"}

# Product prefixes.  megt = topography (the height field you want for orog).
_TYPES = ("megt", "megr", "mega", "megc")


class _ListingParser(HTMLParser):
    """Collect the file links from a PDS directory-listing page."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self._prefix = prefix
        self.files: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name != "href" or not value:
                continue
            # Keep only the wanted product type and the two file extensions.
            leaf = value.rsplit("/", 1)[-1]
            if leaf.startswith(self._prefix) and leaf.endswith((".img", ".lbl")):
                if leaf not in self.files:
                    self.files.append(leaf)


def _list_files(dir_url: str, prefix: str) -> list[str]:
    """Return the product file names in a PDS directory."""
    with urllib.request.urlopen(dir_url, context=_SSL_CTX) as response:
        html = response.read().decode("utf-8", errors="replace")
    parser = _ListingParser(prefix)
    parser.feed(html)
    return sorted(parser.files)


def _download(file_url: str, dest: Path, overwrite: bool) -> None:
    """Download one file.  Skip it if it already exists."""
    if dest.exists() and not overwrite:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  download: {dest.name}")
    with urllib.request.urlopen(file_url, context=_SSL_CTX) as response:
        data = response.read()
    dest.write_bytes(data)
    print(f"    saved {len(data):,} bytes")


def main() -> None:
    ap = argparse.ArgumentParser(description="Download MOLA MEGDR topography grids.")
    ap.add_argument("--resolution", type=int, default=4, choices=sorted(_RES_DIRS),
                    help="Pixels per degree (default 4, enough for a GCM).")
    ap.add_argument("--type", default="megt", choices=_TYPES,
                    help="Product type (default megt = topography).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to save (default data/mola/<meg###>).")
    ap.add_argument("--overwrite", action="store_true",
                    help="Download again even if the file exists.")
    args = ap.parse_args()

    sub = _RES_DIRS[args.resolution]
    dir_url = f"{BASE_URL}/{sub}/"
    out_dir = args.output_dir or (OUTPUT_ROOT / sub)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Listing {dir_url}")
    files = _list_files(dir_url, args.type)
    if not files:
        raise SystemExit(f"No '{args.type}' files found at {dir_url}")

    print(f"Found {len(files)} file(s) for type '{args.type}':")
    for name in files:
        _download(dir_url + name, out_dir / name, args.overwrite)

    print(f"Done. Files are in {out_dir}")


if __name__ == "__main__":
    main()
