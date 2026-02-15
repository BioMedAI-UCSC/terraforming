#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


API_URL = (
    "https://lasp.colorado.edu/maven/sdc/public/files/api/v1/"
    "search/science/fn_metadata/download_zip"
)
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")
VALID_LEVELS = {"insitu", "iuvs"}


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def month_chunks(start: dt.date, end: dt.date, months: int) -> list[tuple[dt.date, dt.date]]:
    chunks: list[tuple[dt.date, dt.date]] = []
    cursor = dt.date(start.year, start.month, 1)
    while cursor <= end:
        year = cursor.year
        month = cursor.month + months
        while month > 12:
            year += 1
            month -= 12
        next_start = dt.date(year, month, 1)
        chunk_start = max(start, cursor)
        chunk_end = min(end, next_start - dt.timedelta(days=1))
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
        cursor = next_start
    return chunks


def year_chunks(start: dt.date, end: dt.date, years: int) -> list[tuple[dt.date, dt.date]]:
    chunks: list[tuple[dt.date, dt.date]] = []
    cursor = dt.date(start.year, 1, 1)
    while cursor <= end:
        next_start = dt.date(cursor.year + years, 1, 1)
        chunk_start = max(start, cursor)
        chunk_end = min(end, next_start - dt.timedelta(days=1))
        if chunk_start <= chunk_end:
            chunks.append((chunk_start, chunk_end))
        cursor = next_start
    return chunks


def build_url(level: str, start: dt.date, end: dt.date) -> str:
    params = {
        "instrument": "kp",
        "level": level,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    return f"{API_URL}?{urllib.parse.urlencode(params)}"


def fetch_zip(url: str, out_zip: Path, timeout: int, retries: int) -> bool:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                head = response.read(4)
                if head not in ZIP_MAGIC:
                    out_zip.with_suffix(".error").write_bytes(head + response.read())
                    return False
                with out_zip.open("wb") as handle:
                    handle.write(head)
                    shutil.copyfileobj(response, handle)
                return True
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"error: {url} ({exc})")
                return False
            time.sleep(2 * attempt)
    return False


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        zf.extractall(out_dir)
    return len(members)


def marker_name(level: str, start: dt.date, end: dt.date) -> str:
    return f"{level}_{start.isoformat()}_{end.isoformat()}.done"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download MAVEN KP data from LASP API in chunks "
            "(supports KP levels: insitu, iuvs)."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=dt.date(2014, 10, 1),
        help="Start date (YYYY-MM-DD). Default: 2014-10-01",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=dt.date.today(),
        help="End date (YYYY-MM-DD). Default: today",
    )
    parser.add_argument(
        "--levels",
        default="insitu,iuvs",
        help="Comma-separated KP levels to download (insitu,iuvs).",
    )
    parser.add_argument(
        "--insitu-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for insitu KP downloads.",
    )
    parser.add_argument(
        "--iuvs-chunk-years",
        type=int,
        default=1,
        help="Chunk size in years for iuvs KP downloads.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/maven/kp_api_full",
        help="Output directory for downloaded and extracted data.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep temporary ZIP files after extraction.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=4,
        help="Retry count for transient download failures.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Sleep between chunk downloads to avoid hammering API.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print chunk URLs without downloading.",
    )
    args = parser.parse_args()

    if args.end_date < args.start_date:
        raise SystemExit("end-date must be on or after start-date")

    levels = [x.strip().lower() for x in args.levels.split(",") if x.strip()]
    invalid = [x for x in levels if x not in VALID_LEVELS]
    if invalid:
        raise SystemExit(f"Invalid levels: {', '.join(invalid)}")

    out_root = Path(args.output_dir)
    zips_dir = out_root / "_zips"
    marker_dir = out_root / "_markers"
    out_root.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    failed = 0

    for level in levels:
        level_dir = out_root / level
        level_dir.mkdir(parents=True, exist_ok=True)
        if level == "insitu":
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.insitu_chunk_months)
            )
        else:
            chunks = year_chunks(
                args.start_date, args.end_date, max(1, args.iuvs_chunk_years)
            )

        print(f"[{level}] chunks: {len(chunks)}")
        for start, end in chunks:
            total_chunks += 1
            marker = marker_dir / marker_name(level, start, end)
            if marker.exists():
                print(f"skip {level} {start}..{end} (done)")
                continue

            url = build_url(level, start, end)
            zip_name = f"kp_{level}_{start.isoformat()}_{end.isoformat()}.zip"
            zip_path = zips_dir / zip_name

            if args.dry_run:
                print(f"dry-run {level} {start}..{end}: {url}")
                continue

            print(f"get  {level} {start}..{end}")
            ok = fetch_zip(url=url, out_zip=zip_path, timeout=args.timeout, retries=args.retries)
            if not ok:
                print(f"fail {level} {start}..{end}")
                failed += 1
                continue

            try:
                count = extract_zip(zip_path, level_dir)
                marker.write_text(f"ok files={count}\n", encoding="utf-8")
                print(f"ok   {level} {start}..{end}: {count} files")
            except zipfile.BadZipFile:
                failed += 1
                print(f"fail {level} {start}..{end}: bad zip")
                continue
            finally:
                if zip_path.exists() and not args.keep_zip:
                    zip_path.unlink()

            time.sleep(max(0.0, args.sleep_seconds))

    print(f"Done. chunks={total_chunks} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


