#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import sys
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


BASE_URL = "https://pds-ppi.igpp.ucla.edu/ditdos/download"


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def build_url(date: dt.date, version_id: str, release: str) -> str:
    ymd = date.strftime("%Y%m%d")
    params = {
        "id": (
            "urn:nasa:pds:maven.euv.modelled:data.minute.spectra:"
            f"mvn_euv_l3_minute_{ymd}::{version_id}"
        ),
        "slot": f"/data/maven-euv-modelled/data/minute/{date.year:04d}/{date.month:02d}",
        "file_name": f"mvn_euv_l3_minute_{ymd}_{release}.xml",
        "data_file": f"mvn_euv_l3_minute_{ymd}_{release}.cdf",
    }
    query = "&".join(
        f"{key}={urllib.parse.quote(value, safe='/:')}"
        for key, value in params.items()
    )
    return f"{BASE_URL}?{query}"


def sample_dates(
    start_date: dt.date, samples: int, martian_year_days: int
) -> tuple[list[dt.date], float]:
    step = martian_year_days / samples
    seen: set[dt.date] = set()
    dates: list[dt.date] = []
    for i in range(samples):
        date = start_date + dt.timedelta(days=round(i * step))
        if date not in seen:
            dates.append(date)
            seen.add(date)
    return dates, step


def download_one(
    date: dt.date,
    output_dir: Path,
    version_id: str,
    release: str,
    keep_zip: bool,
    dry_run: bool,
) -> bool:
    ymd = date.strftime("%Y%m%d")
    base_name = f"mvn_euv_l3_minute_{ymd}_{release}"
    xml_path = output_dir / f"{base_name}.xml"
    cdf_path = output_dir / f"{base_name}.cdf"

    if xml_path.exists() and cdf_path.exists():
        print(f"skip {ymd}: already have .xml and .cdf")
        return True

    url = build_url(date, version_id, release)
    if dry_run:
        print(f"dry-run {ymd}: {url}")
        return True

    zip_path = output_dir / f"{base_name}.zip"
    try:
        with urllib.request.urlopen(url) as response:
            zip_path.write_bytes(response.read())
    except Exception as exc:
        print(f"error {ymd}: download failed ({exc})")
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            archive.extractall(output_dir)
    except zipfile.BadZipFile:
        print(f"error {ymd}: invalid zip returned")
        return False
    finally:
        if zip_path.exists() and not keep_zip:
            zip_path.unlink()

    if not xml_path.exists() or not cdf_path.exists():
        print(f"warn {ymd}: expected files not found after extract")
        return False

    print(f"ok   {ymd}: downloaded")
    return True


def parse_dates_list(raw: str | None) -> list[dt.date]:
    if not raw:
        return []
    return [parse_date(item.strip()) for item in raw.split(",") if item.strip()]


def parse_dates_file(path: str | None) -> list[dt.date]:
    if not path:
        return []
    dates: list[dt.date] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            dates.append(parse_date(line))
    return dates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download MAVEN EUV modeled minute irradiance samples from PDS/PPI."
        )
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=dt.date(2014, 11, 10),
        help="Reference start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=24,
        help="Number of evenly spaced samples across a Martian year.",
    )
    parser.add_argument(
        "--martian-year-days",
        type=int,
        default=687,
        help="Martian year length in Earth days.",
    )
    parser.add_argument(
        "--version-id",
        default="24.0",
        help="PDS version id (default: 24.0).",
    )
    parser.add_argument(
        "--release",
        default="v17_r02",
        help="Product release string in filenames (default: v17_r02).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/maven/euv/_maven.euv.modelled",
        help="Directory to store downloads.",
    )
    parser.add_argument(
        "--dates",
        default=None,
        help="Comma-separated list of YYYY-MM-DD to download.",
    )
    parser.add_argument(
        "--dates-file",
        default=None,
        help="Text file with one YYYY-MM-DD per line.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print URLs without downloading.",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="Keep the downloaded zip files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of downloads (0 means no limit).",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = parse_dates_list(args.dates)
    dates += parse_dates_file(args.dates_file)

    if not dates:
        dates, step = sample_dates(
            args.start_date, args.samples, args.martian_year_days
        )
        print(
            f"Sampling {len(dates)} dates across ~{args.martian_year_days} days "
            f"(step={step:.2f} days)."
        )

    seen: set[dt.date] = set()
    deduped: list[dt.date] = []
    for date in dates:
        if date not in seen:
            deduped.append(date)
            seen.add(date)
    dates = deduped

    failures = 0
    for idx, date in enumerate(dates):
        if args.limit and idx >= args.limit:
            break
        ok = download_one(
            date,
            output_dir=output_dir,
            version_id=args.version_id,
            release=args.release,
            keep_zip=args.keep_zip,
            dry_run=args.dry_run,
        )
        if not ok:
            failures += 1

    if failures:
        print(f"Done with {failures} failures.")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

