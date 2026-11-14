#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
import shutil


BASE_URL = "https://pds-ppi.igpp.ucla.edu/ditdos/download"
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")


class ProductConfig:
    def __init__(
        self,
        key: str,
        description: str,
        urn_prefix: str,
        slot_prefix: str,
        file_prefix: str,
        data_extension: str,
        include_filenames: bool,
        include_slot: bool,
        default_version_id: str,
        default_release: str,
        default_output_dir: str,
        default_start_date: dt.date,
    ) -> None:
        self.key = key
        self.description = description
        self.urn_prefix = urn_prefix
        self.slot_prefix = slot_prefix
        self.file_prefix = file_prefix
        self.data_extension = data_extension
        self.include_filenames = include_filenames
        self.include_slot = include_slot
        self.default_version_id = default_version_id
        self.default_release = default_release
        self.default_output_dir = default_output_dir
        self.default_start_date = default_start_date


PRODUCTS: dict[str, ProductConfig] = {
    "euv-minute": ProductConfig(
        key="euv-minute",
        description="MAVEN EUV modeled minute irradiance spectra",
        urn_prefix="urn:nasa:pds:maven.euv.modelled:data.minute.spectra:",
        slot_prefix="/data/maven-euv-modelled/data/minute",
        file_prefix="mvn_euv_l3_minute_",
        data_extension="cdf",
        include_filenames=True,
        include_slot=True,
        default_version_id="24.0",
        default_release="v17_r02",
        default_output_dir="data/maven/euv/_maven.euv.modelled",
        default_start_date=dt.date(2014, 11, 10),
    ),
    "swia-fine-arc-3d": ProductConfig(
        key="swia-fine-arc-3d",
        description="MAVEN SWIA calibrated fine arc 3D distributions",
        urn_prefix="urn:nasa:pds:maven.swia.calibrated:data.fine_arc_3d:",
        slot_prefix="/data/maven-swia-calibrated/data/fine_arc_3d",
        file_prefix="mvn_swi_l2_finearc3d_",
        data_extension="cdf",
        include_filenames=True,
        include_slot=True,
        default_version_id="2.0",
        default_release="v02_r00",
        default_output_dir="data/maven/swia/fine_arc_3d",
        default_start_date=dt.date(2014, 7, 7),
    ),
    "insitu-kp": ProductConfig(
        key="insitu-kp",
        description="MAVEN Insitu Key Parameters",
        urn_prefix="urn:nasa:pds:maven.insitu.calibrated:data.kp:",
        slot_prefix="/data/maven-insitu-calibrated/data",
        file_prefix="mvn_kp_insitu_",
        data_extension="tab",
        include_filenames=False,
        include_slot=False,
        default_version_id="1.1,39.0",
        default_release="v13_r03,v13_r02,v13_r01,v12_r01,v11_r01",
        default_output_dir="data/maven/insitu/kp",
        default_start_date=dt.date(2014, 10, 1),
    ),
}


def parse_date(value: str) -> dt.date:
    try:
        return dt.date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date: {value}") from exc


def build_url(
    config: ProductConfig, date: dt.date, version_id: str, release: str
) -> str:
    ymd = date.strftime("%Y%m%d")
    params = {
        "id": (
            f"{config.urn_prefix}{config.file_prefix}{ymd}::{version_id}"
        ),
    }
    if config.include_slot:
        params["slot"] = (
            f"{config.slot_prefix}/{date.year:04d}/{date.month:02d}"
        )
    if config.include_filenames:
        params["file_name"] = f"{config.file_prefix}{ymd}_{release}.xml"
        params["data_file"] = (
            f"{config.file_prefix}{ymd}_{release}.{config.data_extension}"
        )
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


def sample_dates_range(
    start_date: dt.date, end_date: dt.date, samples: int
) -> tuple[list[dt.date], float]:
    if end_date < start_date:
        raise ValueError("end_date must be on or after start_date")
    if samples <= 1:
        return [start_date], 0.0
    span = (end_date - start_date).days
    step = span / (samples - 1)
    seen: set[dt.date] = set()
    dates: list[dt.date] = []
    for i in range(samples):
        date = start_date + dt.timedelta(days=round(i * step))
        if date not in seen:
            dates.append(date)
            seen.add(date)
    return dates, step


def download_one(
    config: ProductConfig,
    date: dt.date,
    output_dir: Path,
    version_id: str,
    release: str,
    keep_zip: bool,
    dry_run: bool,
    timeout: int,
    debug: bool,
) -> bool:
    ymd = date.strftime("%Y%m%d")
    base_name = f"{config.file_prefix}{ymd}_{release}"
    xml_path = output_dir / f"{base_name}.xml"
    data_path = output_dir / f"{base_name}.{config.data_extension}"

    if xml_path.exists() and data_path.exists():
        print(f"skip {ymd}: already have .xml and .{config.data_extension}")
        return True

    url = build_url(config, date, version_id, release)
    if dry_run:
        print(f"dry-run {ymd}: {url}")
        return True

    zip_path = output_dir / f"{base_name}.zip"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            head = response.read(4)
            if head not in ZIP_MAGIC:
                if debug:
                    tail = response.read()
                    (output_dir / f"{base_name}.error").write_bytes(head + tail)
                print(f"miss {ymd}: non-zip response")
                return False
            with open(zip_path, "wb") as handle:
                handle.write(head)
                shutil.copyfileobj(response, handle)
    except Exception as exc:
        print(f"error {ymd}: download failed ({exc})")
        return False

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            names = set(archive.namelist())
            xml_name = f"{base_name}.xml"
            data_name = f"{base_name}.{config.data_extension}"
            if xml_name not in names or data_name not in names:
                prefix = f"{config.file_prefix}{ymd}_"
                xml_candidates = [
                    name
                    for name in names
                    if name.startswith(prefix) and name.endswith(".xml")
                ]
                data_candidates = [
                    name
                    for name in names
                    if name.startswith(prefix)
                    and name.endswith(f".{config.data_extension}")
                ]
                if not xml_candidates or not data_candidates:
                    print(f"miss {ymd}: zip missing expected files")
                    return False
                xml_name = sorted(xml_candidates)[0]
                data_name = sorted(data_candidates)[0]
                archive.extract(xml_name, output_dir)
                archive.extract(data_name, output_dir)
                print(
                    f"ok   {ymd}: downloaded ({Path(xml_name).name}, "
                    f"{Path(data_name).name})"
                )
                return True
            archive.extractall(output_dir)
    except zipfile.BadZipFile:
        print(f"miss {ymd}: invalid zip returned")
        return False
    finally:
        if zip_path.exists() and not keep_zip:
            zip_path.unlink()

    if not xml_path.exists() or not data_path.exists():
        print(f"warn {ymd}: expected files not found after extract")
        return False

    print(f"ok   {ymd}: downloaded")
    return True


def build_search_offsets(window_days: int) -> list[int]:
    offsets: list[int] = [0]
    for step in range(1, window_days + 1):
        offsets.append(step)
        offsets.append(-step)
    return offsets


def download_with_search(
    config: ProductConfig,
    date: dt.date,
    output_dir: Path,
    version_id: str,
    release: str,
    keep_zip: bool,
    dry_run: bool,
    timeout: int,
    debug: bool,
    search_window_days: int,
) -> bool:
    offsets = build_search_offsets(search_window_days)
    for offset in offsets:
        candidate = date + dt.timedelta(days=offset)
        ok = download_one(
            config=config,
            date=candidate,
            output_dir=output_dir,
            version_id=version_id,
            release=release,
            keep_zip=keep_zip,
            dry_run=dry_run,
            timeout=timeout,
            debug=debug,
        )
        if ok:
            if offset:
                print(
                    f"note {date.strftime('%Y%m%d')}: "
                    f"used {candidate.strftime('%Y%m%d')}"
                )
            return True
    return False


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


def parse_candidates(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download MAVEN EUV or SWIA sample files from PDS/PPI."
        )
    )
    parser.add_argument(
        "--product",
        choices=sorted(PRODUCTS.keys()),
        default="euv-minute",
        help="Product key to download.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="Reference start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="Optional end date for sampling (YYYY-MM-DD).",
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
        default=None,
        help="PDS version id (defaults depend on --product).",
    )
    parser.add_argument(
        "--release",
        default=None,
        help="Product release string in filenames.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
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
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Network timeout in seconds.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write non-zip responses to *.error for inspection.",
    )
    parser.add_argument(
        "--search-window-days",
        type=int,
        default=0,
        help="If a date is missing, search +/- N days for the nearest file.",
    )

    args = parser.parse_args()

    config = PRODUCTS[args.product]
    start_date = args.start_date or config.default_start_date
    version_ids = parse_candidates(args.version_id) or parse_candidates(
        config.default_version_id
    )
    releases = parse_candidates(args.release) or parse_candidates(
        config.default_release
    )
    output_dir = Path(args.output_dir or config.default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dates = parse_dates_list(args.dates)
    dates += parse_dates_file(args.dates_file)

    if not dates:
        if args.end_date:
            dates, step = sample_dates_range(
                start_date, args.end_date, args.samples
            )
            span_days = (args.end_date - start_date).days
            print(
                f"Sampling {len(dates)} dates across {span_days} days "
                f"(step={step:.2f} days)."
            )
        else:
            dates, step = sample_dates(
                start_date, args.samples, args.martian_year_days
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
        ok = False
        for version_id in version_ids:
            for release in releases:
                ok = download_with_search(
                    config=config,
                    date=date,
                    output_dir=output_dir,
                    version_id=version_id,
                    release=release,
                    keep_zip=args.keep_zip,
                    dry_run=args.dry_run,
                    timeout=args.timeout,
                    debug=args.debug,
                    search_window_days=args.search_window_days,
                )
                if ok:
                    if version_id != version_ids[0] or release != releases[0]:
                        print(
                            "note "
                            f"{date.strftime('%Y%m%d')}: "
                            f"used {version_id}/{release}"
                        )
                    break
            if ok:
                break
        if not ok:
            failures += 1

    if failures:
        print(f"Done with {failures} failures.")
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

