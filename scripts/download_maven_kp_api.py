#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
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
FILE_INFO_URL = (
    "https://lasp.colorado.edu/maven/sdc/public/files/api/v1/"
    "search/science/fn_metadata/file_info"
)
MAG_L2_BASE_URL = "https://lasp.colorado.edu/maven/sdc/public/data/sci/mag/l2"
ZIP_MAGIC = (b"PK\x03\x04", b"PK\x05\x06")
VALID_KP_LEVELS = {"insitu", "iuvs"}
VALID_NGI_LEVELS = {"l1a", "l1b", "l2", "l3"}
VALID_EUV_PLANS = {"l2", "daily", "minute"}
VALID_STA_PLANS = {
    "2a",
    "c0",
    "c2",
    "c4",
    "c6",
    "c8",
    "ca",
    "cc",
    "cd",
    "ce",
    "cf",
    "d0",
    "d1",
    "d4",
    "d6",
    "d7",
    "d8",
    "d9",
    "da",
    "db",
}
VALID_SWI_PLANS = {
    "coarsearc3d",
    "coarsesvy3d",
    "finearc3d",
    "finesvy3d",
    "onboardsvymom",
    "onboardsvyspec",
}
VALID_SWE_PLANS = {"arcpad", "svypad", "arc3d", "svy3d", "svyspec"}
VALID_SEP_PRODUCTS = {
    "s1-raw-svy-full",
    "s2-raw-svy-full",
    "s1-cal-svy-full",
    "s2-cal-svy-full",
    "anc",
}
VALID_LPW_PLANS = {
    "act",
    "adr",
    "atr",
    "euv",
    "hsbmhf",
    "hsbmlf",
    "hsbmmf",
    "hsk",
    "pas",
    "spechfact",
    "spechfpas",
    "speclfact",
    "speclfpas",
    "specmfact",
    "specmfpas",
    "swp1",
    "sw2",
    "lpiv",
    "lpnt",
    "mrgscpot",
    "we12",
    "we12bursthf",
    "we12burstlf",
    "we12burstmf",
    "wn",
    "wspecact",
    "wspecpas",
}
VALID_MAG_PLANS = {"all", "pl", "pc", "ss"}
VALID_MAG1S_PLANS = {"pc1s", "ss1s", "pl1s"}


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


def day_chunks(start: dt.date, end: dt.date, days: int) -> list[tuple[dt.date, dt.date]]:
    chunks: list[tuple[dt.date, dt.date]] = []
    cursor = start
    step = max(1, days)
    while cursor <= end:
        chunk_end = min(end, cursor + dt.timedelta(days=step - 1))
        chunks.append((cursor, chunk_end))
        cursor = chunk_end + dt.timedelta(days=1)
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


def build_url(
    instrument: str, product: str, start: dt.date, end: dt.date
) -> str:
    params = {
        "instrument": instrument,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    if instrument in {"kp", "ngi"}:
        params["level"] = product
    elif instrument == "euv":
        # LASP docs: L2 uses default query (omit plan parameter).
        if product != "l2":
            params["plan"] = product
    elif instrument == "sta":
        params["level"] = "l2"
        params["plan"] = product
    elif instrument == "swi":
        params["level"] = "l2"
        params["plan"] = product
    elif instrument == "swe":
        params["level"] = "l2"
        params["plan"] = product
    elif instrument == "sep":
        if product == "anc":
            params["level"] = "anc"
        else:
            params["level"] = "l2"
            params["descriptor"] = product
    elif instrument == "lpw":
        params["plan"] = product
    elif instrument == "mag":
        # MAG L2 download endpoint currently responds with empty results (HTTP 204)
        # for plan-filtered queries in many windows. Use broad L2 query and filter
        # product types post-download if needed.
        params["level"] = "l2"
    else:
        raise ValueError(f"Unsupported instrument: {instrument}")
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
                    downloaded = len(head)
                    # Show sparse progress for long-running chunked responses.
                    report_step = 100 * 1024 * 1024
                    next_report = report_step
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            mib = downloaded / (1024 * 1024)
                            print(f"    downloaded ~{mib:.0f} MiB")
                            next_report += report_step
                return True
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"error: {url} ({exc})")
                return False
            time.sleep(2 * attempt)
    return False


def fetch_bytes(url: str, timeout: int, retries: int) -> bytes | None:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"error: {url} ({exc})")
                return None
            time.sleep(2 * attempt)
    return None


def fetch_file(url: str, out_file: Path, timeout: int, retries: int) -> bool:
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                out_file.parent.mkdir(parents=True, exist_ok=True)
                with out_file.open("wb") as handle:
                    downloaded = 0
                    report_step = 100 * 1024 * 1024
                    next_report = report_step
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        handle.write(chunk)
                        downloaded += len(chunk)
                        if downloaded >= next_report:
                            mib = downloaded / (1024 * 1024)
                            print(f"    downloaded ~{mib:.0f} MiB")
                            next_report += report_step
                return True
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                print(f"error: {url} ({exc})")
                return False
            time.sleep(2 * attempt)
    return False


def list_mag1s_files(
    start: dt.date, end: dt.date, product: str, timeout: int, retries: int
) -> list[str]:
    params = {
        "instrument": "mag",
        "level": "l2",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    url = f"{FILE_INFO_URL}?{urllib.parse.urlencode(params)}"
    data = fetch_bytes(url=url, timeout=timeout, retries=retries)
    if data is None:
        return []
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return []
    files = payload.get("files", [])
    names: list[str] = []
    token = f"{product}_"
    for item in files:
        name = item.get("file_name")
        if isinstance(name, str) and token in name:
            names.append(name)
    return sorted(set(names))


def mag_l2_relative_path(file_name: str) -> Path | None:
    match = re.search(r"_(\d{8})_v\d+_r\d+\.", file_name)
    if not match:
        return None
    yyyymmdd = match.group(1)
    year = yyyymmdd[:4]
    month = yyyymmdd[4:6]
    return Path(year) / month / file_name


def extract_zip(zip_path: Path, out_dir: Path) -> int:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if not m.endswith("/")]
        zf.extractall(out_dir)
    return len(members)


def marker_name(
    instrument: str, product: str, start: dt.date, end: dt.date
) -> str:
    return (
        f"{instrument}_{product}_{start.isoformat()}_{end.isoformat()}.done"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download MAVEN SDC data from LASP API in chunks."
        )
    )
    parser.add_argument(
        "--instrument",
        choices=["kp", "euv", "ngi", "sta", "swi", "swe", "sep", "lpw", "mag", "mag1s"],
        default="kp",
        help=(
            "Instrument family to download (kp, euv, ngi, sta, swi, swe, "
            "sep, lpw, mag, or mag1s)."
        ),
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
        help=(
            "Comma-separated levels. For kp: insitu,iuvs. "
            "For ngi: l1a,l1b,l2,l3."
        ),
    )
    parser.add_argument(
        "--plans",
        default="minute",
        help=(
            "Comma-separated plans. For euv: l2,daily,minute "
            "(for l2, plan is omitted in API). For sta: "
            "2a,c0,c2,c4,c6,c8,ca,cc,cd,ce,cf,d0,d1,d4,d6,d7,d8,d9,da,db. "
            "For swi: coarsearc3d,coarsesvy3d,finearc3d,finesvy3d,onboardsvymom,onboardsvyspec. "
            "For swe: arcpad,svypad,arc3d,svy3d,svyspec. "
            "For sep: s1-raw-svy-full,s2-raw-svy-full,s1-cal-svy-full,s2-cal-svy-full,anc. "
            "For lpw: act,adr,atr,euv,hsbmhf,hsbmlf,hsbmmf,hsk,pas,spechfact,spechfpas,speclfact,speclfpas,specmfact,specmfpas,swp1,sw2,lpiv,lpnt,mrgscpot,we12,we12bursthf,we12burstlf,we12burstmf,wn,wspecact,wspecpas. "
            "For mag: all,pl,pc,ss (MAG currently downloads as unfiltered L2). "
            "For mag1s: pc1s,ss1s,pl1s."
        ),
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
        "--euv-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for EUV plan downloads.",
    )
    parser.add_argument(
        "--ngi-chunk-months",
        type=int,
        default=3,
        help="Chunk size in months for NGIMS level downloads.",
    )
    parser.add_argument(
        "--sta-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for STATIC L2 plan downloads.",
    )
    parser.add_argument(
        "--swi-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for SWIA L2 plan downloads.",
    )
    parser.add_argument(
        "--swe-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for SWEA L2 plan downloads.",
    )
    parser.add_argument(
        "--swe-chunk-days",
        type=int,
        default=7,
        help=(
            "Chunk size in days for SWEA L2 downloads. "
            "Used instead of --swe-chunk-months when positive."
        ),
    )
    parser.add_argument(
        "--sep-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for SEP downloads.",
    )
    parser.add_argument(
        "--lpw-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for LPW downloads.",
    )
    parser.add_argument(
        "--mag-chunk-months",
        type=int,
        default=1,
        help="Chunk size in months for MAG L2 plan downloads.",
    )
    parser.add_argument(
        "--mag-chunk-days",
        type=int,
        default=7,
        help=(
            "Chunk size in days for MAG L2 downloads. "
            "Used instead of --mag-chunk-months when positive."
        ),
    )
    parser.add_argument(
        "--mag1s-chunk-days",
        type=int,
        default=7,
        help="Chunk size in days for MAG reduced 1s products (pc1s/ss1s/pl1s).",
    )
    parser.add_argument(
        "--output-dir",
        default="data/maven/sdc_api_full",
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

    products: list[str]
    if args.instrument == "kp":
        products = [x.strip().lower() for x in args.levels.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_KP_LEVELS]
        if invalid:
            raise SystemExit(f"Invalid KP levels: {', '.join(invalid)}")
    elif args.instrument == "ngi":
        products = [x.strip().lower() for x in args.levels.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_NGI_LEVELS]
        if invalid:
            raise SystemExit(f"Invalid NGIMS levels: {', '.join(invalid)}")
    elif args.instrument == "euv":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_EUV_PLANS]
        if invalid:
            raise SystemExit(f"Invalid EUV plans: {', '.join(invalid)}")
    elif args.instrument == "sta":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_STA_PLANS]
        if invalid:
            raise SystemExit(f"Invalid STATIC plans: {', '.join(invalid)}")
    elif args.instrument == "mag":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_MAG_PLANS]
        if invalid:
            raise SystemExit(f"Invalid MAG plans: {', '.join(invalid)}")
        # LASP MAG API plan filtering often returns empty 204 responses; use one
        # broad L2 pull to keep behavior reliable.
        if products != ["all"]:
            print(
                "note: MAG plan filtering is not reliable via download_zip API; "
                "falling back to unfiltered MAG L2 download."
            )
        products = ["all"]
    elif args.instrument == "mag1s":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_MAG1S_PLANS]
        if invalid:
            raise SystemExit(f"Invalid MAG1S plans: {', '.join(invalid)}")
    elif args.instrument == "swe":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_SWE_PLANS]
        if invalid:
            raise SystemExit(f"Invalid SWEA plans: {', '.join(invalid)}")
    elif args.instrument == "sep":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_SEP_PRODUCTS]
        if invalid:
            raise SystemExit(f"Invalid SEP products: {', '.join(invalid)}")
    elif args.instrument == "lpw":
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_LPW_PLANS]
        if invalid:
            raise SystemExit(f"Invalid LPW plans: {', '.join(invalid)}")
    else:
        products = [x.strip().lower() for x in args.plans.split(",") if x.strip()]
        invalid = [x for x in products if x not in VALID_SWI_PLANS]
        if invalid:
            raise SystemExit(f"Invalid SWIA plans: {', '.join(invalid)}")

    out_root = Path(args.output_dir)
    zips_dir = out_root / "_zips"
    marker_dir = out_root / "_markers"
    out_root.mkdir(parents=True, exist_ok=True)
    zips_dir.mkdir(parents=True, exist_ok=True)
    marker_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    failed = 0

    for product in products:
        product_dir = out_root / args.instrument / product
        product_dir.mkdir(parents=True, exist_ok=True)
        if args.instrument == "kp":
            if product == "insitu":
                chunks = month_chunks(
                    args.start_date, args.end_date, max(1, args.insitu_chunk_months)
                )
            else:
                chunks = year_chunks(
                    args.start_date, args.end_date, max(1, args.iuvs_chunk_years)
                )
        elif args.instrument == "ngi":
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.ngi_chunk_months)
            )
        elif args.instrument == "sta":
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.sta_chunk_months)
            )
        elif args.instrument == "swi":
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.swi_chunk_months)
            )
        elif args.instrument == "swe":
            if args.swe_chunk_days > 0:
                chunks = day_chunks(
                    args.start_date, args.end_date, args.swe_chunk_days
                )
            else:
                chunks = month_chunks(
                    args.start_date, args.end_date, max(1, args.swe_chunk_months)
                )
        elif args.instrument == "sep":
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.sep_chunk_months)
            )
        elif args.instrument == "lpw":
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.lpw_chunk_months)
            )
        elif args.instrument == "mag":
            if args.mag_chunk_days > 0:
                chunks = day_chunks(
                    args.start_date, args.end_date, args.mag_chunk_days
                )
            else:
                chunks = month_chunks(
                    args.start_date, args.end_date, max(1, args.mag_chunk_months)
                )
        elif args.instrument == "mag1s":
            chunks = day_chunks(
                args.start_date, args.end_date, max(1, args.mag1s_chunk_days)
            )
        else:
            chunks = month_chunks(
                args.start_date, args.end_date, max(1, args.euv_chunk_months)
            )

        print(f"[{args.instrument}:{product}] chunks: {len(chunks)}")
        for start, end in chunks:
            total_chunks += 1
            marker = marker_dir / marker_name(args.instrument, product, start, end)
            if marker.exists():
                print(f"skip {args.instrument}:{product} {start}..{end} (done)")
                continue

            if args.instrument == "mag1s":
                names = list_mag1s_files(
                    start=start,
                    end=end,
                    product=product,
                    timeout=args.timeout,
                    retries=args.retries,
                )
                if args.dry_run:
                    print(
                        f"dry-run {args.instrument}:{product} {start}..{end}: "
                        f"{len(names)} files from file_info"
                    )
                    continue

                print(
                    f"get  {args.instrument}:{product} {start}..{end} "
                    f"({len(names)} files)"
                )
                downloaded_ok = 0
                downloaded_fail = 0
                for idx, name in enumerate(names, start=1):
                    rel = mag_l2_relative_path(name)
                    if rel is None:
                        continue
                    out_file = product_dir / "maven" / "data" / "sci" / "mag" / "l2" / rel
                    if out_file.exists() and out_file.stat().st_size > 0:
                        downloaded_ok += 1
                        continue
                    url = f"{MAG_L2_BASE_URL}/{rel.as_posix()}"
                    print(f"    file {idx}/{len(names)} {name}")
                    ok = fetch_file(
                        url=url,
                        out_file=out_file,
                        timeout=args.timeout,
                        retries=args.retries,
                    )
                    if ok:
                        downloaded_ok += 1
                    else:
                        downloaded_fail += 1

                if downloaded_fail:
                    failed += 1
                    print(
                        f"fail {args.instrument}:{product} {start}..{end}: "
                        f"{downloaded_fail} files failed"
                    )
                    continue

                marker.write_text(
                    f"ok files={downloaded_ok}\n",
                    encoding="utf-8",
                )
                print(
                    f"ok   {args.instrument}:{product} {start}..{end}: "
                    f"{downloaded_ok} files"
                )
                time.sleep(max(0.0, args.sleep_seconds))
                continue

            url = build_url(args.instrument, product, start, end)
            zip_name = (
                f"{args.instrument}_{product}_{start.isoformat()}_"
                f"{end.isoformat()}.zip"
            )
            zip_path = zips_dir / zip_name

            if args.dry_run:
                print(f"dry-run {args.instrument}:{product} {start}..{end}: {url}")
                continue

            print(f"get  {args.instrument}:{product} {start}..{end}")
            ok = fetch_zip(url=url, out_zip=zip_path, timeout=args.timeout, retries=args.retries)
            if not ok:
                print(f"fail {args.instrument}:{product} {start}..{end}")
                failed += 1
                continue

            try:
                count = extract_zip(zip_path, product_dir)
                marker.write_text(f"ok files={count}\n", encoding="utf-8")
                print(
                    f"ok   {args.instrument}:{product} {start}..{end}: "
                    f"{count} files"
                )
            except zipfile.BadZipFile:
                failed += 1
                print(f"fail {args.instrument}:{product} {start}..{end}: bad zip")
                continue
            finally:
                if zip_path.exists() and not args.keep_zip:
                    zip_path.unlink()

            time.sleep(max(0.0, args.sleep_seconds))

    print(f"Done. chunks={total_chunks} failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())


