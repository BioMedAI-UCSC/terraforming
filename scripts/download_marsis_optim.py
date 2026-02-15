#!/usr/bin/env python3
"""
Download MARSIS optimized radargram data and geometry from PDS Geosciences Node.
Only downloads the essential folders for ice mapping baseline.
"""

import os
import re
import ssl
import urllib.request
from pathlib import Path
from html.parser import HTMLParser

# Disable SSL verification for PDS server certificate issues
ssl_ctx = ssl.create_default_context()
ssl_ctx.check_hostname = False
ssl_ctx.verify_mode = ssl.CERT_NONE

BASE_URL = "https://pds-geosciences.wustl.edu/mex/urn-nasa-pds-mex_marsis_optim"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "marsis" / "marsis_optim"


class DirectoryListingParser(HTMLParser):
    """Parse IIS/Apache directory listing HTML to extract file/folder links."""
    def __init__(self, base_path: str):
        super().__init__()
        self.links = []
        self.base_path = base_path  # e.g., "/mex/urn-nasa-pds-mex_marsis_optim/radargram_data/"
    
    def handle_starttag(self, tag, attrs):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    # Skip query strings and parent directory links
                    if value.startswith("?") or value in ("../", "[To Parent Directory]"):
                        continue
                    # Handle absolute paths that match our base path
                    if value.startswith("/"):
                        if value.startswith(self.base_path):
                            # Extract the relative part
                            rel = value[len(self.base_path):]
                            if rel:
                                self.links.append(rel)
                    elif not value.startswith("http"):
                        # Relative path
                        self.links.append(value)


def fetch_directory_listing(url: str) -> list[str]:
    """Fetch and parse a directory listing, returning list of file/folder names."""
    print(f"  Listing: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=60) as resp:
        html = resp.read().decode("utf-8", errors="ignore")
    
    # Extract the path portion for the parser
    from urllib.parse import urlparse
    parsed = urlparse(url)
    base_path = parsed.path
    if not base_path.endswith("/"):
        base_path += "/"
    
    parser = DirectoryListingParser(base_path)
    parser.feed(html)
    return parser.links


def download_file(url: str, dest: Path):
    """Download a single file with resume support."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if already downloaded
    if dest.exists():
        local_size = dest.stat().st_size
        # Get remote size
        req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, context=ssl_ctx, timeout=30) as resp:
                remote_size = int(resp.headers.get("Content-Length", 0))
            if local_size == remote_size and remote_size > 0:
                print(f"  Skip (complete): {dest.name}")
                return
        except Exception:
            pass
    
    print(f"  Downloading: {dest.name}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ssl_ctx, timeout=120) as resp:
        with open(dest, "wb") as f:
            while chunk := resp.read(65536):
                f.write(chunk)


def download_directory(url: str, dest_dir: Path, depth: int = 0, max_subdirs: int = None):
    """Recursively download a directory from PDS."""
    if depth > 5:
        print(f"  Max depth reached, skipping: {url}")
        return
    
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        links = fetch_directory_listing(url)
    except Exception as e:
        print(f"  Error listing {url}: {e}")
        return
    
    subdir_count = 0
    for link in links:
        if link.endswith("/"):
            # It's a subdirectory
            if max_subdirs is not None and subdir_count >= max_subdirs:
                print(f"  (Skipping remaining subdirs, max_subdirs={max_subdirs})")
                break
            subdir_name = link.rstrip("/")
            download_directory(f"{url}{link}", dest_dir / subdir_name, depth + 1)
            subdir_count += 1
        else:
            # It's a file
            file_url = f"{url}{link}"
            file_dest = dest_dir / link
            try:
                download_file(file_url, file_dest)
            except Exception as e:
                print(f"  Error downloading {link}: {e}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download metadata files
    print("Downloading metadata files...")
    for fname in ["readme.txt", "bundle_mex_marsis_optim.xml"]:
        try:
            download_file(f"{BASE_URL}/{fname}", OUTPUT_DIR / fname)
        except Exception as e:
            print(f"  Error: {e}")
    
    # Download radargram_data (essential for ice mapping)
    # NOTE: This is a massive dataset (~200 orbit folders). Download just a sample first.
    print("\nDownloading radargram_data (sample: first 3 orbit folders)...")
    download_directory(f"{BASE_URL}/radargram_data/", OUTPUT_DIR / "radargram_data", max_subdirs=3)
    
    # Download geometry (needed to geolocate radargrams)
    print("\nDownloading geometry...")
    download_directory(f"{BASE_URL}/geometry/", OUTPUT_DIR / "geometry")
    
    print("\nDone!")


if __name__ == "__main__":
    main()

