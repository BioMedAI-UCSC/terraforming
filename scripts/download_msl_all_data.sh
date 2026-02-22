#!/usr/bin/env bash
set -euo pipefail

# Full MSL (Curiosity) archive mirror from PDS Imaging node.
# This can take a very long time and large disk space.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$ROOT_DIR/data/msl/raw_pds_imaging"
LOG_DIR="$ROOT_DIR/data/msl"
LOG_FILE="$LOG_DIR/download_msl_all_$(date +%Y%m%d_%H%M%S).log"
ROOT_URL="https://planetarydata.jpl.nasa.gov/img/data/msl/"
MANIFEST="$LOG_DIR/msl_directories_manifest.txt"

mkdir -p "$TARGET_DIR" "$LOG_DIR"

echo "Starting MSL full mirror from: $ROOT_URL"
echo "Target directory: $TARGET_DIR"
echo "Log file: $LOG_FILE"

python3 - <<'PY' > "$MANIFEST"
import re, urllib.request
root = "https://planetarydata.jpl.nasa.gov/img/data/msl/"
html = urllib.request.urlopen(root, timeout=90).read().decode("utf-8", "replace")
links = sorted(set(re.findall(r'href=["\\\']([^"\\\']+/)["\\\']', html, flags=re.I)))
for link in links:
    if link.startswith("MSL") or link.startswith("msl_"):
        print(root + link)
PY

echo "Directory manifest saved: $MANIFEST"
echo "Beginning per-directory recursive mirror..."

while IFS= read -r dir_url; do
  [ -z "$dir_url" ] && continue
  echo ">>> Mirroring $dir_url" | tee -a "$LOG_FILE"
  wget \
    --recursive \
    --continue \
    --timestamping \
    --no-parent \
    --directory-prefix "$TARGET_DIR" \
    --no-host-directories \
    --cut-dirs=2 \
    --execute robots=off \
    --wait=0.2 \
    --random-wait \
    --tries=20 \
    --retry-connrefused \
    --timeout=30 \
    --reject "index.html*" \
    "$dir_url" | tee -a "$LOG_FILE"
done < "$MANIFEST"

echo "MSL mirror command finished. See log: $LOG_FILE"
