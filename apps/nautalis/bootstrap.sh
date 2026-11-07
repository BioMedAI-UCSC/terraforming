#!/bin/sh
set -eu
mkdir -p "$HOME"
exec python /opt/nautalis/run.py "$@"
