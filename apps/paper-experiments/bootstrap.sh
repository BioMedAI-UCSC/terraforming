#!/bin/sh
set -eu
mkdir -p "$HOME" /work/source
# One RWO volume, one writer. Never update a checkout while an experiment runs.
exec python /opt/paper/run.py "$@"
