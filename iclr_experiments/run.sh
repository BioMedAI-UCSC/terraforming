#!/bin/sh
# Defaults to the complete suite; task names are defined in experiment_tasks.py.
set -eu
if [ "$#" -eq 0 ]; then set -- all; fi
exec python "$(dirname "$0")/run.py" "$@"
