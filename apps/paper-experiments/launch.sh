#!/bin/sh
# Run from the repository root. Stops if staging fails; never switches context.
set -eu
rtk proxy python3 apps/paper-experiments/jobs.py stage > /tmp/terraforming-paper-stage.json
rtk kubectl --context nautilus -n razvanlab apply -f /tmp/terraforming-paper-stage.json
rtk kubectl --context nautilus -n razvanlab wait --for=condition=complete job/paper-stage --timeout=168h
rtk proxy python3 apps/paper-experiments/jobs.py all > /tmp/terraforming-paper-all.json
rtk kubectl --context nautilus -n razvanlab apply -f /tmp/terraforming-paper-all.json
