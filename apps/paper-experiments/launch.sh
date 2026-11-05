#!/bin/sh
# Run from the repository root. Both jobs run independently of this laptop.
set -eu
rtk proxy python3 apps/paper-experiments/jobs.py stage > /tmp/terraforming-paper-stage.json
rtk kubectl --context nautilus -n razvanlab apply -f /tmp/terraforming-paper-stage.json
rtk proxy python3 apps/paper-experiments/jobs.py all > /tmp/terraforming-paper-all.json
rtk kubectl --context nautilus -n razvanlab apply -f /tmp/terraforming-paper-all.json
