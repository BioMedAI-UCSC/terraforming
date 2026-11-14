# Historical raw results

[Public release](https://github.com/BioMedAI-UCSC/terraforming/releases/tag/paper-v1-historical-results-20261114)

`release.json` links the saved one- and ten-year results. The per-run manifests
record every available file, its size and hash, the recorded source revision,
and missing provenance. Archive members were read back and hashed. Uploaded
asset sizes and GitHub SHA-256 digests matched the local files; the public
ten-year manifest was fetched without authentication and its hash checked.

The original runs used older physics and unit contracts. Keep their original
metadata and reports. Do not use their nondimensional restarts as current-model
states. Missing original GPU identity and input records cannot be inferred from
the new A100 campaign.

The year-nine/year-ten audit uses global-mean diagnostics. It interpolates both
years onto common integer-degree solar longitudes, without extrapolation.
Pressure has RMS change 0.1553463 Pa and signed mean change +0.0510829 Pa under
this explicit protocol. These are different statistics. The original report
gives 0.1562763 Pa and +0.0506286 Pa on an unspecified interpolation grid and is
retained separately. Neither calculation gives a spatial RMS. Deep-soil
temperature still changes by 0.6761993 K RMS, so equilibrium is not established.

Reproduce after unpacking the ten-year archive:

```sh
rtk proxy .venv/bin/python scripts/audit_paper_results.py \
  --long-run /path/to/mars-ten-years-dt450-float64-stage-01 \
  --output /tmp/year-reconciliation
```

The release contains saved checkpoints and final maps, not every integration
step. The public files support historical reanalysis, not corrected-model
validation or a matched performance comparison.
