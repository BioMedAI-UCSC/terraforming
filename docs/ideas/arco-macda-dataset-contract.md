# ARCO-MACDA dataset contract

**Verified against the live Hugging Face Zarr store:** 2026-09-16  
**ARCO DOI:** [10.57967/hf/8771](https://doi.org/10.57967/hf/8771)  
**Underlying MACDA v2.0 DOI:** [10.5285/cd037a9ea387438fabf4d674dbe53088](https://doi.org/10.5285/cd037a9ea387438fabf4d674dbe53088)

## Source and license

- Repository: <https://huggingface.co/datasets/ananyo01/ARCO-MACDA>
- Zarr v3 root: `macda_combined.zarr`
- ARCO repository license: CC BY 4.0
- Underlying MACDA v2.0 license: UK Open Government Licence v3.0
- Required attribution: cite both the ARCO conversion and the original MACDA
  v2.0 dataset.

The source is a data-assimilating Mars reanalysis, not direct observational
truth. It combines Mars PCM simulations with TES, THEMIS, and MCS retrievals.

## Verified schema

- Dimensions: `time=96,480`, `lev=35`, `lat=36`, `lon=72`
- Logical uncompressed xarray size: `216,067,346,492 bytes` (about 216 GB)
- Time spacing: exactly `0.083333333333 sol`, or 12 states per sol
- Native 3-D chunks: `120 x 35 x 36 x 72`, corresponding to ten sols
- Native surface chunks: `120 x 36 x 72`
- Latitude: `87.5` to `-87.5` degrees, descending
- Longitude: `-180` to `175` degrees at 5-degree spacing
- Sigma levels: 35 terrain-following midpoints from `0.9995` near the surface
  to `1.0872763e-5` near the model top

Verified variables used by the first model pipeline:

| Variable | Meaning | Units | Dimensions |
|---|---|---|---|
| `temp` | Atmospheric temperature | K | time, lev, lat, lon |
| `uwind` | Zonal wind | m s-1 | time, lev, lat, lon |
| `vwind` | Meridional wind | m s-1 | time, lev, lat, lon |
| `psurf` | Surface pressure | Pa | time, lat, lon |
| `tsurf` | Surface temperature | K | time, lat, lon |
| `coldust` | Visible column dust optical depth | 1 | time, lat, lon |
| `co2ice` | Surface CO2 ice | kg m-2 | time, lat, lon |
| `Ls` | Solar longitude | degree | time |
| `MY_Ls` | Solar-longitude Mars-year label | 1 | time |

Additional verified variables are `dustmmr`, `geop`, `omega`, `swflux`, and
`lwflux`.

## Mars-year boundary audit

The repository describes complete coverage as MY24-35. The live `MY_Ls` array
also contains a partial MY36 tail of 202 states (16.83 sols). Do not treat this
tail as a complete evaluation year.

| MY | Start index | Stop index | States | State count / 12 |
|---:|---:|---:|---:|---:|
| 24 | 0 | 8,023 | 8,023 | 668.58 |
| 25 | 8,023 | 16,046 | 8,023 | 668.58 |
| 26 | 16,046 | 24,069 | 8,023 | 668.58 |
| 27 | 24,069 | 32,092 | 8,023 | 668.58 |
| 28 | 32,092 | 40,320 | 8,228 | 685.67 |
| 29 | 40,320 | 48,139 | 7,819 | 651.58 |
| 30 | 48,139 | 56,162 | 8,023 | 668.58 |
| 31 | 56,162 | 64,185 | 8,023 | 668.58 |
| 32 | 64,185 | 71,848 | 7,663 | 638.58 |
| 33 | 71,848 | 80,231 | 8,383 | 698.58 |
| 34 | 80,231 | 88,255 | 8,024 | 668.67 |
| 35 | 88,255 | 96,278 | 8,023 | 668.58 |
| 36 partial | 96,278 | 96,480 | 202 | 16.83 |

The physical time coordinate remains uniformly spaced. Training code must use
the explicit `MY_Ls` boundaries instead of assuming a fixed number of states per
Mars year.

## Prepared sample

`scripts/stage_arco_macda.py` streams bounded chunks and performs:

1. binary search of the remote `MY_Ls` coordinate;
2. exact cadence validation;
3. daily arithmetic means, with a circular mean for solar longitude;
4. linear interpolation from 35 native sigma levels to 12 Dinosaur midpoints;
5. periodic bilinear interpolation to the T21 32 x 64 Gaussian grid;
6. finite-value validation, per-variable statistics, provenance, and SHA-256.

The verified MY24 sample contains 10 daily states on a `10 x 12 x 32 x 64`
atmospheric grid. Its NetCDF SHA-256 is
`0a973c94bbb49a60b00e1fef305f19053e60a42db5fbf3b5a53f726414873c1d`.
The companion 10-day surface shortwave/longwave radiation sample has SHA-256
`4d3620b514cd4c6e17487c73750f864a4a55cbb500c9d8d57ef3abb2e8e379b9`.
All selected variables in both artifacts have a finite fraction of `1.0`.

The staged time coordinate is a floating-point Martian-sol coordinate with
`units=sol`; it opens with xarray's default settings and does not inherit the
source archive's inapplicable CF datetime/calendar encoding.

## Split contract

- Training: MY24-27 and MY29-31
- Dust-storm out-of-distribution test: MY28, excluded from all fitting and model
  selection
- Validation: MY32-33
- Test: MY34-35
- MY36 partial tail: excluded

The staged MY24 sample verifies the pipeline. It is not a held-out score and
must not be presented as one.
