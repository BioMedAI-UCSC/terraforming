# Empirical benchmark — tform vs. NASA Ames FMS GCM (equatorial, current-Mars)

First **head-to-head numerical run** of this framework's `tform` single-column
model against the NASA Ames FMS Mars GCM on a matched configuration.
July 2026. Empirical companion to the conceptual
[amesgcm-comparison.md](amesgcm-comparison.md) and
[mars-model-landscape.md](mars-model-landscape.md).

**Bottom line up front:** on a matched current-Mars / Ls≈0 / ~705 Pa / 10-sol
setup, `tform` reproduces the GCM's **nighttime radiative floor to within
0.1 K** and the **diurnal-mean surface temperature to within ~4 K** — genuinely
good for a 0-D energy-balance model. Its one material weakness is an
**under-amplified daytime response**: the GCM's single-column diurnal swing is
~113 K vs. tform's ~91 K, so tform runs ~22 K cool at local noon. Pressure is
off by ~50 Pa with the wrong variability structure (tform has no topography or
dynamical redistribution).

---

## 1. Configuration (matched as closely as the two models allow)

| Setting | GCM (Ames FMS) | tform |
|---|---|---|
| Scenario | current Mars, dust + CO₂ cycle | `current-mars` preset |
| Season | Ls 0.26° → 4.87° (vernal equinox) | `--ls 0` |
| Ref. surface pressure | 705 Pa (`PREF`) | `--pressure 705` |
| Length | 10 sols | `--sols 10 --type multi` |
| Integration | FV3 dycore, dt_atmos=924 s | `--accuracy accurate` (RK4), dt=300 s |
| Grid | C24 global cube-sphere, 56 levels | 0-D column (45°N / equator / 40°S @137°E) |
| Albedo | spatially-varying TES map | fixed 0.25 |
| Greenhouse | full radiative transfer | factor 1.02 |

GCM run: `Ames/runs/run_10sol/`. tform run:
`terraforming/outputs/gcm_benchmark/`. Benchmark script + figure:
`Ames/runs/benchmark_tform.py` → `Ames/runs/plots/05_benchmark_vs_tform.png`.

Comparison is **GCM equatorial belt** (cube-sphere tiles 1,2,4,5; tiles 3 & 6
are the poles) vs. **tform equator column**.

---

## 2. Results (equatorial, single column)

| Metric | GCM (Ames FMS) | tform | Δ (GCM − tform) |
|---|---|---|---|
| Diurnal T **min** (night) | 179.9 K | 179.8 K | **+0.1 K** |
| Diurnal T **max** (day) | 293.2 K | 271.1 K | +22.1 K |
| Diurnal T **mean** | 224.8 K | 221.1 K | +3.7 K |
| Diurnal **ΔT** | 113.3 K | 91.3 K | +22.0 K |
| Mean surface P | 672 Pa | 721 Pa | −49 Pa |
| Diurnal-curve RMSE | — | — | 41.7 K |
| Mean solar flux | n/a (3-D RT) | 556 W/m² | — |

**Headline:** two independent models, built on entirely different physics,
agree on the nighttime floor (0.1 K) and the daily mean (3.7 K). They diverge
on the daytime peak — the GCM is ~22 K hotter and swings ~22 K wider.

---

## 3. Two methodology caveats (documented so they aren't rediscovered)

1. **Spatial-average artifact.** FMS's `diurnal24` diagnostic bins by *model
   (universal) time*, not *local solar time*. Averaging the GCM diurnal field
   over all longitudes therefore cancels the day/night signal and produces a
   fake ΔT of only ~6.7 K. The correct apples-to-apples comparison against a
   point model is a **single GCM column**, which recovers the real ~113 K
   swing. `benchmark_tform.py` picks a representative low-latitude column
   (tile 1, j=9, i=20; amplitude 113 K).
2. **Phase is not comparable in this diagnostic.** For the same reason, the
   GCM diurnal curve's "hour" axis is not local-solar at a given column, so it
   is phase-shifted vs. tform's symmetric sinusoid. Amplitude / min / max /
   mean are comparable; **timing is not** until a local-time diagnostic is added.

---

## 4. What can be improved

### tform side (where the real discrepancies are)
- **Daytime peak too low / diurnal swing too narrow (~22 K).** tform's
  surface response under-heats at local noon — consistent with a
  surface heat capacity / thermal inertia that is too large, damping the peak.
  Validate against REMS via the existing `gale-crater` preset (ground-truth).
- **Pressure ~50 Pa high, wrong variability sign.** tform ramps 690→752 Pa
  monotonically (pure CO₂ mass balance); the GCM's equatorial pressure sits
  lower and is shaped by the diurnal tide + topography. tform has neither
  topography nor dynamical pressure redistribution.
- **No horizontal transport or dust radiative feedback** → tform cannot
  reproduce the GCM belt's 140–295 K spatial spread. Relevant for feature
  B7 (1-D banded EBM) and any dust coupling.

### GCM side (for a cleaner benchmark, not correctness)
- **Add a local-time diurnal diagnostic** (or dump hourly `atmos_daily`
  snapshots and fold by each column's longitude) so diurnal *phase* becomes
  comparable.
- **Interpolate to lat/lon** (`diagnostics/runpinterp.csh` / amescap) so a
  true "equator, 137°E" column matches tform's exact coordinate instead of a
  proxy belt.
- **Run longer (≥30 sols) at higher C-number** — 10 sols at C24 is smoke-test
  resolution; daytime peak and dust field are not spun up.

### The comparison itself
- Match **exact lat/lon/elevation** (tform equator = 137°E, 0 m; GCM proxy
  column maps to an unknown cell — `grid_spec` lat/lon came out zero for this
  short run).
- **Isolate physics from boundary conditions:** feed the GCM's local albedo /
  thermal inertia into tform, so the residual reflects *physics* differences
  (RT, transport) rather than differing surface maps.

---

## 5. Takeaways for the thesis framing

- A 0-D EBM matching a 3-D GCM's nighttime and mean surface T to a few K is a
  **defensible fidelity claim** for the fast-mode model — worth a validation
  figure.
- The **daytime-peak / diurnal-amplitude gap is the single most improvable
  physics target** and the cleanest story for a "fast model + learned/tuned
  correction" narrative (cf. differentiable-framework features).
- Pressure realism needs **topography**, which is inherently a ≥1-D concern
  (feature B7) — do not expect the 0-D model to close it.

**Next steps:** (a) exact-coordinate re-run (interpolate GCM → lat/lon, run
tform at that precise point); (b) `gale-crater` preset vs. REMS ground truth;
(c) tune tform surface heat capacity to close the daytime peak and re-measure.
