---
title: Thermal-tide diagnostic overlay and the closed CO₂ mass budget
date: 2026-07-08
target: package/src/celestials/planets/mars.py :: observed_surface_pressure, compute_derivatives, compute_fast_physics; src/engine/diagnostics.py
domain: [ode, algebraic, dimensional]
variables: [P, P_obs, A_tide, ω, φ, t, M_atm, M_ice, R_esc, g, R]
status: verified
---

## Summary

Reviews the pressure refactor that removes the prescribed thermal-tide term
from the prognostic pressure ODE and reapplies it as a closed-form diagnostic
overlay, and the resulting exact CO₂ mass budget enforced by
`engine/diagnostics.py`. **Verdict: verified** — the overlay is the exact
time integral of the removed ODE term (so observed pressure is analytically
unchanged, and *more* accurate than its former discretisation), and with the
tide out of the ledger the budget
ΔM_atm + ΔM_ice + R_esc·Δt − M_injected = 0 closes identically in both
integration modes (measured residuals < 10⁻⁶ Pa-equivalent vs the previous
−0.139 Pa over 10 sols, and ±30 Pa instantaneous contamination).

## Variables and Units

| Symbol | Meaning | SI Unit | Value used in example |
|--------|---------|---------|----------------------|
| `P` | prognostic (global-mean, mass-budget) surface pressure | Pa | 610.0 |
| `P_obs` | observed (local) surface pressure | Pa | — |
| `A_tide` | tide amplitude (`MARS_THERMAL_TIDE_PA`) | Pa | 30.0 |
| `ω` | 2π / rotation period (sol) | s⁻¹ | 7.0779×10⁻⁵ |
| `φ` | tide phase (`MARS_THERMAL_TIDE_PHASE`) | rad | −0.7π |
| `M_atm` | hydrostatic atmospheric mass P·4πR²/g | kg | 2.3672×10¹⁶ |
| `R_esc` | MAVEN non-thermal escape rate | kg s⁻¹ | 0.2 |
| `g` | surface gravity | m s⁻² | 3.72076 |
| `R` | planetary radius | m | 3.3895×10⁶ |

## Physical Law / Starting Point

> **Law (conservation of mass, two-reservoir CO₂ system):** on sub-millennial
> timescales the only CO₂ reservoirs are the atmosphere and the polar caps,
> with a constant escape sink; a *local* pressure oscillation (the thermal
> tide — a planetary-scale wave driven by solar heating) redistributes mass
> around the planet and must not appear in the global mass ledger.
> **Source**: Leighton & Murray (1966), *Science* 153, 136–144; tide
> phenomenology: Zurek et al., in *Mars* (Kieffer et al., eds., 1992), ch. 26.

## Derivation

### Step 1: The old formulation mixed a local signal into the global ledger

The prognostic ODE previously integrated

$$
\frac{dP}{dt} = -\frac{R_{esc}\,g}{4\pi R^2}
               \;-\; \frac{g}{4\pi R^2}\frac{dM_{ice}}{dt}
               \;-\; A_{tide}\,\omega\,\sin(\omega t + \varphi).
$$

The first two terms are genuine mass fluxes. The third is not: integrated over
the planet a tide carries zero net mass, so "global atmospheric mass"
oscillated by ±A_tide·4πR²/g ≈ ±1.16×10¹⁵ kg twice per sol, and no
conservation statement could hold. `[UNIT MISMATCH — semantic]`: the term has
pressure-rate units but represents transport, not source.

### Step 2: Exact integral of the removed term (fundamental theorem of calculus)

$$
\int_0^{t} -A_{tide}\,\omega\,\sin(\omega \tau + \varphi)\,d\tau
= \Big[A_{tide}\cos(\omega\tau+\varphi)\Big]_0^{t}
= A_{tide}\left[\cos(\omega t + \varphi) - \cos\varphi\right].
$$

Therefore defining the prognostic pressure without the tide term and the
observed pressure as

$$
\boxed{\;P_{obs}(t) \;=\; P(t) \;+\; A_{tide}\left[\cos(\omega t+\varphi)-\cos\varphi\right]\;}
$$

reproduces the old trajectory **exactly in continuous time** — the split is a
change of bookkeeping, not of physics. The overlay is evaluated in closed
form, so it is *exact*, whereas the old formulation sampled the sine at the
start of each step (left-endpoint rule inside FAST mode; frozen-substep RK4),
accumulating O(dt) phase error. At t = 0 the bracket vanishes, so
P_obs(0) = P(0) preserves initial-condition semantics.

### Step 3: The closed budget

With the tide removed, both integrators advance P only by the two flux terms,
and the ice ledger receives exactly the negative of the exchange flux, so for
any window [t₀, t₁]:

$$
\underbrace{\Delta\!\left(\frac{P\,4\pi R^2}{g}\right)}_{\Delta M_{atm}}
+\;\Delta M_{ice}
+\;R_{esc}\,(t_1-t_0)
-\;M_{injected}
\;=\;0 ,
$$

identically for FAST (both updates use the same `dMice` tensor) and for RK4
(the budget is a fixed linear combination of the state components, and RK4 is
linear in the RHS). Residuals can only come from float64 rounding and the
`clamp(min=0)` floors, which are inactive in non-collapse regimes.

**Code mapping** (`mars.py :: observed_surface_pressure`):
```python
tide = self._TIDE_PA * (
    torch.cos(omega * self.elapsed_time + self._TIDE_PHASE)
    - torch.cos(self._TIDE_PHASE)
)
return self.atmosphere.surface_pressure + tide
```
[VERIFIED] — matches Step 2 term-for-term; `engine/diagnostics.py ::
mass_budget_residual_kg` implements Step 3 verbatim.

### Step 4: Composition single-source-of-truth (algebraic identity)

All bulk pressure change is CO₂ (inert species neither condense nor escape at
these rates), so after every pressure update

$$
P_{CO_2} = P_{total} - \sum_{i\neq CO_2} P_i ,
$$

maintained by `_sync_composition_co2()` (fast path, `unpack_state`,
inject/decay). Injection adds equal amounts to a partial and to the total, so
the identity is preserved through interventions by construction.

## Worked Numerical Example

**Inputs**: default Mars, caps zeroed, dt = 3600 s, run 0.5 sol
(t = 44,387.622 s).

1. ω = 2π / 88,775.244 = 7.07791×10⁻⁵ s⁻¹; ωt = π.
2. Overlay = 30·[cos(π + (−0.7π)) − cos(−0.7π)] = 30·[cos(0.3π) − cos(0.7π)]
   = 30·[0.587785 − (−0.587785)] = **+35.267 Pa**.
3. Old formulation (audit measurement): prognostic P moved **+28.94 Pa** at
   this instant — the same tide signal, short of the analytic +35.27 Pa by the
   left-endpoint discretisation error (dt = 1 h ⇒ ~9° phase lag), *plus* a
   −0.077 Pa genuine condensation flux.
4. New formulation (test measurement): prognostic ΔP = **−0.0770 Pa**, equal
   to −ΔM_ice·g/(4πR²) to 10⁻⁶ Pa; `P_obs − P` = **+35.267 Pa**, matching
   step 2 to 10⁻⁹ Pa.

**Result**: budget residual < 10⁻⁶ Pa-equivalent (vs −0.139 Pa/10 sols
before); observed pressure keeps the full diurnal wave.
**Relative error**: overlay vs closed form < 10⁻¹²; budget closure limited
only by float64 rounding.

## Theorems, Rules, and Laws Used

### Fundamental theorem of calculus
- **Statement**: ∫₀ᵗ f′(τ)dτ = f(t) − f(0).
- **Wikipedia**: https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus
- **YouTube**: https://www.youtube.com/watch?v=rfG8ce4nNh0 (3Blue1Brown — Essence of Calculus ch. 8)
- **Textbook**: Stewart, *Calculus: Early Transcendentals*, 8th ed., §5.3.
- **Applied here**: Step 2 — the overlay is the exact antiderivative of the removed ODE term.

### Conservation of mass
- **Statement**: mass moves between reservoirs; it is not created or destroyed.
- **Wikipedia**: https://en.wikipedia.org/wiki/Conservation_of_mass
- **YouTube**: no suitable single verified lecture found — flagged rather than guessed; see textbook.
- **Textbook**: Jacobson, *Fundamentals of Atmospheric Modeling*, 2nd ed., ch. 3.
- **Applied here**: Steps 1 & 3 — the tide is transport, not a source; the two-reservoir budget must close.

### Hydrostatic column mass
- **Statement**: the mass of an atmospheric column of surface pressure P is P/g per unit area; globally M = P·4πR²/g.
- **Wikipedia**: https://en.wikipedia.org/wiki/Atmospheric_pressure
- **YouTube**: no suitable single verified lecture found — flagged rather than guessed; see textbook.
- **Textbook**: Wallace & Hobbs, *Atmospheric Science*, 2nd ed., §3.2.
- **Applied here**: converts pressure to mass in the budget and in `atmospheric_mass` at init (fixes the 2.5×10¹⁶ vs 2.367×10¹⁶ kg inconsistency).

### Linearity of Runge–Kutta methods
- **Statement**: RK updates are linear in the RHS, so any fixed linear functional of the state that the RHS conserves is conserved by the discrete step.
- **Wikipedia**: https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
- **YouTube**: no suitable single verified lecture found — flagged rather than guessed; see textbook.
- **Textbook**: Hairer, Nørsett & Wanner, *Solving ODEs I*, 2nd ed., §II.1; conservation: §IV (linear invariants are preserved by all RK methods).
- **Applied here**: Step 3 — why the budget closes exactly in ACCURATE mode too.

## Findings and Recommendations

### [VERIFIED] Overlay equivalence and budget closure
- Observed pressure analytically identical to the old trajectory (now exact);
  budget residual < 10⁻⁶ Pa-equivalent in both modes; tests in
  `tests/engine/test_diagnostics.py` and `tests/mars/basic/test_pressure_budget.py`.

### [APPROXIMATION WARNING] the tide remains prescribed, not emergent
- Amplitude (30 Pa) and phase (−0.7π) are hand-set constants; observed tide
  amplitude varies with season and dust loading. Calibration against Viking
  (ML4/F1) should fit both — now safely, since they no longer touch mass.

### [NOTE] downstream physics sees the mean
- `delta_F_from_composition` now divides by the tide-free total: GHG mixing
  ratios no longer wobble ~4 % diurnally with the tide — physically correct
  (a tide moves all species together, leaving mixing ratios unchanged).

### [NOTE] remaining clamp floors
- `P.clamp(min=0)` and the ice floors can absorb mass exactly at collapse;
  inactive in current-Mars regimes, flagged for the smooth-gates follow-up.

## Cross-references
- [tanh-ice-sublimation-gate.md](tanh-ice-sublimation-gate.md) — same clamp/gate class
- [docs/ideas/master-task-list.md](../ideas/master-task-list.md) — Track A-3 task 18 (F2)
- Pressure audit (session record) — verified bugs #1, #2, #4, #5 fixed here

## References
- Leighton, R. B. & Murray, B. C. (1966). *Science* 153, 136–144.
- Zurek, R. W. et al. (1992). In *Mars*, Univ. of Arizona Press, ch. 26 (tides).
- Wallace, J. M. & Hobbs, P. V. (2006). *Atmospheric Science*, 2nd ed.
- Hairer, Nørsett & Wanner (1993). *Solving Ordinary Differential Equations I*.
- Jakosky, B. M. et al. (2018). *Icarus* 315, 146–157 (MAVEN escape rate).
