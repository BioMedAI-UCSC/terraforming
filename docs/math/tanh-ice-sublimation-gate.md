---
title: Smooth ice-exhaustion gate — tanh gating of the polar CO₂ sublimation flux
date: 2026-07-08
target: package/src/celestials/planets/mars.py :: _gate_sublimation
domain: [algebraic, numerical]
variables: [dM, M_ice, M_ref, Φ, g, A, σ_SB, ε, L_sub, T_frost]
status: verified
---

## Summary

Reviews the opt-in smooth replacement (`Mars(smooth_gates=True)`) for the hard
ice-exhaustion gate on the polar CO₂ sublimation flux. The hard gate
`torch.where((M_ice ≤ 0) ∧ (dM < 0), 0, dM)` is physically correct but has zero
gradient with respect to every input once a cap empties. The smooth gate
multiplies the sublimating branch by `tanh(M_ice / M_ref)` with
`M_ref = 10¹² kg`. **Verdict: verified** — the gate saturates to exactly 1.0 in
IEEE-754 float64 for `M_ice ≳ 20·M_ref`, is exactly 0 with a nonzero derivative
`Φ/M_ref` at the empty-cap boundary, conserves mass where the originally
specified sigmoid form would leak ≈313 Pa of surface pressure per Mars year
(about half the present atmosphere), and reduces bit-exactly to the hard gate at
both extremes.

## Variables and Units

| Symbol | Meaning | SI Unit | Value used in example |
|--------|---------|---------|----------------------|
| `dM` (= −Φ) | gated cap mass flux (negative = sublimation) | kg s⁻¹ | −4.09778×10⁸ |
| `M_ice` | polar cap CO₂ ice mass (one pole) | kg | 0 … 2.5×10¹⁵ |
| `M_ref` | gate width (`ice_ref_kg`) | kg | 10¹² |
| `Q_in` | absorbed insolation at the pole | W m⁻² | 188.3375 |
| `Q_out` | frost-point thermal emission `ε σ_SB T_frost⁴` | W m⁻² | 26.5510 |
| `A_cap` | polar cap area (fraction 0.01 of 4πR²) | m² | 1.44371×10¹² |
| `L_sub` | CO₂ latent heat of sublimation | J kg⁻¹ | 5.7×10⁵ |
| `ε` | surface emissivity | — | 0.95 |
| `σ_SB` | Stefan–Boltzmann constant | W m⁻² K⁻⁴ | 5.670374419×10⁻⁸ |
| `T_frost` | CO₂ frost point | K | 149.0 |
| `g` | Mars surface gravity | m s⁻² | 3.72076 |
| `A` | Mars surface area 4πR² | m² | 1.4437×10¹⁴ |

## Physical Law / Starting Point

> **Law (energy-limited sublimation)**: the mass flux off a CO₂ cap held at its
> frost point is the net absorbed power divided by the latent heat of
> sublimation, Φ = (Q_in − Q_out)·A_cap / L_sub, and **an empty reservoir can
> supply no flux**: M_ice = 0 ⟹ sublimation must vanish (conservation of mass).
> **Source**: Leighton & Murray (1966), "Behavior of Carbon Dioxide and Other
> Volatiles on Mars", *Science* 153, 136–144 —
> https://www.science.org/doi/10.1126/science.153.3732.136

## Derivation

### Step 1: The hard gate and its dead gradient

The pre-existing gate implements the reservoir constraint exactly:

$$
\mathrm{gate_{hard}}(dM, M_{ice}) =
\begin{cases}
0 & M_{ice} \le 0 \ \wedge\ dM < 0 \\
dM & \text{otherwise.}
\end{cases}
$$

Both branch outputs (`0` and `dM`) are constants with respect to `M_ice`, so

$$
\frac{\partial\, \mathrm{gate_{hard}}}{\partial M_{ice}} = 0
\quad \text{everywhere it is defined,}
$$

and it is undefined (a jump) at the switching surface `M_ice = 0`. A
gradient-based optimizer therefore receives no signal from the cap-depletion
regime — the motivating defect (audit item D1,
[iclr2027-workplan.md](../ideas/iclr2027-workplan.md)).

### Step 2: The smooth gate

Replace the indicator with a C^∞ multiplier on the sublimating branch only:

$$
\mathrm{gate_{smooth}}(dM, M_{ice}) =
\begin{cases}
dM \cdot \tanh\!\left(\dfrac{M_{ice}}{M_{ref}}\right) & dM < 0 \\
dM & dM \ge 0.
\end{cases}
$$

By definition, $\tanh x = \dfrac{e^{x}-e^{-x}}{e^{x}+e^{-x}}$, which is smooth,
strictly increasing, with $\tanh 0 = 0$ and $\tanh x \to 1$ as $x \to \infty$.
Since both integration paths clamp $M_{ice} \ge 0$
(`planet.py:137-150`, `mars.py` fast path), the argument is never negative.

### Step 3: Boundary behaviour (the mass-conservation requirement)

At the empty cap, $M_{ice}=0$:

$$
\mathrm{gate_{smooth}} = dM \cdot \tanh 0 = dM \cdot 0 = 0
= \mathrm{gate_{hard}},
$$

so no mass leaves an empty reservoir — **exactly** matching the hard gate, not
approximately. The derivative there, by the chain rule with
$u = M_{ice}/M_{ref}$:

$$
\frac{\partial}{\partial M_{ice}}\left[dM\,\tanh\frac{M_{ice}}{M_{ref}}\right]
= dM \cdot \operatorname{sech}^2\!\left(\frac{M_{ice}}{M_{ref}}\right)
  \cdot \frac{1}{M_{ref}}
\;\Bigg|_{M_{ice}=0}
= \frac{dM}{M_{ref}} \neq 0 ,
$$

because $\operatorname{sech}^2 0 = 1$. The gate is *flat in value* but *alive in
gradient* at the boundary — precisely the property the optimizer needs.

### Step 4: Saturation for abundant ice (float64)

For large $x$: $1 - \tanh x = \dfrac{2}{e^{2x}+1} < 2e^{-2x}$ (Taylor/asymptotic
bound). IEEE-754 double precision rounds any value above $1 - 2^{-54}$ to
exactly 1.0 (the ulp below 1 is $2^{-53}$; round-to-nearest). Solving

$$
2e^{-2x} < 2^{-54} \iff x > \tfrac{55}{2}\ln 2 \approx 19.06 ,
$$

so for $M_{ice} \gtrsim 20\,M_{ref} = 2\times10^{13}\,\mathrm{kg}$ the gate is
**bit-identical to 1.0** and the smooth-mode physics equals hard-mode physics
exactly. Present-day cap inventories in the model are ~2.5×10¹⁵ kg
($x = 2500$), deep inside saturation.

### Step 5: Why tanh and not the sigmoid of the original spec

The feature spec ([differentiable-framework-features.md](../ideas/differentiable-framework-features.md),
A3) proposed $\sigma(M_{ice}/M_{ref})$ with $\sigma(x) = 1/(1+e^{-x})$. But
$\sigma(0) = \tfrac12$: an **empty** cap would keep sublimating at half rate
indefinitely, creating CO₂ from nothing. Quantitatively (worked example below),
the leak is $\tfrac12 \Phi \approx 2.05\times10^{8}$ kg s⁻¹, i.e. a spurious
pressure source of

$$
\Delta P_{leak} = \frac{\tfrac12 \Phi\, g}{A}\, t_{year}
\approx 313\ \mathrm{Pa\ per\ Mars\ year}
$$

— half of Mars's present 610 Pa atmosphere per year. $\tanh$ has the same
right-tail (indeed $\sigma(2x) = \tfrac{1+\tanh x}{2}$, so both saturate
identically fast) but passes through the origin, eliminating the leak while
keeping the boundary derivative nonzero. `[DISCREPANCY with spec — resolved]`:
the implementation deviates from the spec's σ deliberately; this document is
the record of why.

### Final Expression

$$
\boxed{\;
\dot M_{gated} =
\begin{cases}
\dot M \cdot \tanh\!\left(\dfrac{M_{ice}}{M_{ref}}\right) & \dot M < 0
\quad\text{(sublimation)}\\[4pt]
\dot M & \dot M \ge 0 \quad\text{(condensation, never gated)}
\end{cases}
\;}
$$

**Code mapping** (`package/src/celestials/planets/mars.py`, `_gate_sublimation`):
```python
if self._smooth_gates:
    return torch.where(
        dM < 0.0,
        dM * torch.tanh(ice / self._ICE_REF),
        dM,
    )
return torch.where(
    (ice <= 0.0) & (dM < 0.0),
    torch.zeros_like(dM),
    dM,
)
```
[VERIFIED] — `torch.where` propagates gradients through the selected branch;
the condition `dM < 0` switches on the *flux direction* (a kink in `dM`, not a
dead zone in `M_ice`), and the hard branch is the original expression unchanged.

## Worked Numerical Example

**Inputs** (northern summer, Ls = 90°, the configuration used in
`tests/mars/basic/test_smooth_gates.py`):

| Symbol | Value | Source |
|--------|-------|--------|
| solar flux F | 590 W m⁻² | Mars mean at 1.52 AU |
| albedo α | 0.25 | `Mars()` default |
| axial tilt | 25.19° | `MARS_AXIAL_TILT` |
| M_ice (north) | 5×10¹¹ kg | test value, gate half-active |
| M_ref | 10¹² kg | `ice_ref_kg` default |

**Calculation** (step-by-step):

1. Pole insolation: `Q_in = (1−α)·F·sin(tilt) = 0.75 × 590 × sin 25.19° = 0.75 × 590 × 0.425634 = 188.3375 W m⁻²`.
2. Frost-point emission: `Q_out = ε σ_SB T⁴ = 0.95 × 5.670374419×10⁻⁸ × 149⁴`.
   `149² = 22201; 149⁴ = 22201² = 4.928844×10⁸` → `Q_out = 26.5510 W m⁻²`.
3. Cap area: `A_cap = 0.01 × 4π × (3.3895×10⁶)² = 1.44371×10¹² m²`.
4. Sublimation flux: `Φ = (188.3375 − 26.5510) × 1.44371×10¹² / 5.7×10⁵ = 4.09778×10⁸ kg s⁻¹`, so `dM = −4.09778×10⁸ kg s⁻¹`.
5. Gate: `tanh(5×10¹¹ / 10¹²) = tanh(0.5) = 0.4621171573`.
6. Gated flux: `−4.09778×10⁸ × 0.4621171573 = −1.893654×10⁸ kg s⁻¹`.
7. South cap (dark, Q_in = 0) condenses: `+26.5510 × 1.44371×10¹² / 5.7×10⁵ = +6.724913×10⁷ kg s⁻¹` (ungated in both modes).
8. Net: `−1.893654×10⁸ + 6.724913×10⁷ = −1.221163×10⁸ kg s⁻¹`.

**Result**: `dM_ice/dt = −1.221163×10⁸ kg s⁻¹`
**Code output** (`compute_derivatives`, `smooth_gates=True`): `−1.221163×10⁸ kg s⁻¹`
**Relative error**: `< 10⁻⁶` (agreement to all printed digits)

Hard mode on the same state gives `−4.09778×10⁸ + 6.724913×10⁷ =
−3.425287×10⁸ kg s⁻¹`, matching the code's `−3.425287×10⁸` — the two modes
differ only because `M_ice ≈ M_ref` here, i.e. inside the intended transition
band.

Boundary derivative check (autograd, float64): at `M_ice = 0`,
`∂(gate)/∂M_ice = 1.000000×10⁻¹²` = `1/M_ref` exactly as derived; at
`M_ice = 1.9×10¹³` the gate is `0.99999999999999989` and at `2.5×10¹⁵` it is
`1.0` exactly (saturation as computed in Step 4).

## Theorems, Rules, and Laws Used

### Hyperbolic tangent — definition and asymptotics
- **Statement**: tanh x = (eˣ−e⁻ˣ)/(eˣ+e⁻ˣ); tanh 0 = 0, tanh′x = sech²x, 1−tanh x = 2/(e²ˣ+1).
- **Wikipedia**: https://en.wikipedia.org/wiki/Hyperbolic_functions
- **YouTube**: https://www.youtube.com/watch?v=3d6DsjIBzJ4 (3Blue1Brown — Taylor series, for the asymptotic expansion technique used in Step 4)
- **Textbook**: Abramowitz & Stegun, *Handbook of Mathematical Functions*, §4.5.
- **Applied here**: gives exact zero at the origin, the saturation bound, and the derivative sech²(0)=1.

### Chain rule
- **Statement**: d/dx f(g(x)) = f′(g(x))·g′(x).
- **Wikipedia**: https://en.wikipedia.org/wiki/Chain_rule
- **YouTube**: https://www.youtube.com/watch?v=YG15m2VwSjA (3Blue1Brown — Essence of Calculus ch. 4)
- **Textbook**: Stewart, *Calculus: Early Transcendentals*, 8th ed., §3.4.
- **Applied here**: Step 3 — ∂/∂M_ice of dM·tanh(M_ice/M_ref) = dM·sech²(·)/M_ref.

### Conservation of mass
- **Statement**: mass is neither created nor destroyed in a closed exchange between reservoirs (cap ↔ atmosphere).
- **Wikipedia**: https://en.wikipedia.org/wiki/Conservation_of_mass
- **YouTube**: no suitable single verified lecture found — flagged rather than guessed; see textbook.
- **Textbook**: Jacobson, *Fundamentals of Atmospheric Modeling*, 2nd ed., ch. 3 (continuity equation).
- **Applied here**: Steps 3 & 5 — the gate must vanish at M_ice = 0; σ(0)=½ violates this at a rate of ~313 Pa/Mars-year.

### IEEE-754 double precision rounding
- **Statement**: binary64 has a 53-bit significand; the ulp below 1.0 is 2⁻⁵³, so any real in (1−2⁻⁵⁴, 1] rounds to 1.0 under round-to-nearest-even.
- **Wikipedia**: https://en.wikipedia.org/wiki/Double-precision_floating-point_format
- **YouTube**: https://www.youtube.com/watch?v=PZRI1IfStY0 (Computerphile — Floating Point Numbers)
- **Textbook**: Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic", *ACM Computing Surveys* 23(1), 1991.
- **Applied here**: Step 4 — proves smooth mode is bit-identical to hard mode for M_ice ≳ 20·M_ref, which the agreement tests assert with `torch.equal`.

### Stefan–Boltzmann law
- **Statement**: radiant exitance of a grey body = ε σ T⁴.
- **Wikipedia**: https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law
- **YouTube**: https://www.youtube.com/watch?v=fPKGwvmWmvo (Khan Academy — blackbody radiation; used for Q_out in the worked example)
- **Textbook**: Pierrehumbert, *Principles of Planetary Climate*, §3.3.
- **Applied here**: Q_out = ε σ T_frost⁴ term of the sublimation flux in the worked example.

## Findings and Recommendations

### [VERIFIED] tanh gate — value, derivative, saturation, conservation
- All four required properties hold; worked example matches code to all printed digits.
- Affected lines: `mars.py :: _gate_sublimation` and its two call sites (RK4 and fast paths).

### [APPROXIMATION WARNING] transition-band bias
- For 0 < M_ice ≲ 20·M_ref the smooth flux is deliberately smaller than the
  physical (hard) flux — a bias of up to a factor tanh(x)/1 at x = M_ice/M_ref.
  With the default M_ref = 10¹² kg this band is < 0.001 % of the present cap
  inventory; report M_ref alongside any optimization result and re-verify
  conclusions with the hard gate.

### [DISCREPANCY — resolved] spec said sigmoid, implementation uses tanh
- σ(0) = ½ leaks ≈ 2.05×10⁸ kg s⁻¹ (≈ 313 Pa/Mars-year) from an empty cap;
  tanh(0) = 0 does not. Deviation is intentional and strictly dominant
  (same saturation tail, exact boundary). Recommended: update the A3 feature
  spec text when it is next revised.

### [NOTE] condensation branch switch
- `torch.where(dM < 0, …)` introduces a derivative kink in `dM` at dM = 0
  (gradient jumps between gate and 1). This is a corner, not a dead zone —
  both one-sided derivatives are finite — and it affects only the measure-zero
  moment when a pole crosses sublimation ↔ condensation.

## Cross-references
- [docs/ideas/iclr2027-workplan.md](../ideas/iclr2027-workplan.md) — audit item D1, task T5
- [docs/ideas/differentiable-framework-features.md](../ideas/differentiable-framework-features.md) — feature A3
- [docs/ideas/master-task-list.md](../ideas/master-task-list.md) — Track A-1, task 6

## References
- Leighton, R. B. & Murray, B. C. (1966). *Science* 153, 136–144.
- Abramowitz, M. & Stegun, I. A. (1964). *Handbook of Mathematical Functions*. Dover.
- Goldberg, D. (1991). *ACM Computing Surveys* 23(1), 5–48.
- Pierrehumbert, R. T. (2010). *Principles of Planetary Climate*. Cambridge UP.
- Jakosky, B. M. et al. (2018). MAVEN loss rates (escape-rate constant used in the same RHS). *Icarus* 315, 146–157.
