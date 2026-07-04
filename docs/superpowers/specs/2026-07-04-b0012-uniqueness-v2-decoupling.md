# B0012 — Pooled Uniqueness v2: Fit-Weight / Inference Decoupling (design spec)

**Status:** spec (promoted from `backlog/proposed/B0012.json` on 2026-07-04)
**Trigger:** every pooled equity configuration tested 2026-07-03 (dense state, sparse CUSUM,
regime-gated, ungated) converged to effective-N ≈ 650-690 < floor 799 — the mathematical
fingerprint of the ρ=1 cross-asset redundancy assumption in `pooled_avg_uniqueness`. Breadth
buys zero statistical power under the current rule.
**Methodology source:** quant-phd-advisor consult 2026-07-04 (recorded in B0012 history).
Load-bearing citations: AFML §4.3 (concurrency is SAME-price-path only — verified in exact
text; our cross-asset rule is an extension beyond the book), MLAM (denoising), Meucci SSRN
1358533 (Effective Number of Bets).

## 1. The core decision: one number becomes two

The current implementation uses ONE quantity — the ρ=1 pooled uniqueness — for two different
jobs. v2 splits them and they must NEVER be merged again:

| Role | v2 construction | Why |
|---|---|---|
| **Fit-weight** (training influence: base fit, hyperparam search, calibration) | Correlation-discounted concurrency (§2) | Generosity here uses real idiosyncratic signal; it cannot leak and cannot manufacture false discoveries. The ρ=1 rule starves the fit. |
| **Inference effective-N** (pre-flight floor, DSR/MinBTL, any gate) | UNCHANGED ρ=1 conservative count (`pooled_avg_uniqueness`), plus Meucci ENB as a ceiling **diagnostic only** | Generosity here directly manufactures fake statistical power. The gate keeps the never-easier bound. |

**Invariant (the firewall):** no gate, floor, PSR/DSR input, or promote decision may ever
consume the fit-weight sum or the ENB. If this firewall breaks, "PSR/DSR become theater"
(advisor's words). The gate-monotonicity test (§5.4) enforces it empirically.

## 2. Fit-weight v2 — correlation-discounted concurrency

For pooled event `i` on asset `a`, alive at wall-clock instant `t`:

```
c_t^i = n_a(t) + Σ_{b ≠ a} ρ*_{ab}(t) · n_b(t)
u_t^i = 1 / c_t^i          ū_i = mean of u_t^i over i's lifespan
```

where `n_x(t)` = number of open events on asset `x` at `t` (same-asset events share the
actual price path → full AFML §4.3 concurrency, ρ*=1 implicitly via `n_a(t)`), and for
cross-asset pairs:

```
ρ*_{ab}(t) = clip( 0.5 · ρ̂_{ab}(t) + 0.5 · ρ̄(t),  RHO_FLOOR, 1.0 )
```

- `ρ̂_{ab}(t)`: trailing 252-bar Pearson correlation of daily log returns, strictly
  point-in-time (returns up to t−1 only), refreshed every 21 bars (values held constant
  between refreshes — coarse cadence is deliberate stability, not laziness).
- Shrinkage toward the panel mean `ρ̄(t)` (constant-correlation target, Ledoit-Wolf-style)
  with fixed λ=0.5: individual pairwise estimates on 252 obs are noisy; the panel mean is
  the stable component.
- `RHO_FLOOR = 0.15`: guards the advisor's #1 false-edge risk — PIT correlations collapse in
  calm regimes, crediting fake independence exactly when edges are least real. Negative
  estimates clip to the floor too (large-cap equities: negative pairwise ρ at 252d is noise).
- Endpoints sanity: ρ*≡1 recovers the current v1 exactly; ρ*≡0 recovers within-name-only.
  v2 lives strictly between.

**Reduction property (parity guard):** for a single-asset pool the cross terms vanish and
`ū_i` must equal the v1 within-name uniqueness bit-identically.

## 3. Inference side — unchanged, plus persistence (absorbs B0011)

- The pre-flight `pooled_effective_N` (sum of ρ=1 `pooled_avg_uniqueness`) and the refusal
  floor comparison are UNCHANGED regardless of fit-weight mode.
- New: `effective_n.json` persisted at every pooled run's output root:
  `{pool_key, primary, raw_n, effective_n_rho1, fit_weight_mode, fit_weight_sum,
  enb_ceiling, rho_panel_mean_last, computed_at}` — closes the B0011 reproducibility gap
  (these numbers currently exist only as stdout prints).
- `enb_ceiling`: Meucci-style effective number of bets on the LAST refreshed shrunk
  correlation matrix: `exp(entropy(λ_k / Σλ))` over its eigenvalues. Diagnostic ceiling on
  how many independent contemporaneous bets the panel could at most contain (~2-3 expected
  for 35 large-caps at ρ̄≈0.4). Reported, never gated on.

## 4. Wiring — zero changes to `scripts/run_multi_h4.py`

`_run_one_pool` already supports weight injection: with `meta_pooling.pooled_uniqueness:
false` it uses the members' own `m["w"]` (w_peas path) and STILL computes/prints the ρ=1
effective-N pre-flight from `pooled_avg_uniqueness` — exactly the split we need.

- New config key `meta_pooling.fit_weight`: `"rho1_pooled"` (default; today's behavior,
  byte-identical) | `"corr_discounted_v2"` | `"per_asset"` (alias for pooled_uniqueness
  false with untouched member weights).
- In `scripts/run_pooled_equity_d1.py` main(), when `corr_discounted_v2`: build the PIT ρ*
  series from the already-loaded panel closes, compute `ū_i` per member event, overwrite
  `m["w"]`, and call `_run_one_pool` with `pooled_uniqueness=False`. Weight balancing
  (`weight_balance`) applies downstream unchanged.
- `effective_n.json` written for EVERY mode (including default) — persistence is
  unconditional.
- New module functions in `pipeline/sample_weights.py` (additive): `rolling_panel_rho`,
  `corr_discounted_uniqueness`, `effective_number_of_bets`.

## 5. Mandatory validation battery (pre-registered; ALL must pass before any v2 result is read)

1. **Synthetic recovery:** simulate an equicorrelated panel (one market factor, known ρ,
   known idiosyncratic vol), fire synthetic overlapping events; the v2 fit-weight sum must
   recover ≈ N/(1+(N−1)ρ) × (single-series effective-N) within tolerance, and must NOT be
   biased upward (reject the construction if it is).
2. **Single-asset parity:** v2 on a one-asset pool == v1 `avg_uniqueness`-style weights
   bit-identically (cross terms vanish). Also: the H4 FX/metal path is untouched by
   construction (no run_multi_h4 changes) — guarded by the existing H4-parity test files.
3. **Phase-randomization placebo:** destroy true cross-asset dependence (phase-randomize
   returns / shuffle asset identity) preserving marginals → the credited independence
   (fit_weight_sum, ENB) must NOT rise vs the real panel. An estimator that invents breadth
   on noise is rejected.
4. **Gate monotonicity:** the conservative inference effective-N and every gate input are
   IDENTICAL under any `fit_weight` mode (assert in tests); therefore the DSR/floor survivor
   set under v2 is automatically a subset-or-equal of ρ=1 survivors. A "new survivor" can
   only come from the fit actually learning better — visible in OOF metrics, not in the gate.

## 6. What v2 does NOT do

- Does NOT touch the event floor, the refusal cliff, PSR/DSR inputs, or `should_stack`.
- Does NOT use beta/market-residual splits for inference (advisor: sector co-movement makes
  single-factor residuals over-credit independence — the exact false-edge direction).
- Does NOT resolve whether 35 correlated large-caps on D1 can EVER clear the 799 floor. If
  the ENB ceiling reads ~2-3, the honest conclusion is that this universe/timeframe cannot
  demonstrate pooled edge at that floor — a finding to act on (denser bars, wider/less
  correlated universe), not to weight around.

## 7. References

- AFML §4.3-4.6 (`D:\PROJECTS\QuantTradingDocs\extracted\AFML_book.txt`) — same-return
  concurrency definition; harmonic-mean uniqueness.
- MLAM §2 (denoising; constant-correlation shrinkage rationale).
- Meucci (2009), *Managing Diversification*, SSRN 1358533 — Effective Number of Bets.
- B0012 history (`backlog/in_progress/B0012.json`) — full consult record 2026-07-04.
