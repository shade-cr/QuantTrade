# Pooled audit (T002D1) + B0006 cross-sectional-pack validation — honest readout

**Date:** 2026-07-03
**Analyst role:** quant-validation (pre-registered criteria; numbers quoted from artifacts on disk)
**Scope:** three result sets — (1) T002D1 pooled re-audit, (2) B0006 run (a) state primaries + CS pack, (3) B0006 run (b) sparse cusum primary + CS pack — plus the no-CS M3 baseline.
**Universe caveat (B0003 caveat 1):** survivor-selected 35 t0-2006 large-caps; long side mechanically inflated, short side deflated. Only *within-universe relative* claims are admissible.

> **Headline verdicts.**
> **T002D1 — NOT CONFIRMED** (failure mode: breadth collapse → single-name/single-fold artifact; compounded by dual-threshold fragility and STARVED effective-N).
> **B0006 CS pack — FAILS the pre-registered falsification** (criteria 1 and 2 both fire; 3 and 4 not-cleanly-computable this round but consistent with collapse). The CS pack adds no measurable value over the own-bar+macro baseline; §5 protocol prediction (larger uplift under the sparse event primary) is **falsified in direction**.

---

## Part A — T002D1 pooled re-audit verdict

**Proposal:** `20260703-ABT-D1-BULL_QUI-T002D1`, cusum_filter(2.0) primary, BULL_QUIET regime gate, pooled over the 35-name universe. DA verdict was PROCEED_WITH_CAVEAT (four binding caveats). Audit status `completed_pending_human_read` — **no auto-promotion**; this section is the human read.

### A.1 Dual-threshold framing (state both explicitly)

The pooled_v1 grading counts trades at the **audit effective threshold 0.3643** (`ev_breakeven_v1` for tp2.5/sl1.0, C_ATR 0.1, LAMBDA_MARGIN 0.05). The `long_short_split.json` breakout is at the **fixed 0.55 headline**. These are two different operating points and they tell opposite stories:

| Operating point | Trades | Central-tendency Sharpe | Reachability |
|---|---|---|---|
| EV-breakeven **0.3643** (grading) | **515** total across 35 names | median active-fold Sharpe **0.663** | passes numeric bar |
| Fixed **0.55** headline (long_short) | rf 5 long / 13 short; lgbm/lr/xgb **0** | all NaN (n<30) | nearly empty |

### A.2 What the pre-registered criterion says vs what the breadth discipline says

The pre-registered falsification package (STABLE/MARGINAL_2FOLDS, median active-fold Sharpe ≥ 0.5, ≥ 50 trades) is **numerically met on paper**: 0.663 ≥ 0.5 and 515 ≥ 50, so `criterion_eval` records `passed: true` on both. **This is a false positive by the project's own skepticism standard, and here is why:**

- **Breadth = 1/35.** Exactly one asset (**GE**) clears the ≥30-trade + positive-aggregate gate. The reported `median_active_fold_sharpe` of **0.663 is literally GE's number** — with only one active asset, the "median over active folds" degenerates to a single name. The other 34 assets have 3–29 trades each and `aggregate_sharpe: null`.
- **GE's 0.663 is one fold of one model.** GE's `lr` fires [0, 21, 31, 6] trades across folds 0–3; only fold 2 (31 trades) clears the 30-trade gate, and its Sharpe (0.6627) **is** the aggregate. The other three GE models are **anti-discriminative**: OOF ROC-AUC xgb 0.403, lgbm 0.358, rf 0.433 — all *below* 0.5. GE's own DSR values are collapsed to ~0 (xgb 4e-160, rf 1.5e-147, lr 0.0).
- **The 0.55 headline nearly empties and all-loses.** At the fixed headline, only `cusum_filter/rf` fires at all: 5 long (mean pnl −0.136, hit 0.0) and 13 short (mean pnl −0.073, hit 0.0) — **every one a loser**. lgbm, lr, xgb fire 0 long and 0 short. A candidate whose only tradeable pool at the headline threshold is 18 all-losing trades is not an edge.
- **Stack gate never clears.** GE's `stack_decision` is `stack=false` ("competence: only 0/4 models beat baseline in ≥3/4 folds with ≥30 trades each").

### A.3 Binding DA caveats applied

1. **Long/short breakout + B0003 discount (medium).** Applied: the 0.55 breakout shows the survivor long-side is not even tradeable here (5 long, all losers) and the short side is 13 all-losers. There is no long-vs-short asymmetry to discount because there is no surviving trade population on either side at the headline. Any positive signal lives only at the looser 0.364 EV threshold, still carrying full long-side inflation.
2. **cs_spread_21 MDA fold-stability (low).** See Part B.1 — CS-cluster MDA (including the cluster carrying `cs_spread_21`) is in the noise band and cannot be counted as mechanism confirmation.
3. **vix_chg_5 = regime-recognition (low).** Down-weighted per the caveat; not load-bearing given no surviving edge.
4. **68.1%-gate in-vs-out-of-regime (medium).** A gate retaining 68% of bars is barely a gate; the caveat requires an in-regime vs out-of-regime comparison at promote time. Moot this round — there is nothing to promote.

### A.4 T002D1 verdict

**NOT CONFIRMED.** Failure mode: **breadth collapse → single-name, single-model, single-fold artifact.** The nominal pass (0.663 ≥ 0.5; 515 ≥ 50) is manufactured by pooling many near-empty names at the loose 0.364 EV threshold; the entire central-tendency number is one fold (31 trades) of one model (lr) on one name (GE) whose other three models are anti-discriminative and whose DSR is ~0. At the fixed 0.55 headline the strategy is 18 all-losing trades. The hypothesis is **not falsified as a mechanism** (it remains under-tested), but it is **not confirmed** and must not promote. It is additionally **STARVED**: the pre-flight pooled effective-N (~650–690, see Part C) sits below the 799 floor, so even the pooled route did not deliver the independent information the single-name audits lacked.

---

## Part B — B0006 cross-sectional pack: the four pre-registered criteria

Spec §6: the pack **FAILS if ANY** criterion holds. Data: run (a) `clf_equity_m3_d1_cs` (35×{ema_cross, momentum_zscore}) vs no-CS baseline `clf_equity_m3_d1`; run (b) `clf_equity_m3_cusum_cs` (35×cusum_filter).

### B.1 Criterion 1 — "No cluster signal" → **FIRES (fails)**

Clustered MDA (F1-scored, from `clustered_mda.json`, aggregated over every model×fold×cluster). ONC re-clusters per fold, so cluster membership is not a stable object across folds and the pre-registered *per-cluster bootstrap CI over folds* is **not cleanly derivable**; I applied the spirit — classify each cluster instance as **pure-CS** (all members `cs_*`), **mixed-CS** (some `cs_*`), or **non-CS**, and compare the pure-CS bucket against the non-CS noise floor.

| Bucket (run a, state) | n | mean MDA | median | frac > 0 |
|---|---|---|---|---|
| pure-CS | 1680 | **+1.66e-4** | +9e-6 | 0.583 |
| mixed-CS | 3360 | +1.74e-3 | +3.3e-4 | 0.698 |
| non-CS (noise floor) | 9240 | +7.4e-5 | 0.000 | 0.405 |

| Bucket (run b, cusum) | n | mean MDA | median | frac > 0 |
|---|---|---|---|---|
| pure-CS | 700 | **+6.0e-5** | +5e-6 | 0.650 |
| mixed-CS | 1540 | +2.36e-3 | +1.1e-3 | 0.727 |
| non-CS | 5040 | +8.9e-5 | 0.000 | 0.368 |

The **pure-CS clustered-MDA sits in the same ~1e-4 magnitude band as the non-CS noise floor** (1.66e-4 vs 7.4e-5 for state; 6.0e-5 vs 8.9e-5 for cusum) — i.e. dropping the CS clusters barely moves F1. The dominant real driver is the technical/primary cluster (`primary_side|macd|r_*|rsi|z_r20…`) at **0.007–0.032**, two to three orders of magnitude larger. The mixed-CS bucket's larger mean is contaminated: ONC merges one or two CS features into that dominant technical cluster (substitution), so its importance is carried by the technical/primary features, not by the CS features — it cannot be attributed to the pack. The pure-CS sign is weakly positive (frac_pos 0.58–0.65), but (a) the effect size is negligible vs the real drivers and (b) the fold/asset units are cross-sectionally correlated (market-wide bursts), so the apparent >0 tilt is not trustworthy as independent evidence. **Verdict: indistinguishable from zero → criterion 1 fires.**

> **Footnote.** Individual (non-clustered) MDA shows `cs_basket_beta_rank` (75.0% positive-sign folds), `cs_breadth_200` (78.1%), `cs_idio_vol_rank` (71.9%) — sign-consistency above chance, though magnitudes stay ~1e-4, two orders of magnitude below the technical cluster. Verdict unchanged, but this is the concrete starting point if the CS pack is ever revisited outside ONC clustering.

### B.2 Criterion 2 — "No skill uplift" → **FIRES (fails)**

`metrics_per_fold.json` carries no explicit F1; the available NaN-safe classification metrics are ROC-AUC, PR-AUC, precision@recall{0.3,0.5}, MCC (named explicitly). nanmedian over all asset×model×fold cells:

| metric | run (a) CS+state | baseline (no CS) | Δ (a − baseline) | ~1 SE |
|---|---|---|---|---|
| ROC-AUC | **0.5445** | **0.5452** | **−0.0007** | 0.0027 |
| PR-AUC | 0.2802 | 0.2818 | −0.0016 | — |
| precision@recall0.3 | 0.2774 | 0.2772 | +0.0002 | — |
| precision@recall0.5 | 0.2781 | 0.2785 | −0.0004 | — |
| MCC | 0.000 | 0.000 | 0.000 | — |
| OOF ROC-AUC (per-model) | 0.5312 | 0.5297 | +0.0015 | — |

Every delta is at or below one fold-to-fold standard error (ROC-AUC cell SE ≈ 0.0027), and the two best-powered metrics (ROC-AUC, PR-AUC) move slightly **negative**. The CS pack does **not** exceed the baseline. Corroborating: **stack passes 0/70** for run (a), identical to the baseline's **0/70** — the CS pack changed no stack decision. **Criterion 2 fires.**

### B.3 Criterion 3 — "Survivorship artifact" → **not computable this round**

The falsification test requires re-running with features recomputed sector-neutralized / rank-only; no such variant exists on disk this round. Recorded honestly as **not-computable-this-round**. Mitigating design note: the v1 pack features are **already CS percentile-ranks by construction** (spec §2), and #1/#3 are market-residualized (§3), so the "riding absolute winners" mechanism is partially pre-empted by design — but a formal sector-neutral re-run was not executed and is the clean test if the pack is ever revisited.

### B.4 Criterion 4 — "DSR collapse" → **not cleanly computable; consistent with collapse**

PSR/DSR are NaN across almost all cells: **run (a) 10/280** cells non-NaN, **baseline 0/280**, **run (b) 0/140** (`n_trials_familywise = 112` in every file). The 10 computable run-(a) DSR values are effectively zero (e.g. PEP ema_cross/rf DSR 3.7e-119; JNJ/ORCL xgb ~3e-5 on 2 trades). No pool has a DSR that survives the 112-trial familywise deflation. Because criterion 2 already establishes there is no Sharpe *uplift* to deflate, criterion 4 is moot in practice; recorded as **not-cleanly-computable, consistent with DSR collapse**.

### B.5 B0006 verdict

**The pack FAILS the pre-registered falsification** — criteria 1 (no cluster signal) and 2 (no skill uplift) both fire; passing requires the opposite of *all four*. The CS v1 pack does not add relative-value information the meta can use on this universe/geometry.

---

## Part C — Protocol comparison (§5): run (a) state vs run (b) sparse-event, and the effective-N synthesis

### C.1 Is the CS uplift LARGER under the sparse event primary? — **No (falsified in direction)**

Spec §5 predicted CS features "pay off most with idiosyncratically-timed events (higher uniqueness)", so run (b) should show a larger CS-cluster uplift than run (a). The evidence points the other way:

- **Pure-CS clustered MDA is *smaller* under cusum**, not larger: run (a) mean +1.66e-4 vs run (b) +6.0e-5.
- **OOF ROC-AUC is *lower* under cusum**: 0.5219 (b) vs 0.5312 (a); baseline 0.5297.
- **Reachability is worse**: zero-trade model-folds 92.5% (a) → **98.8% (b)**; stack passes 0/35 (b).

Caveat (stated for honesty): there is **no cusum-without-CS baseline on disk**, so run (b)'s CS *uplift* is not strictly measurable — only the *absolute* CS-cluster signal, which is not larger. run (b)'s higher PR-AUC (0.324 vs 0.282) is a base-rate artifact of the sparser primary (fewer, higher-density events), **not** a CS effect. The §5 directional hypothesis is **not supported**.

### C.2 Effective-N convergence — the load-bearing cross-cutting finding

Every one of today's pooling configurations lands at ~650–690 effective bets despite wildly different raw event counts and primary types:

| Configuration | Raw pooled events | Pre-flight `pooled_effective_N` | vs floor 799 |
|---|---|---|---|
| Dense state, ema_cross (baseline) | 143,012 | **688.6** | below |
| Dense state, momentum_zscore (baseline) | 121,050 | **647.4** | below |
| Regime-gated sparse (T002D1 pooled audit) | 11,901 | ~650–690 (pre-flight) | below |
| **Ungated sparse cusum(2.0)** | **19,659** | **654** (~30× concurrency) | below |

The hypothesis motivating the sparse-event route — "asynchronous idiosyncratic events scale effective-N" — is **empirically falsified**. CUSUM bursts **synchronize across correlated names** (market-wide information arrival), so a ~30× raw-to-effective compression appears even for the ungated sparse primary, landing at the *same* ~650 as the dense state primaries. Pooling 35 names did not relieve the sample-starvation constraint under any primary tried.

### C.3 Open methodology question (for quant-phd-advisor — flagged, NOT resolved here)

The binding constraint is **the cross-asset wall-clock uniqueness methodology itself.** `pooled_avg_uniqueness` treats simultaneous events on *different* names as fully redundant (concurrency counted across the whole panel). Whether **full cross-asset redundancy is the correct generalization of AFML §4** — versus a partial-redundancy weighting keyed on realized cross-name return correlation, which would credit correlated-but-not-identical names with some independent information — is a load-bearing, unresolved question. If the answer is "partial", the 799 floor may be over-conservative for pooled panels and several of today's "below floor" verdicts would move. **Recorded as the load-bearing open question; deliberately not resolved in this readout.**

---

## Part D — False-positive checklist

| Check | Finding |
|---|---|
| **Single-class folds** | None. OOF base rate ≈ 0.25 across all runs (independent verification: mean/median of unique per-asset-primary base rates ≈ 0.25; 0 single-class cells in a/baseline/b). Failure mode is starvation, not label collapse. |
| **Zero-trade models** | Pervasive. Zero-trade model-folds: **92.5%** (run a), **95.3%** (baseline), **98.8%** (run b). At the 0.55 headline, xgb/lgbm/lr fire ~0 trades pool-wide in every run. |
| **Calibration keep-rate collapse** | Confirmed via the zero-trade rate above (sigmoid-calibrated 0.55 is effectively unreachable at the ~0.28 win base rate). The CS pack marginally *reduces* collapse (92.5% vs 95.3% zero-trade) but not enough to matter; cusum is worst (98.8%). |
| **One lucky name/fold (nanmedian discipline)** | The decisive check. **T002D1's 0.663 is one name (GE), one model (lr), one fold (31 trades)** — the nanmedian over active folds degenerates to a single cell because breadth = 1/35. For B0006, nanmedian ROC-AUC ≈ 0.545 with no CS uplift — no isolated cell rescues the pack. |
| **OOF discrimination** | ROC-AUC nanmedian ≈ 0.52–0.55 across all runs; at ~0.5 no threshold manufactures an edge. GE's 3 non-lr models are *below* 0.5 (anti-discriminative). |
| **PSR/DSR** | Non-NaN in only 10/280 (run a), 0/280 (baseline), 0/140 (run b) cells; where computable, DSR ≈ 0 after 112-trial deflation. No promotable candidate. |

---

## Summary

- **T002D1 → NOT CONFIRMED** (breadth-collapse single-name/single-fold artifact + dual-threshold fragility + STARVED effective-N). Do not promote; hypothesis remains under-tested, not falsified.
- **B0006 CS pack → FAILS** pre-registered falsification (criteria 1 and 2 fire; 3 and 4 not-cleanly-computable, consistent with collapse). No relative-value uplift on this universe.
- **§5 protocol prediction → falsified in direction** (CS signal not larger under the sparse event primary).
- **Cross-cutting → effective-N converges to ~650–690 across all configs** regardless of primary sparsity; the cross-asset wall-clock uniqueness methodology is the binding constraint and the load-bearing open question for the quant-phd-advisor.

All next steps (whether B0006 closes as no-edge or spawns a partial-redundancy follow-up; whether B0010's pooled route is judged sufficient) are **human decisions** — this readout auto-promotes nothing.
