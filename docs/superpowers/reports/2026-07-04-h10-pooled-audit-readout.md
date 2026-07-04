# h10 pooled audit readout (2026-07-04) — first floor-valid verdict: NO EDGE

**Run:** `results/clf_equity_m3_cusum_cs_v2_h10` — config `configs/equity_m3_cusum_cs_v2_h10.yaml`
(recorded trial: horizon 40→10, tp 3.0→1.5 ATR re-derived per DA tick-3 objection; everything
else identical to the B0012 v2 run). 35 M3 names, cusum_filter, CS features, v2 fit-weights.

## 1. The milestone: pre-flight PASSED for the first time

`PRE-FLIGHT: raw_N=19659 pooled_effective_N=1237.7 refusal_floor≈799 — OK (relieves starvation)`.

Every previous pooled verdict (654 < 799) was formally unreadable — "not enough independent
observations to measure." This one is the project's **first statistically valid pooled verdict**.
Horizon curve for the record: h40→654, h20→716, h10 (tp1.5)→1237.7.

## 2. The verdict: no measurable edge, now with statistical standing

- Panel mean ROC-AUC: **0.531** (vs 0.540 at h40 — marginally worse, both ≈ coin-flip).
- Meta at threshold 0.55 keeps **145 trades of 19,659 events (0.7%)** across all 35 names;
  **zero (asset, fold) cells reach the 30-trade Sharpe floor** → every sharpe_net is NaN by
  invariant (no measurement, not zero skill).
- Stack decisions: **0/35**. Two names produced measurable PSR/DSR for the first time
  (GE PSR 0.75/DSR 0.56; KO PSR 0.61/DSR 0.54) — both well below promote thresholds.
- Long/short split (n≤16 per cell — anecdote, not evidence): shorts positive
  (+7.8%/trade, hit 1.0 on 12 trades), longs negative. NOTE: lgbm and xgb report byte-identical
  short cells (same 12 trades attributed to both models) — treat as one observation, not two.
  With a survivor universe the short side is deflated by construction, which makes the
  positive-shorts anomaly worth ONE followup look, not a claim.

## 3. Honest interpretation

The B0012→B0014 arc closed the statistical-power problem (fit mass 2.6×, floor cleared with
margin) and the answer didn't change: **cusum_filter events carry no signal the meta can
amplify on this panel**, at h40 and at h10. The binding constraint is now unambiguously the
primary, not the plumbing. That is exactly what Loop A exists to attack — and its pre-flight
gate is no longer a guaranteed refusal, so proposals can now be audited to completion.

## 4. Disposition

- Trial recorded in B0014 history (h10 config = trial for DSR counting).
- Next lever: Loop A ticks against the now-passable audit path (the tick-3 BLOCK's
  reachability objection is resolved by this geometry); primaries beyond cusum/momentum.
- The one followup permitted by this data: check whether the 12-16 short trades concentrate
  in a single fold/episode before anyone mentions "short-side edge" again.
