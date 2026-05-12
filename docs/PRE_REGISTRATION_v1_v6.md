# Pre-Registration — Triple-Attention LoRA Variants V1-V6

**Date:** 2026-05-12
**Author:** Thomas Kregon (Chorozion), SOPHIA XT Lab
**Status:** PRE-RESULTS. Decision rules committed BEFORE training.

This document fixes the analysis plan in advance so the conclusions can't
drift to match the data. If the rules below are followed verbatim, the
verdict is determined by the data, not by post-hoc storytelling.

## Goal

After today's null result on the LTMi lattice channel (BLAKE2b / PCA-3D /
random per-locus all empirically equivalent), test five targeted
interventions to determine if any configuration makes the lattice channel
actually carry semantic information that the model uses.

Five interventions, layered cumulatively in V6:
- V1 — LoRA-only baseline (no lattice intervention)
- V2 — V1 + learned MLP projection on lattice coords (path 3 K contribution)
- V3 — V1 + multi-resolution lattice (coarse 4³ + medium 16³ + fine 64³)
- V4 — V1 + softmax temperature annealing on path-mix gate (τ=5 → τ=1)
- V5 — V1 + aux contrastive loss forcing lattice-cond logits to differ
- V6 — V1 + V2 + V3 + V4 + V5 (stack all four interventions)

## Setup

- **Base checkpoint:** `v2_ltmi_triple` (cassandra_t2_ltmi-triple_step500.pt)
- **Training data:** C1/C2/C3 bundles, 144 (Q, A, lattice) triples (identical to T2 training)
- **Steps:** 300 per variant (LoRA needs fewer steps than full fine-tune)
- **LoRA rank:** 16, alpha 32, applied to Q/K/V/O of paths 1 + 3 (content + anchor)
- **Trainable params:** ~12M (V1) to ~23M (V6); base 1.9B frozen
- **Seeds:** 3 per variant (42, 1337, 0xC0FFEE)
- **Total LoRA runs:** 18 (6 variants × 3 seeds)

## Eval corpus

- **Source:** `queries_heldout_extended_v2.json` (Mercury-2 generated, leak-filtered)
- **n=172 paired (corpus, query) pairs** across C5/C6/C7 (56/56/60)
- **Improvement over prior:** n=36 → n=172, a 4.8× expansion. Mercury 2's
  n≥200 target not quite hit but well within 1σ of the recommendation.

Both `forced_anchor=True` and `forced_anchor=False` runs per query, so
final evaluation set is 172 × 2 = 344 (query, force-mode) outcomes per
variant per seed.

## Metrics (frozen)

For each (variant, seed, corpus, query, force_mode):
- `corpus_overlap`: word-overlap with corpus vocabulary, in [0, 1]
- `english_ratio`: fraction of generated tokens that are English words
- `anchor_preservation`: deterministic, expected 1.0 in forced mode

Aggregation:
- Per-arm mean across all 172 queries × force_mode
- 95% bootstrap CI via paired-by-query bootstrap (n_boot=2000) AGAINST
  the reference arm `v2_ltmi_triple_BLAKE2b_baseline` (the existing T2
  checkpoint, evaluated identically)
- Across-seed averaging: report point + CI for each variant's
  seed-averaged mean (3 seeds)

## Decision rules (committed)

For each variant V_k (k = 1..6), the verdict against the reference
v2_ltmi_triple BLAKE2b baseline is determined as follows.

**PASS** = at least 2 of the following 4 metrics show:
1. CI excludes zero at α=0.05 in the direction of improvement
2. Magnitude ≥+0.02 in absolute terms
3. Consistent sign across all 3 seeds (no seed in opposite direction)
4. Mean across seeds ≥ baseline mean

Metrics: forced/corpus_overlap, forced/english_ratio, unforced/corpus_overlap, unforced/english_ratio.

**FAIL** = at most 1 metric meets the PASS criteria, OR any metric shows
CI-significant degradation vs baseline.

**AMBIGUOUS** = mixed across seeds (i.e., the across-seed CI crosses zero
on every metric, but individual seeds show large effects with inconsistent
sign). Treat as FAIL — we don't ship ambiguous.

**No partial credit for variants that fail.** No post-hoc metric search
beyond the 4 above.

## Acceptance for ship

A variant ships to the LTMi-XT v0.4 spec / Cassandra spec if and only if:
- It PASSES per the rule above, AND
- The PASS holds when retested in a fresh eval session (cross-session
  reproducibility check; ~+0.02 noise floor per today's finding)
- Adversarial review by Mercury 2 doesn't surface a load-bearing flaw

If multiple variants PASS, prefer the simplest (lowest variant number)
unless a more complex variant has CI-significantly larger effect.

## Headline negative conclusion (if all fail)

If all 6 variants fail, the formal conclusion is:

> The LTMi lattice channel, as defined in v0.3.1 (path-3-K-additive
> embedding with scalar gate), cannot be made to carry semantic
> information at our scale (sub-1B params, 144 training triples × 500
> base steps + 300 LoRA steps, 28-layer attention) by any of:
> learned MLP projection, multi-resolution embedding, gate temperature
> annealing, or aux contrastive loss.
>
> Next step: redesign the lattice integration entirely (e.g., as
> attention bias per direction D1, NOT as additive K feature), OR
> retire the channel from the spec.

## What this pre-registration does

It STOPS the walk-back cycle. The decision rules above are committed
before seeing any result. When training completes:
- We compute the metrics
- We apply the rules above mechanically
- We report PASS / FAIL / AMBIGUOUS per variant
- We ship or don't ship per the acceptance criteria
- We don't reach for a different metric or seed selection if the
  pre-registered analysis doesn't give the result we wanted

## What this pre-registration doesn't promise

- A positive finding. Most likely outcome (per today's null) is that
  all 6 variants fail. That's a clean negative result and we ship the
  honest "redesign needed" conclusion.
- That the chosen metrics are the right ones. corpus_overlap and
  english_ratio are our standard battery; if the lattice helps a
  metric outside this set, this pre-registration won't catch it. But
  metric-hopping IS the walk-back trap — we commit to these four.

## Compute budget

- Training: 18 LoRAs × ~9 min each = ~2.7 hours
- Evaluation: 18 LoRAs × 344 queries × ~10s each = ~17 hours wall
  (pipelined overnight)
- Bootstrap: ~1 hour all-arms
- **Total: ~21 hours.** All achievable in one overnight + half-day push.
