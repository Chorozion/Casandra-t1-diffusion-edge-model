# V1-V6 LoRA — pre-registered lattice-channel remediation

This subdirectory holds the LoRA-on-triple-attention experiment that
tests whether the empirically content-free LTMi lattice channel can be
rescued by any of five targeted interventions.

**Companion docs:**
- [`docs/PRE_REGISTRATION_v1_v6.md`](../../docs/PRE_REGISTRATION_v1_v6.md) — decision rules committed BEFORE training
- [`eval/lattice_ablation/`](../lattice_ablation/) — the random-coord ablation that motivated this experiment
- [`docs/ltmi-xt-rescue.md`](../../docs/ltmi-xt-rescue.md) (parent dir) — earlier rescue attempts

## The five interventions

| Variant | Intervention |
|---|---|
| **V1** | LoRA-only baseline (no lattice mechanism change) |
| **V2** | V1 + learned MLP projection on lattice coords → path-3 K contribution |
| **V3** | V1 + multi-resolution lattice (coarse 4³ + medium 16³ + fine 64³) |
| **V4** | V1 + softmax temperature annealing on path-mix gate (τ=5 → τ=1) |
| **V5** | V1 + aux contrastive loss forcing lattice-cond logits to differ from un-conditioned at locked positions |
| **V6** | V1 + V2 + V3 + V4 + V5 stacked |

## Files

| File | Purpose |
|---|---|
| `triple_attention_lora.py` | `TripleAttentionLoRA` wrapper class + LoRA/Lattice intervention modules |
| `train_triple_lora.py` | Train a single (variant, seed) combination |
| `train_all_lora_variants.py` | Sequential driver for all 6×3 = 18 runs |
| `eval_with_lora.py` | Load adapter, run forced-anchor eval on n=172 corpus |
| `eval_baseline_n172.py` | Re-eval v2_ltmi_triple on the expanded corpus (baseline arm) |
| `bootstrap_lora_variants.py` | Apply pre-registered decision rules to all 18 evals |
| `run_post_training_pipeline.py` | Orchestrator: waits for training → baseline → 18 evals → bootstrap |
| `expand_eval_corpus_with_mercury2.py` | n=36 → n=172 corpus expansion via Inception API (env var: `INCEPTION_API_KEY`) |
| `ship_smoke_v1_5.py` | Sanity check that the v1.5 base loads + runs |

## Reproducing the experiment

```bash
# 1. Get weights (cassandra_t2_ltmi-triple_step500.pt) — distributed
#    separately, see release notes
#
# 2. (Optional) regenerate the expanded n=172 corpus
export INCEPTION_API_KEY=sk_...
python eval/v6_lora/expand_eval_corpus_with_mercury2.py

# 3. Train all 18 LoRAs (6 variants × 3 seeds)
python eval/v6_lora/train_all_lora_variants.py

# 4. Eval baseline on n=172
python eval/v6_lora/eval_baseline_n172.py

# 5. Eval each adapter
python eval/v6_lora/run_post_training_pipeline.py

# 6. Apply pre-registered verdict
python eval/v6_lora/bootstrap_lora_variants.py
# → produces LORA_VERDICT.md with PASS/FAIL/AMBIGUOUS per variant
```

## Pre-registered decision rules (excerpt)

> **PASS** = ≥2 of 4 metrics CI-significant improvement vs baseline,
> with magnitude ≥+0.02, sign consistent across 3 seeds, mean ≥ baseline mean.
>
> **FAIL** = at most 1 metric passes, OR any metric shows CI-significant
> degradation vs baseline.
>
> **AMBIGUOUS** = sign flips across seeds with large magnitude — treat as FAIL.
>
> No metric-hopping. No post-hoc.

Full rules in [`docs/PRE_REGISTRATION_v1_v6.md`](../../docs/PRE_REGISTRATION_v1_v6.md).

## Why this exists

Earlier in 2026-05 we shipped Cassandra T2 with `attention_use_ltmi_priors=True`
and the v0.1 LTMi-XT spec claimed the lattice coord was a "model-conditioning
signal." The 2026-05-12 three-way ablation (BLAKE2b / PCA-3D / uniform-random
per-locus) showed all three produced statistically indistinguishable
downstream behavior — unforced inference was byte-identical between random
and BLAKE2b arms.

The lattice channel, as wired in v0.1, doesn't get gradient flow strong enough
to do anything useful at our scale. This experiment tests whether any of
five targeted fixes restore the channel's utility — OR formally retires
the `attention_use_ltmi_priors` interface from the spec.

Either outcome ships honestly. Pre-registration means the verdict can't
drift to match the data.
