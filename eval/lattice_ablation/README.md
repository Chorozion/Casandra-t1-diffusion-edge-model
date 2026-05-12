# Lattice-channel ablation — three-way coord-scheme comparison

The 2026-05-12 experiment that walked back the v0.1 LTMi-XT spec's
claim that the lattice coordinate is a "model-conditioning signal."

## Setup

Same Cassandra T2 architecture, same training data (C1/C2/C3 bundles,
144 (Q, A, locus) triples), same warm-start from v1.5, same optimizer,
same 500-step continued-pretrain schedule. **Only the per-locus lattice
coord assignment differs across arms:**

| Arm | Coord scheme |
|---|---|
| `v2_ltmi_triple` (BLAKE2b) | LTMi-XT v0.1 §2.4 default — BLAKE2b digest of breadcrumb prefix |
| `v2_5_pca` | PCA-3D of frozen-encoder embeddings, projected onto top 3 PCs, quantized to 64³ |
| `v2_5_random` | uniform-random per-locus from {0..63}³, deterministic seed=0xC0FFEE |

## Result (paired bootstrap, n=36, n_boot=2000)

**v2_5_pca − v2_ltmi_triple:** 0/4 metrics CI-significantly differ.

**v2_5_random − v2_ltmi_triple:**
- forced corpus_overlap: −0.016, CI [−0.054, +0.022], p=0.20 — not CI-sig
- forced english_ratio: −0.013, CI [−0.044, +0.016], p=0.20 — not CI-sig
- **unforced corpus_overlap: exactly 0, CI [0.000, 0.000]** — byte-identical
- **unforced english_ratio: exactly 0, CI [0.000, 0.000]** — byte-identical

All three training trajectories landed at byte-identical final loss
(`recent_avg=0.2834`). The unforced byte-identicality is the load-bearing
finding: the lattice channel didn't affect training gradients enough to
change the trained weights between arms.

**Conclusion:** the lattice channel as currently consumed by triple-attention
path 3 in our reference implementation is empirically content-free at our
scale. Only the existence of a deterministic per-locus mapping matters —
not the specific mapping.

## Files

| File | Purpose |
|---|---|
| `precompute_random_coords.py` | Generate `lattice_coords_random.json` (uniform random, deterministic seed) |
| `continued_pretrain_t2_5_random.py` | Train T2 variant using random coords (monkey-patches `lattice_for_breadcrumb`) |
| `run_t2_5_random_eval.py` | Eval the random-coord arm against v1.5 + v2_ltmi_triple BLAKE2b |
| `bootstrap_t2_5_random_vs_t2.py` | Paired bootstrap; verdict on lattice-mapping-specificity |
| `lattice_coords_random.json` | Pre-computed coord table (84 loci) used by training + eval |

## What this implies for v0.4

Either:
- **(a)** redesign the lattice integration to give it a real gradient path
  (e.g. attention bias rather than additive K feature — D1 in the architecture
  audit; see `eval/v6_lora/` for the pre-registered V1-V6 test of this), OR
- **(b)** formally retire the `attention_use_ltmi_priors` interface from
  the spec and treat the `lattice` field strictly as a deterministic per-locus
  identifier for retrieval-side spatial clustering.

The V1-V6 experiment in `eval/v6_lora/` is the formal test of (a). If all
six variants fail their pre-registered PASS criteria, the spec moves to (b).

## Companion findings

- Triple-attention itself IS CI-significant — +0.0475 forced corpus_overlap
  vs single-attention v1.5 baseline (n=36). The architecture wins; the
  lattice channel within it does not.
- Anchor-token-masked LoRA (P1, n=87) shows +0.139 nats OOD generalization
  advantage — unrelated to this ablation, but the validated mechanism.

Public reference cards on the lab feed:
- `1778600451549-t5ja` — PCA-3D null result
- `1778604866975-ugrn` — Mercury 2 adversarial review of the audit
- `1778605934535-8oo6` — Random-coord ablation conclusive
- `1778609096104-1hjr` — V1-V6 remediation pre-registration
