# Benchmark and Measurement Methodology

Cassandra T1 has released checkpoints and recorded training losses, but it does not yet have a formal public benchmark suite. This distinction matters. Loss curves and sample outputs help guide development, but public model claims require reproducible evaluation assets.

## Current Public Measurements

| Measurement | Status |
|---|---|
| Checkpoint hashes | Published |
| Checkpoint file sizes | Published |
| Epoch-5 loss note | Published as source-project measurement |
| Qualitative epoch comparison | Published in `docs/coherence-comparison-v2-ep2-vs-t1-ep5.md` |
| General checkpoint eval script | Published in `src/eval/evaluate_t1.py` |
| Reproducible benchmark suite | Not included yet |
| Latency benchmark | Not included yet |
| Memory benchmark | Not included yet |

## Current Coherence Comparison

The current qualitative comparison indicates that Cassandra T1 epoch 5 has a broader but still fragmented output profile, while the v2 scratch epoch-2 checkpoint shows stronger repetition collapse around tokens such as `left`, `right`, `The`, `A`, and `Q`.

This comparison supports using epoch 5 as the more useful continuation baseline until v2 scratch is debugged or retrained. It is not a formal benchmark result.

## Current Evaluation Script

`src/eval/evaluate_t1.py` provides a repeatable prompt suite and lightweight coherence heuristics. It writes raw JSONL generations plus a Markdown report with latency, expected-term hits, unique-word ratio, repeated-token signals, and collapse flags.

## Required Benchmark Package

Future benchmark releases should include:

- Checkpoint name and SHA256.
- Tokenizer SHA256.
- Runtime commit hash.
- Hardware profile.
- Prompt set.
- Decoding parameters.
- Scoring scripts.
- Raw model outputs.
- Human or automated scoring rubric.
- Latency and memory measurement scripts.
- Failure analysis.

## Evaluation Categories

The next measurement package should separate:

- Short factual QA.
- Long-form coherence.
- Identity consistency.
- Technical service reasoning.
- Appliance and field-service troubleshooting.
- Code generation and repair.
- Spatial/layout reasoning.
- Safety and refusal behavior.
- Latency per denoising step.
- VRAM/RAM footprint.

## Reporting Standard

Until that package exists, Cassandra T1 should be described as:

> An experimental masked-diffusion language-model prototype with released checkpoints, documented architecture, and preliminary training measurements.

It should not be described as benchmark-leading, production-ready, or validated against mature autoregressive models without reproducible evidence.
