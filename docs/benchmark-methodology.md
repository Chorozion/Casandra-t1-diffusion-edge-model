# Benchmark Methodology

Cassandra T1 benchmark claims must be reproducible before they are presented as public release numbers. Internal comparison boards can be useful, but they should remain clearly labeled until public evaluation assets are available.

## Answer First: How Should Cassandra Be Benchmarked?

Cassandra should be benchmarked against autoregressive references using fixed prompts, deterministic scoring rules, latency measurements, output-quality rubrics, and published model/runtime settings. Internal results should be labeled as preliminary until the prompts, scoring scripts, and model hashes are released.

## Current Internal Benchmark Board

The SophiaXT website currently uses the following internal benchmark framing:

| Task | Cassandra T1 | Gemma-style AR Reference | Notes |
| --- | ---: | ---: | --- |
| Instruction following | 94 | 96 | ~98% reference envelope |
| Reasoning chain stability | 91 | 93 | ~98% retention margin |
| Code repair prompts | 88 | 90 | ~97.8% score envelope |
| Document QA | 93 | 95 | ~97.9% score envelope |
| Spatial-token tasks | 96 | 91 | Diffusion advantage target |

These numbers should be treated as internal benchmark targets or preliminary internal results unless accompanied by a public evaluation release.

## Required Public Release Artifacts

- Model version and hash
- Runtime version and hardware profile
- Quantization configuration
- Prompt set
- Dataset source or dataset generation procedure
- Scoring script
- Error analysis
- Latency measurements
- Token budget and decoding settings
- Autoregressive reference model details

## Evaluation Categories

### Instruction Following

Measures whether the output follows task requirements, formatting constraints, and user intent.

### Reasoning Chain Stability

Measures consistency across multi-step reasoning prompts and whether late-stage output contradicts earlier constraints.

### Code Repair

Measures the ability to identify a code issue, propose a patch, and preserve surrounding behavior.

### Document QA

Measures answer accuracy against source documents and the ability to avoid unsupported claims.

### Spatial-Token Tasks

Measures tasks where output quality depends on global structure, ordering, layout, routing, or multi-position consistency.

## Latency Benchmark Pattern

Autoregressive generation normally requires one forward pass per generated token. Cassandra is designed to require a fixed number of denoising steps.

Example:

| Output Length | Autoregressive Forward Passes | Cassandra Target Steps |
| ---: | ---: | ---: |
| 128 tokens | 128 | 8-16 |
| 512 tokens | 512 | 8-16 |
| 1,024 tokens | 1,024 | 8-16 |

This is a design advantage only if quality, runtime overhead, and memory footprint remain competitive in measured deployments.

## Public Reporting Rule

Use this language until public evaluation artifacts are available:

> Cassandra T1 is designed to target a ~98% internal quality envelope against a Gemma-style autoregressive reference while reducing sequential generation passes. Public benchmark numbers will be published with model hashes, prompts, scoring scripts, and runtime settings when weights are released.

