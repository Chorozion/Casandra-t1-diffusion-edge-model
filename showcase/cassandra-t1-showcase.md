# Cassandra T1 Open Release

Cassandra T1 is SophiaXT's experimental masked-diffusion language model release. It includes actual checkpoints, tokenizer files, PyTorch architecture code, scheduler logic, and inference scripts.

## What It Demonstrates

- Masked-token diffusion generation for language.
- Parallel denoising over multiple generation steps.
- PDE lattice scheduling experiments.
- A compact BPE tokenizer and custom transformer architecture.
- A path from training checkpoint to local/server inference.

## Current Training Status

The release includes:

- `weights/cassandra_ep5_fp16.pt.part001-002`: verified epoch-5 checkpoint split for GitHub LFS limits.
- `weights/v2_scratch_epoch2_82002.pt.part001-009`: newest checkpoint found in the source directory, split for GitHub LFS limits.
- `release/tokenizer.json`: tokenizer used by the released scripts.

The epoch-5 checkpoint is useful as an architecture validation artifact. It is not a finished assistant. Source notes state that short factual answers work better than long creative output and that additional training is needed.

## Why It Matters

Most language model demos stop at a UI. Cassandra T1 exposes the architecture, training path, model weights, tokenizer, and inference code so technical reviewers can inspect the actual system.

## Current Limitations

- Output quality is preliminary.
- No formal benchmark suite is included.
- Scripts need path cleanup for portable use.
- Training data is not included.
- Quantized deployment formats are not included yet.

## Next Milestones

- Add installable Python package metadata.
- Add reproducible evaluation scripts.
- Add checkpoint-specific model cards.
- Continue training and publish updated checkpoints with hashes.
