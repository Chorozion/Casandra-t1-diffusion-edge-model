# Cassandra T1 Overview

Cassandra T1 is an experimental SophiaXT masked-diffusion language model release. The repository includes real model source code, scheduler code, training scripts, inference scripts, tokenizer artifacts, and released checkpoints.

The project explores whether a small-to-mid-size language model can generate text through iterative masked-token denoising instead of purely autoregressive next-token prediction. This release should be treated as a research checkpoint package and architecture showcase, not a finished production assistant.

## Included in This Release

- PyTorch model architecture in `src/model/`.
- PDE lattice and diffusion scheduler code in `src/scheduler/`.
- Training scripts in `src/train/`.
- Inference and serving scripts in `scripts/`.
- Tokenizer artifact in `release/tokenizer.json`.
- Epoch-5 verified checkpoint and newest v2 scratch checkpoint in `weights/`, split into Git LFS parts under 2 GB each.

## Current Maturity

The epoch-5 checkpoint is documented by the source project as a verified checkpoint with loss around `2.2561`. Longer generations remain rough. The v2 scratch epoch-2 checkpoint is included because it is the newest checkpoint found in the source directory, but available comparison output indicates it is not necessarily better than epoch 5.

No private datasets, secrets, or server credentials are included.
