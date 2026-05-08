# SophiaXT Stack Showcase

Cassandra T1 shows the SophiaXT model stack as a working research pipeline: model architecture, scheduler logic, tokenizer artifact, checkpoints, training scripts, inference scripts, and documentation all live together in one open release.

## Stack Layers

- `src/model/`: PyTorch model architecture and configuration.
- `src/scheduler/`: PDE lattice and diffusion scheduler code.
- `src/train/`: scratch training, continuation training, QLoRA, and merge utilities.
- `scripts/`: local and server inference entry points.
- `release/tokenizer.json`: tokenizer used by the model scripts.
- `weights/`: released Git LFS checkpoint artifacts.

## Why It Matters

The repository demonstrates SophiaXT's ability to move from architecture concept to trained checkpoint, then into a runnable inference path. It is still experimental, but the release contains real artifacts rather than only concept diagrams.

## Current Boundary

The epoch-5 checkpoint remains a proof-of-concept. The newest v2 scratch checkpoint is included for transparency and future evaluation, but the release does not claim production quality or benchmark leadership.
