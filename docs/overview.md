# Cassandra T1 Overview

Cassandra T1 is a SophiaXT prototype model family release for masked-diffusion language generation. It includes actual model code, scheduler code, tokenizer artifacts, checkpoint artifacts, training scripts, and inference scripts. The release is meant to be read as a lab notebook turned public: concrete enough to inspect, honest enough to show what is still unfinished.

## What the Project Is Testing

The central idea is simple but technically demanding: generate language by refining masked token positions over several denoising steps, rather than committing every output token strictly from left to right. Cassandra T1 uses a transformer backbone, but the sampling loop treats generation as iterative reconstruction.

## What Is Included

- PyTorch architecture implementation in `src/model/`.
- PDE lattice and scheduler experiments in `src/scheduler/`.
- Training and continuation scripts in `src/train/`.
- Local and server inference scripts in `scripts/`.
- Tokenizer artifact in `release/tokenizer.json`.
- Epoch-5 verified checkpoint and the latest v2 scratch checkpoint, split into Git LFS parts under 2 GB each.
- Research-style documentation in `docs/research-paper.md`.

## Current Maturity

The epoch-5 checkpoint shows that the architecture can train and produce a checkpoint usable by the release inference path. Source notes describe short factual answers as more usable than long-form generation, while identity consistency and extended coherence need more training.

The latest v2 scratch checkpoint is included because it is the newest artifact found in the source directory. It is not described as the best checkpoint. Current comparison notes indicate instability, so it should be evaluated carefully before being treated as a successor to epoch 5.

## Why It Matters

The release gives reviewers a tangible architecture package: model code, training code, tokenizer, scheduler, inference path, and real checkpoints. That makes Cassandra T1 more than a concept page. It is a prototype that can now be measured, criticized, and improved.
