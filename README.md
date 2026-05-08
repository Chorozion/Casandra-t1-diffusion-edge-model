# Cassandra T1 Diffusion Edge Model

Cassandra T1 is an open SophiaXT masked-diffusion language model release. The repository includes the model architecture code, scheduler code, training/inference scripts, tokenizer, and released PyTorch checkpoints from the Sophia T1/Cassandra T1 development directory.

## Current Status

Cassandra T1 is an experimental proof-of-concept model, not a production-ready assistant. The public release includes real model artifacts and code, but output quality is still preliminary. The epoch-5 checkpoint is the verified Cassandra checkpoint described in the project notes. The newest checkpoint present in the local release source is the v2 scratch epoch-2 checkpoint from April 22, 2026.

## Released Files

| Path | Purpose | Size | SHA256 |
|---|---:|---:|---|
| `weights/cassandra_ep5_fp16.pt.part001-002` | Verified epoch-5 Cassandra T1 FP16 checkpoint, split for GitHub LFS object limits | 2,659,500,664 bytes reassembled | `D70C813C513F5232A25313FA60338F862020BA942ED26D54F62511766FA5F044` |
| `weights/v2_scratch_epoch2_82002.pt.part001-009` | Latest checkpoint found in `I:\sophiat1`, v2 scratch epoch 2 / step 82002, split for GitHub LFS object limits | 16,665,974,950 bytes reassembled | `8BFB5644209AB8FD241D85A725781495B29922811A2BECF8260AAAAD0A26DA6F` |
| `release/tokenizer.json` | Cassandra tokenizer | 2,253,607 bytes | `376A9537FCE79B7004237845E6B2C9991661E6BAEEA0B76AF9C9A3C1EB405C4D` |

Large files are stored through Git LFS. The checkpoints are split into parts because GitHub rejects individual LFS objects above 2 GB.

Reassemble on Windows PowerShell:

```powershell
Get-Content weights\cassandra_ep5_fp16.pt.part* -Encoding Byte -ReadCount 0 | Set-Content weights\cassandra_ep5_fp16.pt -Encoding Byte
Get-Content weights\v2_scratch_epoch2_82002.pt.part* -Encoding Byte -ReadCount 0 | Set-Content weights\v2_scratch_epoch2_82002.pt -Encoding Byte
```

Reassemble on Linux/macOS:

```bash
cat weights/cassandra_ep5_fp16.pt.part* > weights/cassandra_ep5_fp16.pt
cat weights/v2_scratch_epoch2_82002.pt.part* > weights/v2_scratch_epoch2_82002.pt
```

## What Cassandra T1 Demonstrates

- A masked-diffusion language model implemented in PyTorch.
- Parallel token denoising instead of purely left-to-right autoregressive decoding.
- A custom PDE lattice scheduler for choosing token unmasking steps.
- Grouped-query attention, RoPE, RMSNorm, SwiGLU feed-forward layers, and a compact BPE vocabulary.
- Experimental spatial-token vocabulary design.
- Training scripts for scratch pretraining, continuation training, QLoRA experimentation, and LoRA merge support.
- Local and Flask-based inference scripts for the epoch-5 checkpoint.

## Architecture Overview

The model code is in `src/model/sophia_t1.py` and `src/model/config.py`.

Default T1-Base configuration:

- Vocabulary: 32,768 tokens
- Hidden size: 2,048
- Layers: 28
- Query heads: 16
- KV heads: 4
- FFN intermediate size: 5,632
- Attention: grouped-query attention with sliding-window/global-token design
- Position encoding: RoPE
- Normalization: RMSNorm
- Diffusion: masked-token training and iterative unmasking

The scheduler implementations are in `src/scheduler/pde_lattice.py` and `src/scheduler/pde_scheduler.py`.

## Training Status

The release contains a verified epoch-5 checkpoint and a newer v2 scratch checkpoint. Project notes from the source directory describe epoch 5 as loss `2.2561` and state that output quality is still rough for longer generations. The v2 scratch comparison notes show that newer does not automatically mean better output quality, so both checkpoints should be treated as experimental research artifacts.

No private training data is included in this repository.

## Inference Overview

Useful entry points:

- `scripts/run_cassandra.py`: local interactive CUDA inference for the epoch-5 checkpoint.
- `scripts/chunk_gen.py`: chunk-based generation experiment.
- `scripts/serve_ep5.py`: Flask OpenAI-style chat endpoint for an epoch-5 deployment.
- `src/eval/canary_identity.py`: small identity canary evaluation helper.

The scripts include local path assumptions from the original development environment. Adjust checkpoint and tokenizer paths before running in a new environment.

## Technology Stack

- Language: Python
- ML framework: PyTorch
- Tokenizer: `tokenizers`
- Serving: Flask script for epoch-5 checkpoint
- Release storage: Git LFS
- Documentation: Markdown
- License: Apache License 2.0

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── docs/
├── examples/
├── release/
│   ├── README.md
│   └── tokenizer.json
├── scripts/
│   ├── run_cassandra.py
│   ├── chunk_gen.py
│   ├── serve_ep5.py
│   └── cassandra_train_only.py
├── src/
│   ├── model/
│   ├── scheduler/
│   ├── train/
│   └── eval/
├── weights/
│   ├── README.md
│   ├── checksums.sha256
│   ├── cassandra_ep5_fp16.pt.part001
│   ├── cassandra_ep5_fp16.pt.part002
│   └── v2_scratch_epoch2_82002.pt.part001-009
└── showcase/
```

## Limitations

- This is an experimental open release, not a polished production model.
- The epoch-5 checkpoint is verified but still produces rough long-form output.
- The newest v2 scratch checkpoint is included as the latest artifact found, but comparison output in the source directory indicates it may not be better than epoch 5.
- Formal benchmark results are not included.
- The released scripts may require path cleanup before running outside the original SophiaXT development machine.
- Training datasets are not included.

## Roadmap

- Add a clean `requirements.txt` or `pyproject.toml`.
- Add path-configurable inference commands.
- Add reproducible evaluation scripts.
- Add model card metadata for every checkpoint.
- Add quantized release formats if conversion support is available for the custom architecture.
- Continue training with better identity, instruction, and spatial data.

## Security Note

Credentials, API keys, server passwords, private deployment notes, and training datasets are intentionally excluded from this release. Do not commit local status files that contain infrastructure credentials.

## Disclaimer

Cassandra T1 is released as an experimental model architecture and checkpoint package. It should not be represented as production-ready, benchmark-leading, safety-validated, or commercially deployed at scale without additional evidence and evaluation.
