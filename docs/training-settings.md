# Training Settings

Training code is included in `src/train/` and top-level copied scripts under `scripts/`.

## Checkpoints in This Release

| Checkpoint | Notes |
|---|---|
| `weights/cassandra_ep5_fp16.pt.part001-002` | Verified epoch-5 FP16 checkpoint from `I:\sophiat1\checkpoints\epoch5`, split for GitHub LFS limits. |
| `weights/v2_scratch_epoch2_82002.pt.part001-009` | Newest checkpoint found in `I:\sophiat1\backup\runpod_20260422`, split for GitHub LFS limits. |

## Source Project Status

The source project notes describe:

- Epoch 1 loss: `3.78`
- Epoch 2 loss: `2.89`
- Epoch 3 loss: `2.67`
- Epoch 4 loss: `2.42`
- Epoch 5 loss: `2.2561`

The notes also state that epoch-5 short factual answers work better than long creative generations, and that additional training was planned.

## Training Implementation

Relevant files:

- `src/train/pretrain_from_scratch.py`
- `src/train/continue_from_ep5.py`
- `src/train/train_diffusion.py`
- `src/train/train_qlora.py`
- `scripts/cassandra_train_only.py`
- `src/model/sophia_t1.py`

The code uses masked-token objectives, random/span masking, beta-style mask-ratio sampling, PyTorch training loops, and checkpoint save/load behavior.

## Data

Training data is not included in this repository. The release intentionally excludes multi-GB JSONL datasets and private/local training corpora.
