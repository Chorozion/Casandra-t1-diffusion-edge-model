# Training Settings and Measurements

This release includes the training code used to support the Cassandra T1 prototype direction. The scripts are not yet packaged into a polished training framework, but they show the actual implementation pattern: masked-token objectives, random and span masking, curriculum-style mask ratios, category weighting experiments, AdamW optimization, checkpointing, and canary-style qualitative checks.

## Released Checkpoints

| Checkpoint parts | Source | Meaning |
|---|---|---|
| `weights/cassandra_ep5_fp16.pt.part001-002` | `I:\sophiat1\checkpoints\epoch5\cassandra_ep5_fp16.pt` | Verified epoch-5 FP16 checkpoint used by the release inference scripts after reassembly. |
| `weights/v2_scratch_epoch2_82002.pt.part001-009` | `I:\sophiat1\backup\runpod_20260422\v2_scratch_epoch2_82002.pt` | Newest checkpoint found by timestamp; included for analysis, not promoted as the best checkpoint. |

## Recorded Loss Progression

| Stage | Recorded loss |
|---|---:|
| Epoch 1 | 3.78 |
| Epoch 2 | 2.89 |
| Epoch 3 | 2.67 |
| Epoch 4 | 2.42 |
| Epoch 5 | 2.2561 |

These measurements show optimization progress, not public benchmark performance. The most useful interpretation is that the epoch-5 training line reached a checkpoint worth inspecting, but not a finished assistant.

## Training Implementation Details

Relevant files:

- `src/train/pretrain_from_scratch.py`
- `src/train/continue_from_ep5.py`
- `src/train/train_diffusion.py`
- `src/train/train_qlora.py`
- `scripts/cassandra_train_only.py`

The training code includes:

- Masked-position cross entropy.
- Random masking and span masking.
- Mask-ratio curriculum behavior.
- AdamW with betas `(0.9, 0.95)` and weight decay.
- Mixed precision with gradient scaling in the scratch/continuation scripts.
- Periodic and epoch-boundary checkpoint saves.
- Canary prompt checks for qualitative regression tracking.
- Category weighting experiments for identity and service-domain data.
- Z-loss regularization in continuation experiments.

## Future Training Improvements

The next training phase should be measurement-first:

- Choose a fixed prompt suite before training starts.
- Compare epoch 5, v2 scratch, and future checkpoints under identical decoding settings.
- Track short factual QA, long generation, identity consistency, appliance/service reasoning, code, and spatial/layout prompts separately.
- Add model cards with checkpoint hashes, training source summary, loss curves, known failures, and recommended decoding settings.
- Integrate spatial tokens into real tokenized samples before measuring spatial capability.

Training data is not included in this repository.
