# Model Development

Cassandra T1 was developed as a SophiaXT prototype for masked-diffusion language modeling. The development pattern is intentionally practical: define the architecture, train it far enough to produce a working checkpoint, test the checkpoint through local inference, then use failure modes to shape the next training cycle.

## Implemented Components

- `SophiaT1Config` defines the T1 model size, token IDs, denoising settings, and optional Q3-style additions.
- `SophiaT1Model` implements embeddings, transformer blocks, attention, FFN layers, masked loss, generation, and checkpoint remapping support.
- `spatial_tokens.py` defines an experimental coordinate-token vocabulary for future layout/spatial tasks.
- `pde_lattice.py` and `pde_scheduler.py` define scheduler experiments for masked-token unmasking.
- Training scripts cover scratch pretraining, continuation from epoch 5, diffusion training, QLoRA experimentation, and LoRA merge work.
- Inference scripts expose both local interactive generation and an OpenAI-style Flask endpoint.

## Development Readout

The epoch-5 checkpoint is the current verified proof-of-concept artifact. It shows that the architecture and training flow can produce a runnable model, but it is still early. Source notes describe short factual answers as more usable than long-form generations. Identity and coherence still need work.

The v2 scratch checkpoint is newer by timestamp and much larger, but preliminary comparison output shows instability. This is a useful development lesson: checkpoint recency is not the same as checkpoint quality. Future development should choose continuation points through evaluation, not file timestamps.

## What Should Improve Next

The model needs a tighter development loop:

- Fixed regression prompts before every training run.
- Separate measurement for factual QA, long-form coherence, identity, service-domain reasoning, code, and spatial/layout prompts.
- Better data balance for identity and instruction following.
- Active integration of spatial-token samples before spatial claims are made.
- Checkpoint cards that record hashes, losses, sample outputs, decoding settings, memory use, and known failures.

The goal is not to make the documentation sound larger than the model. The goal is to make each release more measurable than the last.
