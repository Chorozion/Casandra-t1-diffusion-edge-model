# Model Development

Cassandra T1 was developed as an experimental SophiaXT masked-diffusion language model. The open release contains the active architecture code, scheduler code, training scripts, tokenizer, and checkpoint artifacts from the source development directory.

## Implemented Components

- `SophiaT1Config` in `src/model/config.py`.
- `SophiaT1Model` in `src/model/sophia_t1.py`.
- Spatial-token helper code in `src/model/spatial_tokens.py`.
- PDE lattice scheduler in `src/scheduler/pde_lattice.py`.
- Training scripts for scratch training and epoch-5 continuation.
- Local/server inference scripts.

## Development Status

The epoch-5 checkpoint is verified but not production quality. The newer v2 scratch checkpoint is included as the latest artifact found, but it needs evaluation before it should be treated as the best checkpoint.

## Future Development Needs

- Cleaner package and dependency management.
- Path-independent inference scripts.
- Reproducible evaluation.
- Checkpoint-specific model cards.
- More training and validation.
