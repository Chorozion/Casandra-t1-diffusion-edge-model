# Architecture Overview

Cassandra T1 is implemented as a PyTorch masked-diffusion language model. The release contains the model implementation, scheduler code, tokenizer, and checkpoint artifacts.

## Core Files

- `src/model/sophia_t1.py`: transformer and generation implementation.
- `src/model/config.py`: T1 configuration values.
- `src/model/spatial_tokens.py`: experimental spatial-token vocabulary helpers.
- `src/scheduler/pde_lattice.py`: PDE lattice unmasking scheduler.
- `src/scheduler/pde_scheduler.py`: diffusion scheduler utilities.

## Generation Concept

The model appends masked output positions after a prompt, predicts all masked positions, reveals selected tokens over several steps, and decodes the resulting sequence.
