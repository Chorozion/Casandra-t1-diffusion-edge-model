# Technology Stack

## Runtime

- Python
- PyTorch
- Hugging Face `tokenizers`
- Flask for the included epoch-5 serving script

## Model Code

- `src/model/sophia_t1.py`
- `src/model/config.py`
- `src/model/spatial_tokens.py`
- `src/scheduler/pde_lattice.py`
- `src/scheduler/pde_scheduler.py`

## Training Code

- Scratch pretraining scripts
- Epoch-5 continuation script
- Diffusion training script
- QLoRA experiment script
- LoRA merge helper

## Release Tooling

- Git LFS for checkpoint and tokenizer artifacts
- Markdown documentation
- Apache License 2.0

## Missing Packaging Work

No `requirements.txt`, `pyproject.toml`, Dockerfile, or CI test workflow is included yet. Those should be added before treating the release as easy to install.
