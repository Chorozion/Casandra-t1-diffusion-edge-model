# Roadmap

## Short-Term

- Add `requirements.txt` or `pyproject.toml`.
- Replace absolute local paths in inference scripts with CLI arguments.
- Add a minimal checkpoint loading smoke test.
- Add model card metadata for each checkpoint.
- Add reproducible evaluation prompts and expected output logging.

## Training

- Continue training from the best stable checkpoint.
- Improve identity and instruction-following data.
- Evaluate whether v2 scratch should supersede epoch 5.
- Integrate spatial tokens into the active training pipeline if supported by data.

## Inference

- Add a clean local CLI.
- Add CPU/GPU device selection.
- Add configurable checkpoint path, tokenizer path, max tokens, steps, temperature, and top-p.
- Add a small server package that does not depend on local SophiaXT paths.

## Release

- Publish checksums with every model artifact.
- Add quantized formats if the custom architecture can be converted reliably.
- Add third-party reproducibility notes.
