# Inference Design

The release includes real Python inference scripts for Cassandra T1.

## Entry Points

- `scripts/run_cassandra.py`: interactive local CUDA inference for the epoch-5 checkpoint.
- `scripts/chunk_gen.py`: chunk-based semi-autoregressive generation experiment.
- `scripts/serve_ep5.py`: Flask server exposing an OpenAI-style `/v1/chat/completions` endpoint.
- `src/eval/canary_identity.py`: small identity canary generation helper.

## Model Loading

The scripts load:

- Checkpoint: `weights/cassandra_ep5_fp16.pt` or the original local path.
- Tokenizer: `release/tokenizer.json` or the original local path.
- Architecture: `src/model/sophia_t1.py`.
- Config: `src/model/config.py`.

Some scripts still contain original absolute paths from `I:\sophiat1` or `/opt/sophiaxt/cassandra`. Adjust these before running in a different environment.

## Generation

Cassandra T1 generation appends mask tokens after the prompt, predicts logits over masked positions, samples or ranks candidate tokens, reveals selected tokens over multiple denoising steps, and decodes the generated span.

The epoch-5 scripts use nucleus sampling, repetition penalties, chunk generation, and 8-16 denoising steps depending on the entry point.

## Limitations

- Long-form output is still unstable.
- The newest v2 scratch checkpoint needs evaluation before use.
- No clean package installer is included yet.
- No quantized runtime package is included.
