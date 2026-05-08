# Roadmap

Cassandra T1 should move forward as a measured research program, not just a sequence of larger checkpoint uploads. The next work should make the model easier to run, easier to evaluate, and easier to improve.

## Short-Term Release Cleanup

- Add `requirements.txt` or `pyproject.toml`.
- Replace absolute local paths in inference scripts with CLI arguments.
- Add a clean local inference command.
- Add checkpoint reassembly verification helpers.
- Add checkpoint-specific model cards.

## Training Improvements

- Continue from the most stable checkpoint after a fixed evaluation pass.
- Improve identity and instruction-following data quality.
- Add stronger service-domain reasoning data.
- Use canary prompts during training and save outputs by checkpoint.
- Track loss, qualitative output, latency, and memory together.

## Architecture Improvements

- Evaluate the Q3-style AdaLN continuation path under controlled settings.
- Train the confidence head rather than relying on handcrafted confidence only.
- Integrate spatial tokens into actual tokenized training data.
- Compare fixed top-k, quantile unmasking, and PDE lattice schedules under the same prompt suite.

## Evaluation Improvements

- Add reproducible prompt sets.
- Add raw output logs.
- Add latency and VRAM measurement scripts.
- Add a benchmark document per release.
- Clearly separate internal notes from public benchmark claims.

## Deployment Improvements

- Package the Flask server as a configurable runtime.
- Add CPU/GPU device selection.
- Add runtime health and model metadata endpoints.
- Investigate quantized release formats if the custom architecture can be converted reliably.
