# Roadmap

This roadmap is based on the current repository state.

## Short-Term Cleanup

- Keep the README aligned with the proof-of-concept status.
- Remove or qualify unsupported benchmark claims.
- Add a repository inventory table that distinguishes implemented code from architecture intent.
- Add instructions for running the placeholder demo if a package file is introduced.

## Documentation Improvements

- Add model-card style documentation.
- Add a reproducibility section.
- Add hardware and runtime assumptions.
- Add a glossary for masked diffusion, denoising, confidence maps, and PDE-lattice scheduling.

## Training Improvements

- Add training configuration.
- Add tokenizer documentation.
- Add dataset preparation notes.
- Add checkpoint behavior.
- Add training logs for the 5-epoch prototype if safe to publish.
- Add optimizer, batch size, and learning-rate details.

## Evaluation Improvements

- Add fixed prompt sets.
- Add scoring rubrics.
- Add evaluation scripts.
- Add latency measurement scripts.
- Add benchmark artifact versioning.
- Publish model hashes when weights are released.

## Inference Improvements

- Replace the placeholder TypeScript client with a real runtime or clearly separate mock/demo code.
- Add model loading.
- Add tokenizer loading.
- Add denoising loop implementation.
- Add output validation.
- Add error handling.
- Add streaming or progress events if useful.

## Deployment Improvements

- Add a server implementation only after API security requirements are clear.
- Add Docker or deployment files if public deployment is intended.
- Add CI checks.
- Add secret scanning.
- Add release tagging.

## Platform Integration Improvements

- Document how Cassandra connects to SophiaXT services.
- Add a safe integration interface.
- Add examples for workflow automation, document QA, diagnostic reasoning, and edge inference.
- Keep private production credentials and customer workflows out of the repository.

