# SophiaXT Stack Showcase

The Cassandra T1 repository showcases the SophiaXT stack as an architecture direction: model concept, inference interface, release discipline, and future platform integration.

## Architecture Concept

The stack is organized around a masked-diffusion language model concept. Cassandra T1 is presented as a model that would refine masked token fields through denoising steps rather than generate strictly left-to-right.

## Training Pipeline

The current public repository does not include the training pipeline. The project context states that Cassandra T1 has completed 5 epochs, but the training code, dataset loader, tokenizer files, and checkpoint behavior are not present.

## Inference Path

The current inference path is represented by `examples/cassandra.demo.ts`. It defines a developer-facing API:

- load model options
- generate with a prompt
- return text, confidence, and denoising-step metadata

This is a placeholder implementation and does not run real model inference.

## Configuration

The repository includes `.env.example` with future endpoint placeholders:

- `CASSANDRA_API_BASE_URL`
- `CASSANDRA_API_KEY`

No server-side runtime consumer was found.

## Developer Workflow

The current developer workflow is documentation review and interface inspection. A future workflow should add installation instructions, training commands, inference commands, and test commands.

## Platform Integration

The repository is positioned as part of SophiaXT's broader model and workflow stack. Based on the current files, platform integration is conceptual rather than implemented in this repo.

## Future Edge-Native Direction

The example load options include `cpu`, `edge-gpu`, and `cuda`, suggesting an intended edge-native direction. No hardware-specific runtime implementation is currently included.

