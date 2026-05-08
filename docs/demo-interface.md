# Demo and Interface

The repository includes both a TypeScript interface sketch and real Python inference entry points.

## Python Inference

- `scripts/run_cassandra.py` runs local interactive inference.
- `scripts/chunk_gen.py` tests chunk-based generation.
- `scripts/serve_ep5.py` exposes an OpenAI-style Flask chat endpoint.

## TypeScript Interface

`examples/cassandra.demo.ts` documents the intended client-side API shape for applications that call a Cassandra backend. It should be treated as interface documentation, while the Python scripts are the real inference entry points in this release.
