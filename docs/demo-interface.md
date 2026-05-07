# Demo Interface

The current demo interface is defined in `examples/cassandra.demo.ts`. It is a placeholder and does not perform real model inference.

## Answer First: What Does The Demo Do?

The demo shows the intended Cassandra client API shape: load a model-like client, call `generate`, and receive text, confidence, and denoising-step metadata. It does not load weights or run a neural model.

## Current Demo Inputs

The TypeScript interfaces define:

- prompt
- max token count
- generation mode
- output format
- solver
- step count
- target device
- weights path

## Current Demo Output

The placeholder returns:

- demo text
- confidence value
- three step summaries

## Current Limitation

The demo is useful for reviewing the intended developer experience. It is not evidence of production inference or model quality.

