# Inference Design

The current repository includes a placeholder TypeScript inference example. It documents the intended client interface, but it does not perform real model inference.

## Entry Point Scripts

Found:

- `examples/cassandra.demo.ts`

Not found:

- production inference server
- CLI generation script
- Python inference script
- checkpoint loader
- tokenizer loader
- model runtime implementation

## Model Loading

The placeholder client exposes:

```ts
static async load(options: CassandraLoadOptions)
```

The options include:

- `weights`
- `solver`
- `steps`
- `device`

The method currently returns a new client instance and includes a comment saying runtime initialization should be connected once weights and inference code are released.

## Tokenizer Use

No tokenizer files or tokenizer loading code were found in the repository.

## Masking Or Denoising Process

The example does not implement real masking or denoising. It returns a mock step list:

- masked field initialized
- semantic anchors stabilized
- decoded output field

This supports the conceptual demo but should not be represented as actual model execution.

## Generation Loop

The intended generation call is:

```ts
const output = await model.generate({
  prompt: "Summarize this repair log and route the next action.",
  maxTokens: 512,
  mode: "parallel-denoise",
  outputFormat: "report",
});
```

The output shape includes:

- `text`
- `confidence`
- `steps`

## Output Format

The TypeScript interfaces support output formats:

- `text`
- `json`
- `code`
- `report`

The placeholder method does not change behavior based on output format yet.

## Current Limitations

- No real model inference.
- No tokenizer use.
- No checkpoint loading.
- No server endpoint.
- No streaming output.
- No validation of generation options.
- No actual masked diffusion loop.
- No hardware-specific runtime logic.

