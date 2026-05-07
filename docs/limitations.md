# Limitations

Cassandra T1 is an early architecture concept and technical showcase. The current repository should not be interpreted as a production model release.

## Known Limitations

- The current training status is only 5 epochs.
- No production-ready model weights are included.
- No real training script was found.
- No tokenizer files were found.
- No dataset loading logic was found.
- No evaluation scripts were found.
- No formal benchmark report was found.
- No inference server was found.
- The TypeScript client is a placeholder.
- No deployment files were found.
- No tests were found.
- No CI workflow was found.

## Validation Limits

Because the repository does not include weights, training scripts, eval scripts, or datasets, reviewers cannot reproduce model quality claims from the current files alone.

Any benchmark numbers should be treated as preliminary targets unless supported by reproducible evaluation assets.

## Stability Limits

The inference design is not stable yet. The current API shape may change once real runtime code, tokenizer handling, checkpoint loading, and hardware-specific execution are added.

## Security Notes

No secrets were found in the documentation package. The repository includes `.env.example`, but real `.env` files and keys should never be committed.

## Placeholder Areas

The file `examples/cassandra.demo.ts` explicitly contains placeholder behavior. The comments state that actual runtime initialization should be connected after weights and inference code are released.

