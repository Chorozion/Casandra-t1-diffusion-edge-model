# Limitations

Cassandra T1 is an experimental open model release.

## Known Limitations

- The epoch-5 checkpoint is verified but not production-ready.
- Long-form outputs can degrade or become incoherent.
- The latest v2 scratch checkpoint is included, but source comparison output shows unstable behavior.
- No formal benchmark suite is included.
- No packaged dependency file is included yet.
- Scripts contain local path assumptions from the original development environment.
- Training datasets are excluded.
- Spatial token code exists, but source notes indicate it was not fully active in the released epoch-5 training run.
- GGUF conversion is not included because the architecture is custom.

## Safety and Release Boundaries

The repository intentionally excludes credentials, API keys, server passwords, private deployment notes, and training data. The included checkpoints should be evaluated before use in any user-facing system.
