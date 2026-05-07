# Cassandra T1 Weights

This directory is reserved for the open-release Cassandra T1 model artifacts.

Git LFS is configured for this directory so model weights, checkpoints, tokenizer assets, and packaged release artifacts can be published without storing large binary files directly in normal Git history.

## Expected Artifact Types

- `.safetensors`
- `.gguf`
- `.pt`
- `.pth`
- `.onnx`
- `.ckpt`

## Current Status

No Cassandra T1 weight file is included in this directory yet.

The repository is prepared for an open release, but the actual weight artifact must be added from the trained model output before this section can claim public weights are available.

## Adding Weights

Place the release artifact in this directory and commit it normally. Git LFS will track it automatically:

```bash
git add weights/
git commit -m "Add Cassandra T1 open-release weights"
git push origin main
```

Before publishing, verify that the artifact does not contain secrets, private datasets, or unrelated build output.
