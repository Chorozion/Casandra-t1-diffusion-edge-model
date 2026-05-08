# Cassandra T1 Evaluation Script

The release includes one training canary and one general checkpoint evaluation runner.

## Existing Canary

`src/eval/canary_identity.py` checks whether identity prompts drift into competitor-brand contamination. It was designed for use during training and continuation runs.

## General T1 Evaluation

`src/eval/evaluate_t1.py` is the reusable evaluation runner for the T1/epoch-5 model line. It loads a checkpoint, runs a fixed prompt suite, writes raw JSONL outputs, and creates a Markdown report.

The script records:

- prompt category
- generated output
- latency per prompt
- word count
- unique-word ratio
- repeated-bigram ratio
- expected term hits
- left/right repetition count
- non-ASCII artifact ratio
- collapse flags
- identity contamination flags

## Intended Use

The script is meant to compare checkpoint behavior over time. It is not a formal public leaderboard benchmark. Its purpose is to make regressions visible: repetition collapse, identity drift, loss of domain relevance, output artifacts, and unstable long-form behavior.

Example after reassembling the epoch-5 checkpoint:

```bash
python src/eval/evaluate_t1.py \
  --checkpoint weights/cassandra_ep5_fp16.pt \
  --tokenizer release/tokenizer.json \
  --output-dir eval/runs
```

To test another checkpoint with the same prompt suite:

```bash
python src/eval/evaluate_t1.py \
  --checkpoint weights/v2_scratch_epoch2_82002.pt \
  --tokenizer release/tokenizer.json \
  --output-dir eval/runs
```

Reports are written to `eval/runs/`.
