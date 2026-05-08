# Coherence Comparison: T1 Epoch 5 vs V2 Scratch Epoch 2

This note documents the observed coherence difference between the verified Cassandra T1 epoch-5 checkpoint and the newer v2 scratch epoch-2 checkpoint. The comparison is based on the captured test output from `I:\sophiat1\test_v2_comparison.txt`.

The comparison should be read as a qualitative development test, not a formal benchmark. It is useful because both checkpoints were tested against the same prompt set, exposing different failure modes.

## Compared Checkpoints

| Checkpoint | Release artifact | Source note |
|---|---|---|
| Cassandra T1 epoch 5 | `weights/cassandra_ep5_fp16.pt.part001-002` | Recorded loss near `2.26`; verified checkpoint used by the epoch-5 inference scripts. |
| V2 scratch epoch 2 / step 82002 | `weights/v2_scratch_epoch2_82002.pt.part001-009` | Newer checkpoint by timestamp; comparison output indicates severe repetition collapse. |

## Prompt Set

The captured test used ten prompts:

1. Who are you?
2. Are you ChatGPT?
3. What company built you?
4. How do you generate text?
5. My refrigerator is making a loud humming noise, what could be wrong?
6. What should I check if my washing machine won't drain?
7. My dryer won't heat up, what's wrong?
8. Explain quantum entanglement in simple terms.
9. Write a short paragraph about the ocean.
10. What is your architecture?

## Observed Coherence Difference

### Cassandra T1 Epoch 5

Epoch 5 does not yet produce coherent assistant-quality answers in this captured test. Its outputs contain fragmented words, corrupted token pieces, punctuation artifacts, and topic drift. However, its failure mode is broader: it samples a wider vocabulary and occasionally surfaces domain-relevant fragments such as appliance, technical, architecture, denoising, or Sophia-related tokens.

That suggests the checkpoint has learned some distributional structure, but not enough stable decoding behavior or instruction alignment to reliably form complete responses.

### V2 Scratch Epoch 2

The v2 scratch checkpoint shows a much more severe collapse pattern. Across prompts, the output repeatedly falls into sequences dominated by `left`, `right`, `The`, `A`, `Q`, punctuation, and short repeated scaffolding tokens.

This is a stronger coherence failure than the epoch-5 output. The model appears to be stuck in a narrow local token pattern rather than sampling semantically varied language. Based on this test, the v2 scratch checkpoint should not be treated as an improvement over epoch 5 without further training or debugging.

## Development Interpretation

The key result is not that epoch 5 is production-ready. It is not. The useful result is that the newer v2 scratch checkpoint has a worse observed coherence profile under this prompt set.

For future training, checkpoint selection should be based on stability and evaluation output, not timestamp or raw file size. Epoch 5 is currently the more useful baseline for continuing practical inference work, while v2 scratch should be investigated for repetition collapse, tokenizer/data imbalance, scheduler mismatch, or checkpoint-state issues.

## Recommended Next Measurement

The next evaluation pass should add:

- fixed decoding parameters recorded in the result file
- exact checkpoint hashes
- latency per prompt
- output length
- repeated-token ratio
- unique-token ratio
- human coherence rating from 1 to 5
- identity accuracy rating
- domain relevance rating

This would turn the current qualitative comparison into a repeatable coherence scorecard.

## Raw Test Transcript

The raw captured output is included at:

`eval/v2-ep2-vs-t1-ep5-coherence-raw.txt`
