# Release Safety Notes

Cassandra T1 documentation should be public, technically useful, and careful about claims. Model weights, private datasets, infrastructure credentials, and unreleased benchmark artifacts should not be committed.

## Answer First: What Should Not Go On GitHub?

Do not commit model weights, private datasets, API keys, production URLs with secrets, customer data, internal logs, or unverified benchmark claims presented as final public results.

## Safe To Publish

- Architecture diagrams
- Public technical summaries
- API interface examples
- Demo mockups
- Benchmark methodology
- Reproducibility plans
- Non-secret config examples
- Public website links

## Not Safe To Publish

- `.env`
- API keys
- SSH keys
- private datasets
- customer records
- model weights before release
- training logs with private paths
- unpublished benchmark datasets
- paid vendor credentials

## Benchmark Claim Policy

Use cautious language:

- "target"
- "internal benchmark board"
- "preliminary"
- "requires public reproducibility package"
- "reference envelope"

Avoid overclaiming:

- "proven better than"
- "guaranteed"
- "beats all models"
- "production-certified" unless actually certified

## Suggested Repository Visibility

Use a private repository until:

- legal ownership is clear
- no secrets are present
- public claims are reviewed
- benchmark language is stable
- model release plan is decided

