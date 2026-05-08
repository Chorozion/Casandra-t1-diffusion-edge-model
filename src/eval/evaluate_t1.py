"""
Cassandra T1 evaluation runner.

This script loads a Cassandra/Sophia T1 checkpoint, runs a fixed prompt suite,
and writes both JSONL outputs and a Markdown summary report. It is designed for
prototype checkpoint comparison rather than leaderboard benchmarking.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from tokenizers import Tokenizer

from model.config import sophia_t1_base, sophia_t1_base_q3
from model.sophia_t1 import SophiaT1Model


PROMPTS = [
    {
        "id": "identity_who_are_you",
        "category": "identity",
        "prompt": "Who are you?",
        "expected_terms": ["cassandra", "sophia", "diffusion"],
    },
    {
        "id": "identity_company",
        "category": "identity",
        "prompt": "What company built you?",
        "expected_terms": ["sophia", "sophiaxt", "sophia xt"],
    },
    {
        "id": "architecture",
        "category": "architecture",
        "prompt": "What is your architecture?",
        "expected_terms": ["diffusion", "masked", "transformer"],
    },
    {
        "id": "generation_method",
        "category": "architecture",
        "prompt": "How do you generate text?",
        "expected_terms": ["mask", "diffusion", "denoise"],
    },
    {
        "id": "refrigerator_noise",
        "category": "technical_service",
        "prompt": "My refrigerator is making a loud humming noise. What could be wrong?",
        "expected_terms": ["compressor", "fan", "motor", "condenser", "refrigerator"],
    },
    {
        "id": "washer_drain",
        "category": "technical_service",
        "prompt": "What should I check if my washing machine will not drain?",
        "expected_terms": ["drain", "pump", "hose", "filter", "clog"],
    },
    {
        "id": "dryer_heat",
        "category": "technical_service",
        "prompt": "My dryer will not heat up. What should I check first?",
        "expected_terms": ["thermal", "element", "fuse", "vent", "dryer"],
    },
    {
        "id": "simple_science",
        "category": "general_qa",
        "prompt": "Explain quantum entanglement in simple terms.",
        "expected_terms": ["particles", "connected", "measure", "state"],
    },
    {
        "id": "short_writing",
        "category": "writing",
        "prompt": "Write a short paragraph about the ocean.",
        "expected_terms": ["ocean", "water", "waves", "sea"],
    },
    {
        "id": "code_reasoning",
        "category": "code",
        "prompt": "Explain what a Python dictionary is in two sentences.",
        "expected_terms": ["key", "value", "python", "dictionary"],
    },
]


CONTAMINATION = re.compile(
    r"\b(chatgpt|openai|anthropic|claude|gemini|bard|llama|mistral)\b",
    re.IGNORECASE,
)

REPEATED_LEFT_RIGHT = re.compile(r"\b(left|right)\b", re.IGNORECASE)
WORD_RE = re.compile(r"[A-Za-z0-9_]+")


@dataclass
class EvalConfig:
    checkpoint: Path
    tokenizer: Path
    output_dir: Path
    max_tokens: int
    steps: int
    temperature: float
    top_p: float
    beta: float
    rep_penalty: float
    use_q3_shell: bool
    device: str


def load_model(cfg: EvalConfig) -> tuple[SophiaT1Model, dict]:
    checkpoint = torch.load(str(cfg.checkpoint), map_location=cfg.device, weights_only=False)
    model_cfg = sophia_t1_base_q3() if cfg.use_q3_shell else sophia_t1_base()
    model = SophiaT1Model(model_cfg).to(cfg.device)

    state = checkpoint["model"]
    if cfg.use_q3_shell:
        state = SophiaT1Model.remap_baseline_state_dict(state, use_adaln=model_cfg.use_adaln)
    state = {k: v.float() if getattr(v, "is_floating_point", lambda: False)() else v for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    model.eval()

    metadata = {
        "epoch": checkpoint.get("epoch"),
        "loss": checkpoint.get("loss"),
        "missing_keys": len(result.missing_keys),
        "unexpected_keys": len(result.unexpected_keys),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    return model, metadata


def generate(model: SophiaT1Model, tok: Tokenizer, prompt: str, cfg: EvalConfig) -> tuple[str, float]:
    sys_prompt = "You are Cassandra T1, a diffusion language model by SophiaXT. Answer directly."
    full = f"Q: {sys_prompt}\n\n{prompt}\nA:"
    ids = torch.tensor([tok.encode(full).ids], device=cfg.device)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=cfg.max_tokens,
            num_steps=cfg.steps,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            beta=cfg.beta,
            rep_penalty=cfg.rep_penalty,
            timestep_power=1.5,
        )
    elapsed = time.time() - start
    text = tok.decode(out[0].tolist())
    text = text.replace(chr(288), " ").replace(chr(266), "\n").strip()
    return text, elapsed


def repeated_bigram_ratio(words: list[str]) -> float:
    if len(words) < 3:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    repeats = sum(1 for i in range(1, len(bigrams)) if bigrams[i] == bigrams[i - 1])
    return repeats / len(bigrams)


def score_output(text: str, expected_terms: Iterable[str]) -> dict:
    words = [w.lower() for w in WORD_RE.findall(text)]
    unique_words = set(words)
    total_words = len(words)
    unique_ratio = len(unique_words) / total_words if total_words else 0.0
    repeated_lr_count = len(REPEATED_LEFT_RIGHT.findall(text))
    expected_hits = [term for term in expected_terms if term.lower() in text.lower()]
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    char_count = len(text)

    collapse_flags = []
    if total_words and repeated_lr_count / max(total_words, 1) > 0.15:
        collapse_flags.append("left_right_repetition")
    if unique_ratio < 0.35 and total_words > 20:
        collapse_flags.append("low_unique_word_ratio")
    if repeated_bigram_ratio(words) > 0.15:
        collapse_flags.append("repeated_bigram_pattern")
    if char_count and non_ascii / char_count > 0.08:
        collapse_flags.append("high_non_ascii_artifact_ratio")

    return {
        "word_count": total_words,
        "char_count": char_count,
        "unique_word_ratio": round(unique_ratio, 4),
        "repeated_bigram_ratio": round(repeated_bigram_ratio(words), 4),
        "left_right_count": repeated_lr_count,
        "non_ascii_ratio": round(non_ascii / char_count, 4) if char_count else 0.0,
        "expected_term_hits": expected_hits,
        "expected_term_hit_count": len(expected_hits),
        "contamination_detected": bool(CONTAMINATION.search(text)),
        "collapse_flags": collapse_flags,
    }


def write_report(path: Path, records: list[dict], metadata: dict, cfg: EvalConfig) -> None:
    scores = [r["metrics"] for r in records]
    avg_unique = statistics.mean(s["unique_word_ratio"] for s in scores) if scores else 0.0
    avg_expected = statistics.mean(s["expected_term_hit_count"] for s in scores) if scores else 0.0
    collapses = sum(1 for s in scores if s["collapse_flags"])
    contamination = sum(1 for s in scores if s["contamination_detected"])

    lines = [
        "# Cassandra T1 Evaluation Report",
        "",
        "## Checkpoint",
        "",
        f"- Checkpoint: `{cfg.checkpoint}`",
        f"- Tokenizer: `{cfg.tokenizer}`",
        f"- Epoch: `{metadata.get('epoch')}`",
        f"- Loss: `{metadata.get('loss')}`",
        f"- Parameters: `{metadata.get('parameters')}`",
        f"- Missing keys after load: `{metadata.get('missing_keys')}`",
        f"- Unexpected keys after load: `{metadata.get('unexpected_keys')}`",
        "",
        "## Decode Settings",
        "",
        f"- Max tokens: `{cfg.max_tokens}`",
        f"- Steps: `{cfg.steps}`",
        f"- Temperature: `{cfg.temperature}`",
        f"- Top-p: `{cfg.top_p}`",
        f"- Beta: `{cfg.beta}`",
        f"- Repetition penalty: `{cfg.rep_penalty}`",
        f"- Q3 shell: `{cfg.use_q3_shell}`",
        "",
        "## Aggregate Signals",
        "",
        f"- Prompts: `{len(records)}`",
        f"- Average unique-word ratio: `{avg_unique:.4f}`",
        f"- Average expected-term hits: `{avg_expected:.2f}`",
        f"- Outputs with collapse flags: `{collapses}`",
        f"- Outputs with contamination flags: `{contamination}`",
        "",
        "## Prompt Results",
        "",
    ]

    for rec in records:
        lines.extend(
            [
                f"### {rec['id']}",
                "",
                f"- Category: `{rec['category']}`",
                f"- Latency: `{rec['latency_seconds']:.3f}s`",
                f"- Expected hits: `{', '.join(rec['metrics']['expected_term_hits']) or 'none'}`",
                f"- Collapse flags: `{', '.join(rec['metrics']['collapse_flags']) or 'none'}`",
                "",
                "Prompt:",
                "",
                f"> {rec['prompt']}",
                "",
                "Output:",
                "",
                "```text",
                rec["output"],
                "```",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate a Cassandra T1 checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer", default=Path("release/tokenizer.json"), type=Path)
    parser.add_argument("--output-dir", default=Path("eval/runs"), type=Path)
    parser.add_argument("--max-tokens", default=100, type=int)
    parser.add_argument("--steps", default=12, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--beta", default=0.5, type=float)
    parser.add_argument("--rep-penalty", default=1.3, type=float)
    parser.add_argument("--base-shell", action="store_true", help="Use base architecture shell instead of Q3 shell.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    return EvalConfig(
        checkpoint=args.checkpoint,
        tokenizer=args.tokenizer,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        steps=args.steps,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        rep_penalty=args.rep_penalty,
        use_q3_shell=not args.base_shell,
        device=args.device,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(cfg.tokenizer))
    model, metadata = load_model(cfg)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    jsonl_path = cfg.output_dir / f"cassandra-t1-eval-{run_id}.jsonl"
    report_path = cfg.output_dir / f"cassandra-t1-eval-{run_id}.md"

    records = []
    with jsonl_path.open("w", encoding="utf-8") as f:
        for item in PROMPTS:
            output, latency = generate(model, tok, item["prompt"], cfg)
            record = {
                "id": item["id"],
                "category": item["category"],
                "prompt": item["prompt"],
                "output": output,
                "latency_seconds": latency,
                "metrics": score_output(output, item["expected_terms"]),
            }
            records.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{item['id']}: {latency:.2f}s | flags={record['metrics']['collapse_flags']}")

    write_report(report_path, records, metadata, cfg)
    print(f"\nWrote JSONL: {jsonl_path}")
    print(f"Wrote report: {report_path}")


if __name__ == "__main__":
    main()
