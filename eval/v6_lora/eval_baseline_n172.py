"""Re-eval v2_ltmi_triple on the n=172 expanded held-out corpus.

This is the baseline arm that all 18 LoRA variants are compared against.
Same code path as run_t2_multidomain_eval — only the corpus pointer
changes.

Output: eval_out/lora_baseline_v2_ltmi_triple_n172.jsonl
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "cassandra_src"))

import torch

import run_t2_multidomain_eval as eval_mod
from cassandra_loader import load


def main():
    # Point eval_mod at expanded corpus
    extended = ROOT / "queries_heldout_extended_v2.json"
    eval_mod.QUERIES_EXT = extended
    print(f"[baseline] using extended corpus: {extended.name}")

    queries = json.loads(extended.read_text(encoding="utf-8"))
    total_q = sum(len(v) for k, v in queries.items() if isinstance(v, list))
    print(f"[baseline] total queries: {total_q}")

    torch.manual_seed(eval_mod.SEED)

    out_path = ROOT / "eval_out" / "lora_baseline_v2_ltmi_triple_n172.jsonl"
    print(f"[baseline] loading v2_ltmi_triple...")
    model, tok, _ = load("v2_ltmi_triple", dtype=torch.bfloat16)
    model.eval()
    supply_ltmi = bool(getattr(model.config, "attention_use_ltmi_priors", False))

    rows = eval_mod.evaluate_checkpoint(model, tok, "v2_ltmi_triple",
                                         supply_ltmi_priors=supply_ltmi)
    print(f"[baseline] wrote {len(rows)} rows")

    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Also dump a quick summary
    summary = eval_mod.summarize(rows)
    print(f"\n[baseline] summary:")
    for k, v in summary.items():
        if k.endswith("/forced") or k.endswith("/unforced"):
            print(f"  {k:<45} n={v['n']:<4} overlap={v['corpus_overlap_avg']:.4f}  "
                  f"eng={v['english_ratio_avg']:.4f}")

    print(f"\n[baseline] saved → {out_path}")


if __name__ == "__main__":
    main()
