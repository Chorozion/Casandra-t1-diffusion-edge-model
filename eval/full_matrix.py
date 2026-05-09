"""Build the full 4-cell × strict/lenient matrix from the two eval logs.

Cells:
  baseline_unforced (broken model, no LTMi-XT)
  baseline_forced   (broken model + LTMi-XT-anchored decoding)
  lora_unforced     (LoRA-trained model, no anchors)
  lora_forced       (LoRA-trained model + anchors)

Produces:
  - eval_matrix.json (machine-readable)
  - eval_matrix.md   (publication-ready table + per-query detail)
"""
from __future__ import annotations
import json, re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "eval_out"


def strict(text: str, kws: list[str]) -> bool:
    t = text.lower()
    return all(kw.lower() in t for kw in kws)

def lenient(text: str, kws: list[str]) -> bool:
    no_ws = re.sub(r"\s+", "", text.lower())
    return all(re.sub(r"\s+", "", kw.lower()) in no_ws for kw in kws)


def main():
    forced_log = [json.loads(l) for l in (OUT / "forced_eval_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lora_log   = [json.loads(l) for l in (OUT / "lora_eval_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    # Index by (corpus, query)
    bidx = {(r["corpus"], r["query"]): r for r in forced_log}
    lidx = {(r["corpus"], r["query"]): r for r in lora_log}

    keys = list(bidx.keys())
    n = len(keys)

    cells = {
        "baseline_unforced": {"hits_strict": 0, "hits_lenient": 0, "ms": [], "anchor": []},
        "baseline_forced":   {"hits_strict": 0, "hits_lenient": 0, "ms": [], "anchor": []},
        "lora_unforced":     {"hits_strict": 0, "hits_lenient": 0, "ms": [], "anchor": []},
        "lora_forced":       {"hits_strict": 0, "hits_lenient": 0, "ms": [], "anchor": []},
    }

    per_query = []
    for k in keys:
        b = bidx[k]; l = lidx[k]
        kws = b["expected_keywords"]
        bu_text = b["unforced"]["text"]; bf_text = b["forced"]["text"]
        lu_text = l["lora_unforced"]["text"]; lf_text = l["lora_forced"]["text"]

        bu_s, bu_l = strict(bu_text, kws), lenient(bu_text, kws)
        bf_s, bf_l = strict(bf_text, kws), lenient(bf_text, kws)
        lu_s, lu_l = strict(lu_text, kws), lenient(lu_text, kws)
        lf_s, lf_l = strict(lf_text, kws), lenient(lf_text, kws)

        cells["baseline_unforced"]["hits_strict"]  += int(bu_s)
        cells["baseline_unforced"]["hits_lenient"] += int(bu_l)
        cells["baseline_unforced"]["ms"].append(b["unforced"]["ms"])

        cells["baseline_forced"]["hits_strict"]  += int(bf_s)
        cells["baseline_forced"]["hits_lenient"] += int(bf_l)
        cells["baseline_forced"]["ms"].append(b["forced"]["ms"])
        cells["baseline_forced"]["anchor"].append(b["forced"]["anchor_preservation"])

        cells["lora_unforced"]["hits_strict"]  += int(lu_s)
        cells["lora_unforced"]["hits_lenient"] += int(lu_l)
        cells["lora_unforced"]["ms"].append(l["lora_unforced"]["ms"])

        cells["lora_forced"]["hits_strict"]  += int(lf_s)
        cells["lora_forced"]["hits_lenient"] += int(lf_l)
        cells["lora_forced"]["ms"].append(l["lora_forced"]["ms"])
        cells["lora_forced"]["anchor"].append(l["lora_forced"]["anchor_preservation"])

        per_query.append({
            "corpus": k[0], "query": k[1],
            "expected_keywords": kws,
            "retrieved_locus": b["retrieved_statement"],
            "baseline_unforced": {"text": bu_text, "strict": bu_s, "lenient": bu_l},
            "baseline_forced":   {"text": bf_text, "strict": bf_s, "lenient": bf_l, "anchor_preservation": b["forced"]["anchor_preservation"]},
            "lora_unforced":     {"text": lu_text, "strict": lu_s, "lenient": lu_l},
            "lora_forced":       {"text": lf_text, "strict": lf_s, "lenient": lf_l, "anchor_preservation": l["lora_forced"]["anchor_preservation"]},
        })

    # Pretty-print
    print(f"\n{'='*78}\nFULL EVAL MATRIX — Cassandra T1 v1 + LTMi-XT (n={n})\n{'='*78}")
    print(f"{'cell':<22} {'strict':>14} {'lenient':>14} {'avg ms':>10} {'anchor':>10}")
    for name, c in cells.items():
        s, l = c["hits_strict"], c["hits_lenient"]
        avg_ms = mean(c["ms"]) if c["ms"] else 0
        avg_anchor = (mean(c["anchor"]) * 100) if c["anchor"] else float("nan")
        anchor_str = f"{avg_anchor:.1f}%" if not (avg_anchor != avg_anchor) else "  n/a"
        print(f"{name:<22} {s}/{n} = {s/n*100:>4.1f}% {l}/{n} = {l/n*100:>4.1f}%  {avg_ms:>7.0f}   {anchor_str:>8}")

    out = {
        "n_queries": n,
        "cells": {
            name: {
                "hits_strict": c["hits_strict"],
                "hits_lenient": c["hits_lenient"],
                "strict_pct": round(c["hits_strict"] / n * 100, 1),
                "lenient_pct": round(c["hits_lenient"] / n * 100, 1),
                "avg_ms": round(mean(c["ms"]), 1) if c["ms"] else None,
                "avg_anchor_preservation": round(mean(c["anchor"]) * 100, 1) if c["anchor"] else None,
            }
            for name, c in cells.items()
        },
        "per_query": per_query,
    }
    (OUT / "eval_matrix.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Markdown
    md = []
    md.append("# Cassandra T1 + LTMi-XT — Inference & LoRA Evaluation Matrix\n")
    md.append(f"Run: 2026-05-08 · 3090 Ti · Cassandra T1 v1 (epoch 5, loss 2.2561) · LoRA r16 on q/k/v/o\n")
    md.append("## Headline\n")
    md.append("| Mode | Strict hit | Lenient hit (BPE-aware) | Avg latency | Anchor preservation |")
    md.append("|---|---:|---:|---:|---:|")
    name_pretty = {
        "baseline_unforced": "Baseline (broken model, no LTMi-XT)",
        "baseline_forced":   "Baseline + **LTMi-XT forced anchors**",
        "lora_unforced":     "LoRA-trained on LTMi-XT (no anchors)",
        "lora_forced":       "LoRA-trained + **LTMi-XT forced anchors**",
    }
    for name, c in cells.items():
        s, l = c["hits_strict"], c["hits_lenient"]
        avg_ms = mean(c["ms"]) if c["ms"] else 0
        avg_anchor = (mean(c["anchor"]) * 100) if c["anchor"] else None
        anchor_str = f"{avg_anchor:.1f}%" if avg_anchor is not None else "n/a"
        md.append(f"| {name_pretty[name]} | {s}/{n} = {s/n*100:.1f}% | {l}/{n} = {l/n*100:.1f}% | {avg_ms:.0f} ms | {anchor_str} |")
    md.append("")
    md.append("- **Strict** grader requires every expected keyword as an exact substring.")
    md.append("- **Lenient** grader strips internal whitespace from both sides — handles BPE-split tokens (`'20 48'` vs `'2048'`, `'Ro PE'` vs `'RoPE'`).")
    md.append("- **Anchor preservation** = % of locked locus tokens still present at their assigned positions in the final output. Should be ~100% by construction.\n")

    md.append("## Per-query detail\n")
    md.append("| Corpus | Query | Baseline unforced | Baseline + forced | LoRA unforced | LoRA + forced |")
    md.append("|---|---|---|---|---|---|")
    def cell(d):
        marks = []
        if d["lenient"]: marks.append("✓L")
        if d["strict"]: marks.append("✓S")
        prefix = " ".join(marks) if marks else "·"
        snippet = d["text"][:60].replace("\n", " ").replace("|", "\\|")
        return f"{prefix} `{snippet}...`"
    for r in per_query:
        md.append(f"| {r['corpus']} | {r['query'][:60]} | {cell(r['baseline_unforced'])} | {cell(r['baseline_forced'])} | {cell(r['lora_unforced'])} | {cell(r['lora_forced'])} |")

    (OUT / "eval_matrix.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {OUT/'eval_matrix.json'}")
    print(f"Wrote {OUT/'eval_matrix.md'}")


if __name__ == "__main__":
    main()
