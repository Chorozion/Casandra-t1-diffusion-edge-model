"""Re-grade the forced-eval log with a BPE-aware lenient grader.

The strict grader requires literal substring match. BPE tokenization
splits things like '2048' -> '20 48' and 'RoPE' -> 'Ro PE' in the answer
slot, which the strict grader treats as misses even though the answer
is right there. The lenient grader strips internal whitespace from both
sides before comparison, which restores the BPE-split tokens.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def strict_hit(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return all(kw.lower() in t for kw in keywords)

def lenient_hit(text: str, keywords: list[str]) -> bool:
    """Strip all whitespace from both sides — handles BPE-split tokens."""
    no_ws = re.sub(r"\s+", "", text.lower())
    return all(re.sub(r"\s+", "", kw.lower()) in no_ws for kw in keywords)

def regrade(which: str = "v1"):
    log_path = ROOT / "eval_out" / f"forced_eval_{which}.jsonl"
    rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"\n{'='*78}\nRE-GRADE: {which} ({len(rows)} queries)\n{'='*78}")

    metrics = {
        "unforced_strict": 0,
        "unforced_lenient": 0,
        "forced_strict": 0,
        "forced_lenient": 0,
    }
    detail = []
    for r in rows:
        ut = r["unforced"]["text"]
        ft = r["forced"]["text"]
        kws = r["expected_keywords"]
        us = strict_hit(ut, kws); ul = lenient_hit(ut, kws)
        fs = strict_hit(ft, kws); fl = lenient_hit(ft, kws)
        metrics["unforced_strict"] += int(us)
        metrics["unforced_lenient"] += int(ul)
        metrics["forced_strict"] += int(fs)
        metrics["forced_lenient"] += int(fl)
        detail.append({
            "corpus": r["corpus"],
            "query": r["query"],
            "unforced_strict": us, "unforced_lenient": ul,
            "forced_strict": fs, "forced_lenient": fl,
            "anchor_preservation": r["forced"]["anchor_preservation"],
        })

    n = len(rows)
    print(f"{'metric':<28} {'hits':<8} {'pct':<8}")
    for k, v in metrics.items():
        print(f"  {k:<26} {v}/{n}      {v/max(1,n)*100:.1f}%")

    print(f"\n  avg anchor preservation:    {sum(d['anchor_preservation'] for d in detail) / max(1, n):.1%}")

    print(f"\n=== PER-QUERY (S = strict, L = lenient) ===")
    print(f"{'corpus':<7} {'unS':<4} {'unL':<4} {'fS':<4} {'fL':<4}  query")
    for d in detail:
        print(f"{d['corpus']:<7} "
              f"{'1' if d['unforced_strict'] else '.':<4} "
              f"{'1' if d['unforced_lenient'] else '.':<4} "
              f"{'1' if d['forced_strict'] else '.':<4} "
              f"{'1' if d['forced_lenient'] else '.':<4}  "
              f"{d['query'][:70]}")

    out = {
        "checkpoint": which,
        "n_queries": n,
        "metrics": {k: {"hits": v, "pct": round(v/max(1,n)*100, 1)} for k, v in metrics.items()},
        "avg_anchor_preservation": round(sum(d['anchor_preservation'] for d in detail) / max(1, n), 4),
        "per_query": detail,
    }
    out_path = ROOT / "eval_out" / f"forced_eval_{which}_regraded.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")

if __name__ == "__main__":
    regrade(sys.argv[1] if len(sys.argv) > 1 else "v1")
