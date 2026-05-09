"""Run forced vs unforced decoding on v1 across the 15 LTMi-XT eval queries.

For each query:
  1. Retrieve top-1 locus from the matching .ltmi bundle (C1/C2/C3).
  2. Tokenize the locus statement.
  3. Generate twice:
       (a) Unforced — normal diffusion unmasking (broken-model baseline)
       (b) Forced  — locus tokens locked at the START of answer slot,
                     diffusion unmasks the remaining positions only.
  4. Score: did the expected_keywords appear anywhere in the output?

Writes per-query JSONL log + a markdown summary table.
"""
from __future__ import annotations
import sys, json, time, os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import torch
from cassandra_loader import load, decode
from forced_decode import load_bundle, retrieve_top1, generate_with_anchors

QUERIES_PATH = ROOT / "queries.json"
BUNDLE_PATHS = {
    "C1": ROOT / "C1.json",
    "C2": ROOT / "C2.json",
    "C3": ROOT / "C3.json",
}
OUT_DIR = ROOT / "eval_out"
OUT_DIR.mkdir(exist_ok=True)

SYS = "You are Cassandra T1, a diffusion language model by SOPHIA XT. Direct, helpful, honest."
ANSWER_LEN = 96  # tokens to generate
SEED = 1234


def hit(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return all(kw.lower() in t for kw in keywords)


def main(which: str = "v1"):
    print(f"\n{'='*72}\nFORCED-DECODE EVAL: {which}\n{'='*72}")
    torch.manual_seed(SEED)

    model, tok, _ = load(which)
    bundles = {cid: load_bundle(p) for cid, p in BUNDLE_PATHS.items()}
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    log_path = OUT_DIR / f"forced_eval_{which}.jsonl"
    log_f = log_path.open("w", encoding="utf-8")
    rows: list[dict] = []

    for cid, qlist in queries.items():
        loci = bundles[cid]
        for q in qlist:
            query = q["q"]
            expected = q["expect_keywords"]
            print(f"\n--- {cid}: {query}")

            # Retrieve top-1 locus
            locus = retrieve_top1(loci, query)
            locus_stmt = locus["statement"] if locus else "(no locus)"
            print(f"  retrieved: {locus_stmt}")

            # Build prompt
            full_prompt = f"Q: {SYS}\n\n{query}\nA:"
            prompt_ids = torch.tensor([tok.encode(full_prompt).ids], device="cuda")

            # ─── (a) Unforced ─────────────────────────────────────
            torch.manual_seed(SEED)
            t0 = time.time()
            unforced_out, _ = generate_with_anchors(
                model, prompt_ids, answer_len=ANSWER_LEN, locked_positions=None,
                num_steps=12, temperature=0.8, top_p=0.9, beta=0.5, rep_penalty=1.3,
            )
            unforced_text = decode(tok, unforced_out[0]).strip()
            unforced_dt = time.time() - t0
            unforced_hit = hit(unforced_text, expected)

            # ─── (b) Forced ───────────────────────────────────────
            # Lock locus tokens at positions [0..L) of answer slot.
            # Leading space matters for BPE — most tokenizers want a space prefix.
            locus_tokens = tok.encode(" " + locus_stmt).ids
            # Truncate locus tokens to leave at least 8 free positions for glue
            max_anchor_len = ANSWER_LEN - 8
            anchor_tokens = locus_tokens[:max_anchor_len]
            locked = {i: t for i, t in enumerate(anchor_tokens)}

            torch.manual_seed(SEED)
            t0 = time.time()
            forced_out, init_mask = generate_with_anchors(
                model, prompt_ids, answer_len=ANSWER_LEN, locked_positions=locked,
                num_steps=12, temperature=0.8, top_p=0.9, beta=0.5, rep_penalty=1.3,
            )
            forced_text = decode(tok, forced_out[0]).strip()
            forced_dt = time.time() - t0
            forced_hit = hit(forced_text, expected)

            # Anchor preservation = % of anchor tokens still at locked positions
            anchor_preserved = sum(
                1 for i, t in enumerate(anchor_tokens) if forced_out[0, i].item() == t
            ) / max(1, len(anchor_tokens))

            row = {
                "corpus": cid,
                "query": query,
                "expected_keywords": expected,
                "retrieved_statement": locus_stmt,
                "unforced": {
                    "text": unforced_text,
                    "hit": unforced_hit,
                    "ms": int(unforced_dt * 1000),
                },
                "forced": {
                    "text": forced_text,
                    "hit": forced_hit,
                    "ms": int(forced_dt * 1000),
                    "anchor_token_count": len(anchor_tokens),
                    "anchor_preservation": round(anchor_preserved, 3),
                },
            }
            rows.append(row)
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_f.flush()

            print(f"  [unforced @{unforced_dt:.1f}s, hit={unforced_hit}] {unforced_text[:120]!r}")
            print(f"  [forced   @{forced_dt:.1f}s, hit={forced_hit}, anchor={anchor_preserved:.0%}] {forced_text[:120]!r}")

    log_f.close()

    # Summary
    n = len(rows)
    unforced_hits = sum(1 for r in rows if r["unforced"]["hit"])
    forced_hits = sum(1 for r in rows if r["forced"]["hit"])
    avg_anchor = sum(r["forced"]["anchor_preservation"] for r in rows) / max(1, n)

    print(f"\n{'='*72}\nSUMMARY ({which})\n{'='*72}")
    print(f"queries:                    {n}")
    print(f"unforced top-1 hit:         {unforced_hits}/{n} = {unforced_hits/max(1,n)*100:.1f}%")
    print(f"forced   top-1 hit:         {forced_hits}/{n} = {forced_hits/max(1,n)*100:.1f}%")
    print(f"avg anchor preservation:    {avg_anchor:.1%}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "v1"
    main(which)
