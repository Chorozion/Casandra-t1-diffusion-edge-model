"""Eval v1 + LoRA adapter on the same 15 queries, in two modes:
  (a) LoRA-only unforced — does fine-tune alone make outputs coherent?
  (b) LoRA + forced      — does forcing still work after training?

Together with run_forced_eval.py output, this gives the full 4-cell matrix:
    baseline        | LTMi-XT-forced
    LoRA-on-LTMi-XT | LoRA + LTMi-XT-forced
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import torch
from cassandra_loader import load, decode
from forced_decode import load_bundle, retrieve_top1, generate_with_anchors
from lora_finetune import wrap_lora, LoRALinear

ADAPTER_PATH = ROOT / "lora_adapters" / "v1_ltmi_r16.pt"
QUERIES_PATH = ROOT / "queries.json"
BUNDLE_PATHS = {cid: ROOT / f"{cid}.json" for cid in ("C1", "C2", "C3")}
OUT_DIR = ROOT / "eval_out"
OUT_DIR.mkdir(exist_ok=True)

SYS = "You are Cassandra T1, a diffusion language model by SOPHIA XT. Direct, helpful, honest."
ANSWER_LEN = 96
SEED = 1234


def hit(text: str, keywords: list[str]) -> bool:
    t = text.lower()
    return all(kw.lower() in t for kw in keywords)


def load_lora_into(model, adapter_path: Path):
    """Wrap base layers with LoRA structure, then load A/B weights."""
    # Freeze base, wrap
    for p in model.parameters():
        p.requires_grad = False
    n = wrap_lora(model)
    # Move LoRA modules to GPU/fp32
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            mod.A = mod.A.to(device="cuda", dtype=torch.float32)
            mod.B = mod.B.to(device="cuda", dtype=torch.float32)
    # Load adapter weights
    payload = torch.load(adapter_path, map_location="cuda", weights_only=False)
    state = payload["lora_state"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if ".A." in k or ".B." in k]
    print(f"[lora] wrapped {n} layers; loaded {len(state)} adapter weights")
    if real_missing:
        print(f"[lora] WARN: {len(real_missing)} adapter weights missing")
    if unexpected:
        print(f"[lora] {len(unexpected)} unexpected (likely base, fine)")
    print(f"[lora] training meta: {payload['training']}")


def main():
    print(f"\n{'='*72}\nLoRA EVAL: v1 + adapter\n{'='*72}")
    torch.manual_seed(SEED)

    model, tok, _ = load("v1")
    load_lora_into(model, ADAPTER_PATH)
    model.eval()

    bundles = {cid: load_bundle(p) for cid, p in BUNDLE_PATHS.items()}
    queries = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))

    log_path = OUT_DIR / "lora_eval_v1.jsonl"
    log_f = log_path.open("w", encoding="utf-8")
    rows: list[dict] = []

    for cid, qlist in queries.items():
        for q in qlist:
            query = q["q"]
            expected = q["expect_keywords"]
            print(f"\n--- {cid}: {query}")

            locus = retrieve_top1(bundles[cid], query)
            locus_stmt = locus["statement"] if locus else "(no locus)"

            full_prompt = f"Q: {SYS}\n\n{query}\nA:"
            prompt_ids = torch.tensor([tok.encode(full_prompt).ids], device="cuda")

            # ─── (a) LoRA-only unforced ──────────────────────────
            torch.manual_seed(SEED)
            t0 = time.time()
            unforced_out, _ = generate_with_anchors(
                model, prompt_ids, answer_len=ANSWER_LEN, locked_positions=None,
                num_steps=12, temperature=0.8, top_p=0.9, beta=0.5, rep_penalty=1.3,
            )
            unforced_text = decode(tok, unforced_out[0]).strip()
            unforced_dt = time.time() - t0
            unforced_hit = hit(unforced_text, expected)

            # ─── (b) LoRA + forced ────────────────────────────────
            locus_tokens = tok.encode(" " + locus_stmt).ids
            anchor_tokens = locus_tokens[: ANSWER_LEN - 8]
            locked = {i: t for i, t in enumerate(anchor_tokens)}

            torch.manual_seed(SEED)
            t0 = time.time()
            forced_out, _ = generate_with_anchors(
                model, prompt_ids, answer_len=ANSWER_LEN, locked_positions=locked,
                num_steps=12, temperature=0.8, top_p=0.9, beta=0.5, rep_penalty=1.3,
            )
            forced_text = decode(tok, forced_out[0]).strip()
            forced_dt = time.time() - t0
            forced_hit = hit(forced_text, expected)

            anchor_preserved = sum(
                1 for i, t in enumerate(anchor_tokens) if forced_out[0, i].item() == t
            ) / max(1, len(anchor_tokens))

            row = {
                "corpus": cid,
                "query": query,
                "expected_keywords": expected,
                "retrieved_statement": locus_stmt,
                "lora_unforced": {
                    "text": unforced_text, "hit": unforced_hit, "ms": int(unforced_dt * 1000),
                },
                "lora_forced": {
                    "text": forced_text, "hit": forced_hit, "ms": int(forced_dt * 1000),
                    "anchor_token_count": len(anchor_tokens),
                    "anchor_preservation": round(anchor_preserved, 3),
                },
            }
            rows.append(row)
            log_f.write(json.dumps(row, ensure_ascii=False) + "\n"); log_f.flush()

            print(f"  [LoRA unforced @{unforced_dt:.1f}s, hit={unforced_hit}] {unforced_text[:120]!r}")
            print(f"  [LoRA forced   @{forced_dt:.1f}s, hit={forced_hit}, anchor={anchor_preserved:.0%}] {forced_text[:120]!r}")

    log_f.close()

    n = len(rows)
    uh = sum(1 for r in rows if r["lora_unforced"]["hit"])
    fh = sum(1 for r in rows if r["lora_forced"]["hit"])
    avg_anchor = sum(r["lora_forced"]["anchor_preservation"] for r in rows) / max(1, n)

    print(f"\n{'='*72}\nSUMMARY (v1 + LoRA)\n{'='*72}")
    print(f"queries:                 {n}")
    print(f"LoRA-unforced strict:    {uh}/{n} = {uh/max(1,n)*100:.1f}%")
    print(f"LoRA-forced strict:      {fh}/{n} = {fh/max(1,n)*100:.1f}%")
    print(f"avg anchor preservation: {avg_anchor:.1%}")
    print(f"log: {log_path}")


if __name__ == "__main__":
    main()
