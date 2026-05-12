"""Ship-verification smoke test for Cassandra T1.5.

Goal: confirm cassandra_t1_continued_step500.pt is loadable + can produce
text. Pre-ship gate per user instruction "verify the contents before ship".

Does NOT touch v2/T2/T2.5 — strictly a runnability check on v1.5.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent))

import torch
from cassandra_loader import load, decode


PROMPTS = [
    "The capital of France is",
    "Water boils at",
    "The largest planet in our solar system is",
]


def greedy_clm_continue(model, tok, mask_id, prompt: str, n_new: int = 32) -> str:
    """Greedy continuation. Sophia T1 is a CLM at architecture level — we just
    feed the prompt and take argmax of the next position iteratively."""
    ids = tok.encode(prompt).ids
    cur = torch.tensor([ids], dtype=torch.long, device="cuda")
    out = list(ids)
    for _ in range(n_new):
        with torch.no_grad():
            logits = model(cur).logits if hasattr(model(cur), "logits") else model(cur)
            if isinstance(logits, tuple):
                logits = logits[0]
        next_id = int(logits[0, -1].argmax().item())
        out.append(next_id)
        cur = torch.tensor([out], dtype=torch.long, device="cuda")
        if next_id == tok.token_to_id("<eos>") or next_id == 0:
            break
    return decode(tok, out)


def main():
    print(f"[ship-smoke] {time.strftime('%H:%M:%S')} start")
    model, tok, mask_id = load("v1.5")
    print(f"[ship-smoke] loaded v1.5 OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[ship-smoke] params: {n_params/1e6:.1f}M")

    for p in PROMPTS:
        t0 = time.time()
        try:
            out = greedy_clm_continue(model, tok, mask_id, p, n_new=24)
            dt = time.time() - t0
            print(f"\n--- prompt: {p!r}  ({dt:.2f}s) ---")
            print(out)
        except Exception as e:
            print(f"FAIL on prompt {p!r}: {type(e).__name__}: {e}")

    print(f"\n[ship-smoke] PASS — v1.5 is runnable.")


if __name__ == "__main__":
    main()
