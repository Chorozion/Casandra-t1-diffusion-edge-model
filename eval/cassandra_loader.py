"""Load Cassandra T1 (v1 ep5 or v2 ep2-step82k) onto CUDA.

Both checkpoints share the same architecture (sophia_t1_base). v2 includes
optimizer state alongside the model weights — we discard the optimizer.
"""
from __future__ import annotations
import sys, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cassandra_src"))

import torch
from model.sophia_t1 import SophiaT1Model
from model.config import sophia_t1_base
from tokenizers import Tokenizer

WEIGHTS = ROOT / "weights"
TOK_PATH = ROOT / "tokenizer.json"

CHECKPOINTS = {
    "v1": WEIGHTS / "cassandra_ep5_fp16.pt",
    "v2": WEIGHTS / "v2_scratch_epoch2_82002.pt",
}


def load(which: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
    assert which in CHECKPOINTS, f"unknown checkpoint: {which}"
    path = CHECKPOINTS[which]
    print(f"[load] reading {path.name} ({path.stat().st_size / 1e9:.2f} GB) -> CPU first")
    t0 = time.time()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    print(f"[load] read in {time.time()-t0:.1f}s, keys: {list(ckpt.keys())[:6]}")

    cfg = sophia_t1_base()
    print(f"[load] building model: {cfg.num_params_millions:.0f}M params target")
    model = SophiaT1Model(cfg)

    state = ckpt.get("model", ckpt)
    state = {k: (v.to(dtype) if v.is_floating_point() else v) for k, v in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        print(f"[load] {len(result.missing_keys)} missing keys (first 3): {result.missing_keys[:3]}")
    if result.unexpected_keys:
        print(f"[load] {len(result.unexpected_keys)} unexpected keys (first 3): {result.unexpected_keys[:3]}")

    if which == "v1":
        epoch = ckpt.get("epoch", "?")
        loss = ckpt.get("loss", float("nan"))
        print(f"[load] v1: epoch={epoch}, loss={loss:.4f}" if isinstance(loss, float) else f"[load] v1 ckpt: epoch={epoch}")
    else:
        epoch = ckpt.get("epoch", "?")
        step = ckpt.get("step", ckpt.get("global_step", "?"))
        loss = ckpt.get("loss", ckpt.get("train_loss", float("nan")))
        print(f"[load] v2: epoch={epoch}, step={step}, loss={loss}")

    print(f"[load] -> {device} ({dtype})")
    model = model.to(device=device, dtype=dtype)
    model.eval()
    print(f"[load] model on {device}; cuda_mem_alloc={torch.cuda.memory_allocated()/1e9:.2f} GB")

    tok = Tokenizer.from_file(str(TOK_PATH))
    mask_id = tok.token_to_id("<mask>") or cfg.mask_token_id
    print(f"[load] tokenizer ready, mask_id={mask_id}, vocab={tok.get_vocab_size()}")

    # Free CPU copy
    del ckpt, state
    import gc; gc.collect()
    torch.cuda.empty_cache()

    return model, tok, mask_id


def decode(tok: Tokenizer, ids) -> str:
    """Cassandra tokens use chr(288) for space and chr(266) for newline."""
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    raw = tok.decode(ids)
    return raw.replace(chr(288), " ").replace(chr(266), "\n")
