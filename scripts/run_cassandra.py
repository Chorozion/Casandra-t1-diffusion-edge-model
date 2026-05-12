# Cassandra T1 — local inference REPL (Q3-enhanced quantile unmasking).
#
# Quickstart:
#     export CASSANDRA_T1_ROOT=/path/to/cassandra-t1            # repo root
#     export CASSANDRA_T1_CHECKPOINT=/path/to/cassandra_ep5_fp16.pt
#     # (optional) export CASSANDRA_T1_TOKENIZER=/path/to/tokenizer.json
#     python scripts/run_cassandra.py
#
# Default behavior (no env vars):
#   - CASSANDRA_T1_ROOT defaults to the parent of this script.
#   - CASSANDRA_T1_CHECKPOINT defaults to release/cassandra_ep5_fp16.pt
#     (LFS-hosted; download separately).
#   - CASSANDRA_T1_TOKENIZER defaults to release/tokenizer.json.
#
# Works on the public ep5 checkpoint. The Q3 generator improves output
# quality without re-training (AdaLN zero-init → no behavior change
# until you actually fine-tune the new parameters).

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch


# ─── Path resolution via env vars (no hardcoded user paths) ──────────────

DEFAULT_ROOT = Path(__file__).resolve().parent.parent
ROOT = Path(os.environ.get("CASSANDRA_T1_ROOT", str(DEFAULT_ROOT)))

# Source tree on sys.path so the `model.*` imports resolve
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

CHECKPOINT_PATH = Path(
    os.environ.get(
        "CASSANDRA_T1_CHECKPOINT",
        str(ROOT / "release" / "cassandra_ep5_fp16.pt"),
    )
)
TOKENIZER_PATH = Path(
    os.environ.get(
        "CASSANDRA_T1_TOKENIZER",
        str(ROOT / "release" / "tokenizer.json"),
    )
)
USE_Q3 = os.environ.get("CASSANDRA_T1_USE_Q3", "1") != "0"


def _check_path(path: Path, label: str) -> None:
    if not path.exists():
        sys.stderr.write(
            f"[run_cassandra] {label} not found at: {path}\n"
            f"  Set the corresponding env var to point at your local copy:\n"
            f"    CASSANDRA_T1_ROOT          (repo root, currently {ROOT})\n"
            f"    CASSANDRA_T1_CHECKPOINT    (.pt weights)\n"
            f"    CASSANDRA_T1_TOKENIZER     (tokenizer.json)\n"
            f"  Weights are distributed via Git LFS or the release section\n"
            f"  — see release/README.md for the download URL.\n"
        )
        sys.exit(1)


_check_path(SRC_DIR, "source tree")
_check_path(CHECKPOINT_PATH, "checkpoint")
_check_path(TOKENIZER_PATH, "tokenizer")


from model.sophia_t1 import SophiaT1Model
from model.config import sophia_t1_base, sophia_t1_base_q3
from tokenizers import Tokenizer


# ─── Load model + tokenizer ──────────────────────────────────────────────

print(f"Loading Cassandra T1 from {CHECKPOINT_PATH.name}...")
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cuda", weights_only=False)
cfg = sophia_t1_base_q3() if USE_Q3 else sophia_t1_base()
model = SophiaT1Model(cfg).cuda()
state = SophiaT1Model.remap_baseline_state_dict(ckpt["model"], use_adaln=cfg.use_adaln)
state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
result = model.load_state_dict(state, strict=False)
print(f"  loaded epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
if result.missing_keys:
    print(f"  {len(result.missing_keys)} Q3 additions zero-initialized (normal for ep5)")
model.eval()

tok = Tokenizer.from_file(str(TOKENIZER_PATH))
mask_id = tok.token_to_id("<mask>") or 4
print("Ready. Type a message and press Enter. 'quit' to exit.\n")

SYS = "You are Cassandra T1, a diffusion language model by SOPHIA XT. Direct, helpful, honest."


def generate(prompt: str, max_tokens: int = 100, num_steps: int = 12) -> str:
    full = f"Q: {SYS}\n\n{prompt}\nA:"
    ids = torch.tensor([tok.encode(full).ids]).cuda()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_tokens,
            num_steps=num_steps,
            temperature=0.8,
            top_p=0.9,
            beta=0.5,
            rep_penalty=1.3,
            timestep_power=1.5,
        )
    raw = tok.decode(out[0].tolist())
    return raw.replace(chr(288), " ").replace(chr(266), "\n").strip()


# ─── REPL loop ───────────────────────────────────────────────────────────

while True:
    try:
        prompt = input("YOU: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue
        response = generate(prompt)
        print(f"\nCASSANDRA: {response}\n")
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("Goodbye!")
