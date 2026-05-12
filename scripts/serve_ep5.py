# Cassandra T1 — Flask serving endpoint.
#
# Environment variables (set in your container / systemd unit):
#   CASSANDRA_T1_SRC         — directory containing config.py + sophia_t1.py
#                              (default: /opt/sophiaxt/cassandra)
#   CASSANDRA_T1_CHECKPOINT  — .pt weights file
#                              (default: $CASSANDRA_T1_SRC/cassandra_ep5.pt)
#   CASSANDRA_T1_TOKENIZER   — tokenizer.json
#                              (default: $CASSANDRA_T1_SRC/tokenizer.json)
#   CASSANDRA_T1_PORT        — bind port (default: 8000)

from flask import Flask, request, jsonify
import os
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(os.environ.get("CASSANDRA_T1_SRC", "/opt/sophiaxt/cassandra"))
CHECKPOINT_PATH = Path(
    os.environ.get("CASSANDRA_T1_CHECKPOINT", str(SRC_DIR / "cassandra_ep5.pt"))
)
TOKENIZER_PATH = Path(
    os.environ.get("CASSANDRA_T1_TOKENIZER", str(SRC_DIR / "tokenizer.json"))
)
PORT = int(os.environ.get("CASSANDRA_T1_PORT", "8091"))

sys.path.insert(0, str(SRC_DIR))
from config import sophia_t1_base
from sophia_t1 import SophiaT1Model
from tokenizers import Tokenizer

app = Flask(__name__)

print(f"Loading Cassandra T1 epoch 5 from {CHECKPOINT_PATH}...", flush=True)
cfg = sophia_t1_base()
model = SophiaT1Model(cfg)
ckpt = torch.load(str(CHECKPOINT_PATH), map_location="cpu", weights_only=False)
state = {k: v.float() if v.is_floating_point() else v for k, v in ckpt["model"].items()}
model.load_state_dict(state)
model.eval()
tok = Tokenizer.from_file(str(TOKENIZER_PATH))
mask_id = tok.token_to_id("<mask>") or 4
print("Loaded! Epoch %d, loss %.4f" % (ckpt["epoch"], ckpt["loss"]), flush=True)

def generate(prompt, max_tokens=80, chunk_size=40, steps=12):
    sys_prompt = "You are Cassandra T1 by SOPHIA XT. Answer directly."
    full = "Q: " + sys_prompt + "\n\n" + prompt + "\nA:"
    enc = tok.encode(full)
    pl = len(enc.ids)
    ids_list = enc.ids + [mask_id] * max_tokens
    ids = torch.tensor([ids_list])
    pos = pl
    while pos < len(ids_list):
        cs = min(chunk_size, len(ids_list) - pos)
        end = min(pos + cs, ids.shape[1])
        placed = {}
        for step in range(steps):
            t = step / max(steps - 1, 1)
            temp = 1.0 + (0.5 - 1.0) * t
            with torch.no_grad():
                logits = model(ids)
            for p in range(pos, end):
                tid = ids[0, p].item()
                if tid != mask_id and tid in placed:
                    logits[0, :, tid] /= 1.5 * min(placed[tid], 3)
            scaled = logits[0] / temp
            sl, si = torch.sort(scaled, descending=True, dim=-1)
            cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            sl[cum - torch.softmax(sl, dim=-1) >= 0.9] = float("-inf")
            probs = torch.softmax(sl, dim=-1)
            sr = torch.multinomial(probs, 1).squeeze(-1)
            sampled = si.gather(1, sr.unsqueeze(1)).squeeze(1)
            sp = probs.gather(1, sr.unsqueeze(1)).squeeze(1)
            ism = torch.zeros(ids.shape[1], dtype=torch.bool, device=ids.device)
            for p2 in range(pos, end):
                if ids[0, p2] == mask_id:
                    ism[p2] = True
            if not ism.any():
                break
            conf = sp.clone()
            conf[~ism] = -1
            rem = ism.sum().item()
            nu = max(1, int(rem * min(1.5 / max(steps - step, 1), 0.5)))
            ti = conf.topk(min(nu, rem))[1]
            for idx in ti:
                v = sampled[idx].item()
                ids[0, idx] = v
                placed[v] = placed.get(v, 0) + 1
        pos += cs
    raw = tok.decode(ids[0, pl:].tolist())
    return raw.replace(chr(288), " ").replace(chr(266), "\n").strip()

@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": "cassandra-t1-ep5"})

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])
    prompt = messages[-1].get("content", "") if messages else ""
    start = time.time()
    out = generate(prompt, max_tokens=80, chunk_size=40, steps=12)
    return jsonify({
        "choices": [{"message": {"role": "assistant", "content": out}}],
        "model": "cassandra-t1-ep5",
        "timing": round(time.time() - start, 1)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
