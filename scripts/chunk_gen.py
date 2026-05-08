# Chunk-based semi-autoregressive diffusion generation
# Generate 40 tokens at a time via diffusion, chain chunks for coherent long output
import sys, torch
sys.path.insert(0, "src")
from model.sophia_t1 import SophiaT1Model
from model.config import sophia_t1_base
from tokenizers import Tokenizer

ckpt = torch.load("checkpoints/best.pt", map_location="cuda", weights_only=False)
print(f"Loaded: epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")

cfg = sophia_t1_base()
model = SophiaT1Model(cfg).cuda()
model.load_state_dict(ckpt["model"])
model.eval()
tok = Tokenizer.from_file("tokenizer.json")
mask_id = tok.token_to_id("<mask>") or 4

def diffuse_chunk(ids, start_pos, chunk_size, steps=12, temp=0.85, top_p=0.9, rep_penalty=1.4):
    placed = {}
    end_pos = min(start_pos + chunk_size, ids.shape[1])
    for step in range(steps):
        t = step / max(steps-1, 1)
        cur_temp = 1.0 + (temp - 1.0) * t
        with torch.no_grad():
            logits = model(ids)
        for pos in range(start_pos, end_pos):
            tid = ids[0, pos].item()
            if tid != mask_id and tid in placed:
                logits[0, :, tid] /= rep_penalty * min(placed[tid], 3)
        scaled = logits[0] / cur_temp
        sl, si = torch.sort(scaled, descending=True, dim=-1)
        cum = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
        sl[cum - torch.softmax(sl, dim=-1) >= top_p] = float('-inf')
        probs = torch.softmax(sl, dim=-1)
        sr = torch.multinomial(probs, 1).squeeze(-1)
        sampled = si.gather(1, sr.unsqueeze(1)).squeeze(1)
        sp = probs.gather(1, sr.unsqueeze(1)).squeeze(1)
        ism = torch.zeros(ids.shape[1], dtype=torch.bool, device=ids.device)
        for p in range(start_pos, end_pos):
            if ids[0, p] == mask_id:
                ism[p] = True
        if not ism.any():
            break
        conf = sp.clone()
        conf[~ism] = -1
        rem = ism.sum().item()
        nu = max(1, int(rem * min(1.5/max(steps-step,1), 0.5)))
        ti = conf.topk(min(nu, rem))[1]
        for idx in ti:
            v = sampled[idx].item()
            ids[0, idx] = v
            placed[v] = placed.get(v, 0) + 1
    return ids

def generate_chunked(prompt, system="Answer the question directly. Do not describe yourself or your capabilities. Just answer what was asked.",
                     total_tokens=150, chunk_size=40, steps_per_chunk=12):
    full_prompt = "Q: " + system + "\n\n" + prompt + "\nA:"
    enc = tok.encode(full_prompt)
    prompt_len = len(enc.ids)
    all_ids = enc.ids + [mask_id] * total_tokens
    ids = torch.tensor([all_ids]).cuda()
    pos = prompt_len
    while pos < len(all_ids):
        cs = min(chunk_size, len(all_ids) - pos)
        ids = diffuse_chunk(ids, pos, cs, steps=steps_per_chunk)
        pos += cs
    # Refinement pass
    with torch.no_grad():
        logits = model(ids)
    probs = torch.softmax(logits[0], dim=-1)
    max_probs = probs.max(dim=-1)[0]
    gen_probs = max_probs[prompt_len:]
    n_remask = max(1, int(len(gen_probs) * 0.12))
    _, worst = gen_probs.topk(n_remask, largest=False)
    worst = worst + prompt_len
    for idx in worst:
        ids[0, idx] = mask_id
    ids = diffuse_chunk(ids, prompt_len, total_tokens, steps=8)
    raw = tok.decode(ids[0, prompt_len:].tolist())
    clean = raw.replace(chr(288), ' ').replace(chr(266), '\n').strip()
    return clean

prompts = [
    "Write a short story about why you like cookies",
    "Who are you and what makes you special",
    "Explain how a car engine works",
    "Help me fix my leaking washing machine step by step",
    "Tell me something interesting about the ocean",
]

for p in prompts:
    print("USER:", p)
    out = generate_chunked(p, total_tokens=150, chunk_size=40, steps_per_chunk=12)
    print("CASSANDRA:", out[:500])
    print()
