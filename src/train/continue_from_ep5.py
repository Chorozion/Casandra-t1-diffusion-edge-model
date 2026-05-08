"""
Cassandra T1 — Continuation training from epoch 5 on CLEANED + EXPANDED corpus.

Q3-enhanced (vs. prior version):
  1. Loads ep5 via remap_baseline_state_dict → Q3-enhanced model (AdaLN zero-init)
  2. Passes per-sample diffusion timestep t to model forward
  3. Uses model.compute_diffusion_loss (includes span masking + curriculum)
  4. Q3 PGM learning rate schedule with exponential smoothing
  5. Category-weighted loss (identity 3x, appliance 2x)
  6. Canary eval every 500 steps — auto-abort on persistent failures
  7. Z-loss regularization

Designed to run on RunPod A100 SXM 80GB. Also runnable locally on 3090 for smoke tests (very slow).
"""
import os, sys, json, random, time, math
os.chdir("/workspace/cassandra" if os.path.exists("/workspace/cassandra") else "I:/sophiat1")
sys.path.insert(0, "src")

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model.config import sophia_t1_base_q3
from model.sophia_t1 import SophiaT1Model

# ===== CONFIG =====
CKPT_IN        = "checkpoints/cassandra_ep5_fp16.pt"
DATA_PATH      = "data/train_v2.jsonl"
TOKENIZER      = "tokenizer.json"
CKPT_DIR       = "checkpoints"
CANARY_LOG     = f"{CKPT_DIR}/canary_continuation.jsonl"

BS             = 8            # down from 16 — Q3 additions (AdaLN + conf head) use more VRAM
SEQ            = 1024
ADDITIONAL_EP  = 5            # epochs 6 → 10
GA             = 4            # up from 2 → effective batch 32 (better than before)
PEAK_LR        = 5e-5        # fine-tune LR
MIN_LR         = 5e-6

# Q3 PGM hyperparameters (Q3 Appendix A.2.1–A.2.5)
PGM_ALPHA      = 0.8         # sub-linear rise exponent
PGM_BETA       = 0.01        # exponential smoothing coefficient
PGM_GAMMA      = 0.2         # decay shape
PGM_DELTA      = 0.5         # decay speed (0.5 → O(1/T) convex)
WARMUP_FRAC    = 0.03

LAMBDA_IDENTITY  = 3.0
LAMBDA_APPLIANCE = 2.0
Z_LOSS_ALPHA     = 0.001

CANARY_EVERY   = 500
MAX_CANARY_FAIL = 3

print("=" * 72)
print("  CASSANDRA T1 — CONTINUATION TRAINING from EPOCH 5  (Q3-enhanced)")
print("  AdaLN + span mask + curriculum + PGM LR + category weighting + canary")
print("=" * 72, flush=True)

# ===== Load checkpoint + remap for Q3 model =====
print(f"\nLoading: {CKPT_IN}", flush=True)
ckpt = torch.load(CKPT_IN, map_location="cpu", weights_only=False)
start_epoch = ckpt.get("epoch", 5)
prev_loss   = ckpt.get("loss", float("inf"))
print(f"  Source: epoch {start_epoch}, loss {prev_loss:.4f}", flush=True)

cfg   = sophia_t1_base_q3()
model = SophiaT1Model(cfg).cuda()
state = SophiaT1Model.remap_baseline_state_dict(ckpt["model"], use_adaln=cfg.use_adaln)
state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
result = model.load_state_dict(state, strict=False)
print(f"  Loaded {len(state)} tensors; {len(result.missing_keys)} Q3 additions stay zero-init")
del ckpt, state
torch.cuda.empty_cache()

tok = Tokenizer.from_file(TOKENIZER)
mask_id = tok.token_to_id("<mask>") or 4
print(f"  Tokenizer: {tok.get_vocab_size()} tokens | mask_id={mask_id}", flush=True)

# ===== Dataset =====
class DS(Dataset):
    def __init__(self, path, tok, ml=1024):
        self.samples, self.tok, self.ml = [], tok, ml
        cat_counts = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                try: r = json.loads(line)
                except: continue
                t = r.get("t", "")
                c = r.get("c", "?")
                if len(t) > 30:
                    self.samples.append((t, c))
                    cat_counts[c] = cat_counts.get(c, 0) + 1
        print(f"  Dataset: {len(self.samples):,} samples", flush=True)
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {c:20s} {n:>8,}", flush=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        text, cat = self.samples[i]
        ids = self.tok.encode(text).ids[:self.ml]
        pad = self.ml - len(ids)
        if cat == "identity":          cat_w = LAMBDA_IDENTITY
        elif cat == "appliance_repair": cat_w = LAMBDA_APPLIANCE
        else:                           cat_w = 1.0
        return {
            "input_ids":      torch.tensor(ids + [0]*pad, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(ids) + [0]*pad, dtype=torch.long),
            "cat_weight":     torch.tensor(cat_w, dtype=torch.float32),
        }

ds = DS(DATA_PATH, tok, SEQ)
dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

# ===== Q3 PGM learning rate schedule =====
class Q3PGMSchedule:
    """
    Q3 Parabolic Gradient Modifier (Appendix A.2.1–A.2.5):
      Rise (s ≤ S_rise):
        η(s) = η_min + (η_max-η_min) * (s/S_rise)^α * (1 - exp(-β*s))
      Decay (s > S_rise):
        η(s) = η_max * (1 - ((s-S_rise)/(S-S_rise))^γ)^δ
    With α=0.8 the rise is sub-linear; β=0.01 smooths the early spike.
    With γ=0.2, δ=0.5 the decay is convex O(1/T).
    """
    def __init__(self, opt, total_steps, warmup_frac, peak, floor,
                 alpha=0.8, beta=0.01, gamma=0.2, delta=0.5):
        self.opt = opt
        self.total = total_steps
        self.warmup = max(1, int(total_steps * warmup_frac))
        self.peak = peak
        self.floor = floor
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.step_count = 0

    def step(self):
        self.step_count += 1
        s = self.step_count
        if s <= self.warmup:
            frac = s / self.warmup
            lr = self.floor + (self.peak - self.floor) * (frac ** self.alpha) * (1 - math.exp(-self.beta * s))
        else:
            decay_frac = (s - self.warmup) / max(self.total - self.warmup, 1)
            decay_frac = min(decay_frac, 1.0)
            lr = self.floor + (self.peak - self.floor) * (1 - decay_frac ** self.gamma) ** self.delta
        for pg in self.opt.param_groups: pg["lr"] = lr
        return lr

opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
total_steps = len(dl) * ADDITIONAL_EP // GA
sched  = Q3PGMSchedule(opt, total_steps, WARMUP_FRAC, PEAK_LR, MIN_LR,
                       alpha=PGM_ALPHA, beta=PGM_BETA, gamma=PGM_GAMMA, delta=PGM_DELTA)
scaler = torch.amp.GradScaler("cuda")

print(f"\n  Epochs: {start_epoch+1} → {start_epoch+ADDITIONAL_EP}")
print(f"  Steps: {total_steps:,}  BS: {BS}×{GA}  SEQ: {SEQ}")
print(f"  Q3 PGM: peak={PEAK_LR:.0e} floor={MIN_LR:.0e} α={PGM_ALPHA} β={PGM_BETA} γ={PGM_GAMMA} δ={PGM_DELTA}")
print(f"  Category weights: identity={LAMBDA_IDENTITY}×  appliance={LAMBDA_APPLIANCE}×")
print(f"  Z-loss α={Z_LOSS_ALPHA}  Canary every {CANARY_EVERY} steps", flush=True)

# ===== Canary =====
try:
    from eval.canary_identity import run_canary
except Exception as e:
    print(f"  WARN: canary import failed ({e}) — disabling canary")
    run_canary = None

def canary_gen(prompt, max_tokens=80):
    model.eval()
    with torch.no_grad():
        full = f"Q: {prompt}\nA:"
        ids = torch.tensor([tok.encode(full).ids]).cuda()
        out = model.generate(ids, max_new_tokens=max_tokens, num_steps=10,
                             temperature=0.8, top_p=0.9, beta=0.5, rep_penalty=1.3)
        txt = tok.decode(out[0].tolist())
    model.train()
    return txt.replace(chr(288), " ").replace(chr(266), "\n").strip()

# ===== Training loop =====
model.train(); opt.zero_grad()
gs = 0
best = prev_loss
t0 = time.time()
canary_fail_streak = 0

for ep in range(ADDITIONAL_EP):
    epoch_num = start_epoch + ep + 1
    el, es = 0.0, 0
    for bi, batch in enumerate(dl):
        ids  = batch["input_ids"].cuda()
        msk  = batch["attention_mask"].cuda()
        cw   = batch["cat_weight"].cuda()

        # Training fraction for curriculum
        step_frac = gs / max(total_steps, 1)

        with torch.amp.autocast("cuda"):
            # Randomize mask ratio (curriculum-bounded inside compute_diffusion_loss)
            # Use Beta(2,2) scaled into (r_lo, r_hi) from config
            r_lo, r_hi = cfg.mask_ratio_min, cfg.mask_ratio_max
            if step_frac < 0.1:
                r_hi = min(r_hi, 0.6)
            mr = r_lo + (r_hi - r_lo) * random.betavariate(2.0, 2.0)

            # Build mask + timestep + forward inline (for category weighting)
            B, S = ids.shape
            rm = torch.rand(B, S, device=ids.device) < mr
            # Mix in span masking for span_mask_prob of batch
            span_prob = cfg.span_mask_prob
            if random.random() < span_prob:
                # Replace first ~half with span pattern for diversity
                span_len = max(1, int(random.gauss(cfg.span_mean_length, 1.0)))
                for b in range(B // 2):
                    start = random.randint(0, max(1, S - span_len - 1))
                    rm[b, start:start + span_len] = True
            rm = rm & msk.bool()
            tgt = ids.clone()
            masked_ids = ids.clone()
            masked_ids[rm] = mask_id

            # Timestep = fraction masked per sample
            gamma_t = rm.float().mean(dim=-1).clamp(0.05, 0.95)

            logits = model(masked_ids, msk, t=gamma_t)

            fl = logits[rm]
            ft = tgt[rm]
            if fl.numel() > 0:
                per_tok = F.cross_entropy(fl, ft, reduction="none")

                # Spatial upweight (tokens ≥ 32000)
                spatial_w = 1.0 + (ft >= 32000).float() * 1.0

                # Per-sample category weight broadcast to masked positions
                sample_idx = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, S)[rm]
                sample_w = cw[sample_idx]

                weighted = per_tok * spatial_w * sample_w
                loss = weighted.mean() / GA

                # Z-loss: penalize logit magnitude
                if Z_LOSS_ALPHA > 0:
                    z = torch.logsumexp(logits, dim=-1)
                    loss = loss + Z_LOSS_ALPHA * (z ** 2).mean() / GA
            else:
                loss = torch.tensor(0.0, device=ids.device, requires_grad=True)

        scaler.scale(loss).backward()
        el += loss.item() * GA
        es += 1

        if (bi + 1) % GA == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            cur_lr = sched.step()
            opt.zero_grad()
            gs += 1

            if gs % 50 == 0:
                avg = el / es
                sps = gs / (time.time() - t0)
                eta = (total_steps - gs) / max(sps, 1e-6) / 60
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"  step {gs:>6d}/{total_steps} ep{epoch_num} | loss {avg:.4f} | lr {cur_lr:.2e} | mr {mr:.2f} | {sps:.2f}s/s | ETA {eta:.0f}m | {vram:.1f}GB", flush=True)

            if run_canary is not None and gs % CANARY_EVERY == 0:
                res = run_canary(model, tok, canary_gen, step=f"ep{epoch_num}-s{gs}", log_path=CANARY_LOG)
                if res["fail_rate"] > 0.5:
                    canary_fail_streak += 1
                    print(f"  *** CANARY FAIL streak={canary_fail_streak} ***", flush=True)
                    if canary_fail_streak >= MAX_CANARY_FAIL:
                        print(f"  ABORTING: {MAX_CANARY_FAIL} consecutive canary failures", flush=True)
                        torch.save({"model": model.state_dict(), "config": cfg, "loss": el/max(es,1), "epoch": epoch_num},
                                   f"{CKPT_DIR}/aborted_ep{epoch_num}_s{gs}.pt")
                        sys.exit(1)
                else:
                    canary_fail_streak = 0

    avg_ep = el / max(es, 1)
    print(f"\n  Epoch {epoch_num} | loss: {avg_ep:.4f}", flush=True)

    # Save every epoch + FP16 for VPS
    torch.save({"model": model.state_dict(), "config": cfg, "loss": avg_ep, "epoch": epoch_num},
               f"{CKPT_DIR}/epoch_{epoch_num}.pt")
    fp16_state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    torch.save({"model": fp16_state, "config": cfg, "loss": avg_ep, "epoch": epoch_num},
               f"{CKPT_DIR}/epoch_{epoch_num}_fp16.pt")
    print(f"  -> saved epoch_{epoch_num}.pt and fp16", flush=True)

    if avg_ep < best:
        best = avg_ep
        torch.save({"model": model.state_dict(), "config": cfg, "loss": best, "epoch": epoch_num},
                   f"{CKPT_DIR}/best.pt")
        print(f"  -> NEW BEST ({best:.4f})", flush=True)

hrs = (time.time() - t0) / 3600
print(f"\n{'='*72}")
print(f"  DONE! {hrs:.2f}hrs | Final: {el/max(es,1):.4f} | Best: {best:.4f}")
print(f"{'='*72}", flush=True)
