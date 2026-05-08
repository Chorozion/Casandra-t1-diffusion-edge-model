"""
Cassandra T1 — PRETRAIN FROM SCRATCH with Q3-native architecture.

Different from continue_from_ep5.py:
  - Random init (no checkpoint load)
  - AdaLN + confidence head trained from step 0 (not retrofit)
  - Pretraining LR: peak 2e-4 (not fine-tune 5e-5)
  - Save every 5K steps (survives crashes, 36 saves per run)
  - NO in-training FP16 save (regen separately after training)
  - Longer warmup (5%) and curriculum (30% easy phase)
  - Gentler category weighting (identity 2x, appliance 1.5x vs 3x/2x)
  - Loss curriculum: assumes from-scratch needs time to escape garbage minima

Designed for RunPod A100 SXM 80GB, ~5-6 days for 5 epochs.
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
DATA_PATH  = "data/train_v2.jsonl"
TOKENIZER  = "tokenizer.json"
CKPT_DIR   = "checkpoints"
CANARY_LOG = f"{CKPT_DIR}/canary_pretrain.jsonl"
RUN_NAME   = "v2_scratch"

BS = 8
SEQ = 1024
EPOCHS = 5
GA = 4              # effective batch 32
PEAK_LR = 2e-4      # pretraining LR, NOT 5e-5 fine-tune
MIN_LR  = 1e-5
WARMUP_FRAC = 0.05
PGM_ALPHA = 0.8
PGM_BETA  = 0.01
PGM_GAMMA = 0.2
PGM_DELTA = 0.5

# Category weights — gentler than fine-tune (identity signal comes from pretraining too)
LAMBDA_IDENTITY  = 2.0
LAMBDA_APPLIANCE = 1.5
LAMBDA_SPATIAL   = 2.0
Z_LOSS_ALPHA     = 0.001

# Curriculum: 30% easy phase (vs 10% in ep6 run)
CURRICULUM_EASY_FRAC = 0.30
CURRICULUM_EASY_MAX_RATIO = 0.55

# Checkpointing
SAVE_EVERY_STEPS = 5000
CANARY_EVERY     = 500
MAX_CANARY_FAIL  = 3

# ===== Build model from scratch =====
print("=" * 72)
print("  CASSANDRA T1 — PRETRAIN FROM SCRATCH (Q3-native)")
print(f"  Run: {RUN_NAME} | {EPOCHS} epochs | save every {SAVE_EVERY_STEPS} steps")
print("=" * 72, flush=True)

cfg = sophia_t1_base_q3()
model = SophiaT1Model(cfg).cuda()
print(f"  Params: {model.count_parameters()/1e6:.1f}M | AdaLN native | conf head native", flush=True)

# ===== Optional resume =====
RESUME_FROM = os.environ.get("CASSANDRA_RESUME", "")
resume_step = 0
resume_ep = 0
resume_opt_state = None
resume_sched_step = 0
if RESUME_FROM and os.path.exists(RESUME_FROM):
    print(f"\n[RESUME] Loading {RESUME_FROM}...", flush=True)
    ck = torch.load(RESUME_FROM, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    resume_step = ck.get("step", 0)
    resume_ep   = ck.get("epoch", 1) - 1  # start of that epoch
    resume_opt_state = ck.get("optimizer")
    resume_sched_step = ck.get("sched_step", resume_step)
    print(f"  Resumed at step {resume_step} ep{resume_ep+1} loss {ck.get('loss','?')}", flush=True)
    del ck
    torch.cuda.empty_cache()

tok = Tokenizer.from_file(TOKENIZER)
mask_id = tok.token_to_id("<mask>") or 4
print(f"  Tokenizer: {tok.get_vocab_size()} tokens | mask_id={mask_id}", flush=True)

# ===== Dataset =====
class DS(Dataset):
    def __init__(self, path, tok, ml=1024):
        self.samples = []
        self.tok, self.ml = tok, ml
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
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1])[:15]:
            print(f"    {c:20s} {n:>8,}", flush=True)

    def __len__(self): return len(self.samples)

    def __getitem__(self, i):
        text, cat = self.samples[i]
        ids = self.tok.encode(text).ids[:self.ml]
        pad = self.ml - len(ids)
        if cat == "identity":            cat_w = LAMBDA_IDENTITY
        elif cat == "appliance_repair":  cat_w = LAMBDA_APPLIANCE
        else:                            cat_w = 1.0
        return {
            "input_ids":      torch.tensor(ids + [0]*pad, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(ids) + [0]*pad, dtype=torch.long),
            "cat_weight":     torch.tensor(cat_w, dtype=torch.float32),
        }

ds = DS(DATA_PATH, tok, SEQ)
dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
total_steps = len(dl) * EPOCHS // GA
print(f"  Steps: {total_steps:,} | BS: {BS}x{GA}={BS*GA} | SEQ: {SEQ}", flush=True)

# ===== Q3 PGM LR =====
class Q3PGMSchedule:
    def __init__(self, opt, total, warmup_frac, peak, floor,
                 alpha=PGM_ALPHA, beta=PGM_BETA, gamma=PGM_GAMMA, delta=PGM_DELTA):
        self.opt = opt; self.total = total; self.warmup = max(1, int(total * warmup_frac))
        self.peak, self.floor = peak, floor
        self.alpha, self.beta, self.gamma, self.delta = alpha, beta, gamma, delta
        self.step_count = 0
    def step(self):
        self.step_count += 1
        s = self.step_count
        if s <= self.warmup:
            frac = s / self.warmup
            lr = self.floor + (self.peak-self.floor) * (frac**self.alpha) * (1 - math.exp(-self.beta*s))
        else:
            d = min((s-self.warmup)/max(self.total-self.warmup, 1), 1.0)
            lr = self.floor + (self.peak-self.floor) * (1 - d**self.gamma) ** self.delta
        for pg in self.opt.param_groups: pg["lr"] = lr
        return lr

opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.1)
if resume_opt_state is not None:
    opt.load_state_dict(resume_opt_state)
    print(f"  Optimizer state restored", flush=True)
sched = Q3PGMSchedule(opt, total_steps, WARMUP_FRAC, PEAK_LR, MIN_LR)
sched.step_count = resume_sched_step
scaler = torch.amp.GradScaler("cuda")

print(f"  Q3 PGM: peak={PEAK_LR:.0e} floor={MIN_LR:.0e} warmup={WARMUP_FRAC*100:.0f}% α={PGM_ALPHA} δ={PGM_DELTA}")
print(f"  Category weights: identity={LAMBDA_IDENTITY}× appliance={LAMBDA_APPLIANCE}× spatial={LAMBDA_SPATIAL}×")
print(f"  Curriculum: {CURRICULUM_EASY_FRAC*100:.0f}% easy (max mask {CURRICULUM_EASY_MAX_RATIO}) then full")
print(f"  Save every {SAVE_EVERY_STEPS} steps | canary every {CANARY_EVERY}", flush=True)

# ===== Canary =====
try:
    from eval.canary_identity import run_canary
except Exception as e:
    print(f"  WARN: canary disabled ({e})")
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

# ===== Robust save with verify =====
# Strategy: write to /tmp (local disk), atomic rename to MooseFS.
# MooseFS chokes on large sequential writes; local-disk → atomic mv avoids it.
# Include optimizer ONLY for full saves (epoch-boundary). Step saves are model-only.
def _move_state_to_cpu(state_dict):
    return {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in state_dict.items()}

KEEP_LAST_STEP_CKPTS = 2   # delete older step saves to stay under pod quota

def _purge_old_step_ckpts():
    """Keep only the K newest `step_` checkpoints; delete the rest. Never touches epoch_ saves."""
    import re
    pat = re.compile(rf"^{RUN_NAME}_step_(\d+)\.pt$")
    entries = []
    for f in os.listdir(CKPT_DIR):
        m = pat.match(f)
        if m:
            entries.append((int(m.group(1)), f))
    entries.sort(reverse=True)
    for _, f in entries[KEEP_LAST_STEP_CKPTS:]:
        try:
            os.remove(os.path.join(CKPT_DIR, f))
            print(f"  -> purged {f}", flush=True)
        except Exception as e:
            print(f"  -> failed to purge {f}: {e}", flush=True)

def save_checkpoint(step, epoch, loss, tag="step", include_optimizer=False):
    final = f"{CKPT_DIR}/{RUN_NAME}_{tag}_{step}.pt"
    tmp_local = f"/tmp/{RUN_NAME}_{tag}_{step}.pt"
    try:
        payload = {
            "model": _move_state_to_cpu(model.state_dict()),
            "config": cfg, "loss": loss, "epoch": epoch, "step": step,
            "sched_step": sched.step_count,
        }
        if include_optimizer:
            payload["optimizer"] = opt.state_dict()
        torch.save(payload, tmp_local)
        # Verify by reloading from local
        test = torch.load(tmp_local, map_location="cpu", weights_only=False)
        assert len(test["model"]) == len(model.state_dict()), "size mismatch on reload"
        # Atomic move to MooseFS
        import shutil
        shutil.move(tmp_local, final)
        size_gb = os.path.getsize(final) / 1024**3
        print(f"  -> SAVED {final} ({size_gb:.1f}GB)", flush=True)
        # Purge older step checkpoints to stay under pod quota
        if tag == "step":
            _purge_old_step_ckpts()
        return True
    except Exception as e:
        print(f"  -> SAVE FAILED {type(e).__name__}: {e} (tried {final})", flush=True)
        try: os.remove(tmp_local)
        except: pass
        return False

# ===== Training loop =====
model.train()
opt.zero_grad()
gs = resume_step
best = float("inf")
t0 = time.time()
canary_fail_streak = 0
# Skip epochs already completed
_start_epoch = resume_ep

os.makedirs(CKPT_DIR, exist_ok=True)

for ep in range(_start_epoch, EPOCHS):
    el, es = 0.0, 0
    for bi, batch in enumerate(dl):
        ids  = batch["input_ids"].cuda()
        msk  = batch["attention_mask"].cuda()
        cw   = batch["cat_weight"].cuda()

        step_frac = gs / max(total_steps, 1)
        # Curriculum: easy phase uses bounded mask ratio
        if step_frac < CURRICULUM_EASY_FRAC:
            r_hi = cfg.mask_ratio_min + (CURRICULUM_EASY_MAX_RATIO - cfg.mask_ratio_min) * (step_frac / CURRICULUM_EASY_FRAC)
        else:
            r_hi = cfg.mask_ratio_max
        mr = cfg.mask_ratio_min + (r_hi - cfg.mask_ratio_min) * random.betavariate(2.0, 2.0)

        with torch.amp.autocast("cuda"):
            B, S = ids.shape
            rm = torch.rand(B, S, device=ids.device) < mr
            # 15% of batch gets span masking (Q3 §3.2.7)
            if random.random() < cfg.span_mask_prob:
                span_len = max(1, int(random.gauss(cfg.span_mean_length, 1.0)))
                for b in range(B // 2):
                    start = random.randint(0, max(1, S - span_len - 1))
                    rm[b, start:start + span_len] = True
            rm = rm & msk.bool()
            tgt = ids.clone()
            masked = ids.clone()
            masked[rm] = mask_id

            gamma_t = rm.float().mean(dim=-1).clamp(0.05, 0.95)
            logits = model(masked, msk, t=gamma_t)

            fl = logits[rm]
            ft = tgt[rm]
            if fl.numel() > 0:
                per_tok = F.cross_entropy(fl, ft, reduction="none")
                spatial_w = 1.0 + (ft >= 32000).float() * (LAMBDA_SPATIAL - 1.0)
                sample_idx = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, S)[rm]
                sample_w = cw[sample_idx]
                weighted = per_tok * spatial_w * sample_w
                loss = weighted.mean() / GA
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
                print(f"  step {gs:>6d}/{total_steps} ep{ep+1} | loss {avg:.4f} | lr {cur_lr:.2e} | mr {mr:.2f} | {sps:.2f}s/s | ETA {eta:.0f}m | {vram:.1f}GB", flush=True)

            # Canary
            if run_canary is not None and gs % CANARY_EVERY == 0:
                res = run_canary(model, tok, canary_gen, step=f"ep{ep+1}-s{gs}", log_path=CANARY_LOG)
                if res["fail_rate"] > 0.5:
                    canary_fail_streak += 1
                    if canary_fail_streak >= MAX_CANARY_FAIL:
                        print(f"  ABORT: {MAX_CANARY_FAIL} consecutive canary fails", flush=True)
                        save_checkpoint(gs, ep+1, el/max(es,1), tag="abort")
                        sys.exit(1)
                else:
                    canary_fail_streak = 0

            # Periodic save — model-only, no optimizer (saves 11GB per write)
            if gs % SAVE_EVERY_STEPS == 0:
                avg = el / max(es, 1)
                save_checkpoint(gs, ep+1, avg, tag="step", include_optimizer=False)
                if avg < best:
                    best = avg
                    # Track best in a metadata file, don't duplicate the 5GB ckpt
                    with open(f"{CKPT_DIR}/{RUN_NAME}_best.txt", "w") as bf:
                        bf.write(f"step={gs}\nepoch={ep+1}\nloss={best:.6f}\nfile=v2_scratch_step_{gs}.pt\n")
                    print(f"  -> NEW BEST loss {best:.4f} (pointer in {RUN_NAME}_best.txt)", flush=True)

    # End-of-epoch save — INCLUDES optimizer state so we can resume
    avg_ep = el / max(es, 1)
    print(f"\n  Epoch {ep+1} complete | loss {avg_ep:.4f}", flush=True)
    save_checkpoint(gs, ep+1, avg_ep, tag=f"epoch{ep+1}", include_optimizer=True)
    if avg_ep < best:
        best = avg_ep
        with open(f"{CKPT_DIR}/{RUN_NAME}_best.txt", "w") as bf:
            bf.write(f"step={gs}\nepoch={ep+1}\nloss={best:.6f}\nfile={RUN_NAME}_epoch{ep+1}_{gs}.pt\n")
        print(f"  -> NEW BEST after epoch {ep+1}: loss {best:.4f}", flush=True)

hrs = (time.time() - t0) / 3600
print(f"\n{'='*72}")
print(f"  DONE! {hrs:.2f}hrs | Final: {el/max(es,1):.4f} | Best: {best:.4f}")
print(f"{'='*72}", flush=True)
