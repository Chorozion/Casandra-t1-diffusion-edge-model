"""Cassandra T1 — Training ONLY. Data already uploaded. All novel architecture features enabled."""
import os, sys, json, random, time, math
os.chdir("/workspace/cassandra")
sys.path.insert(0, "src")

print("=" * 60)
print("  CASSANDRA T1 — FULL TRAINING")
print("  Novel PDE Lattice + Parabolic GD + Multi-Objective Loss")
print("=" * 60, flush=True)

import torch, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from model.config import sophia_t1_base
from model.sophia_t1 import SophiaT1Model

# === BPE TOKENIZER ===
tok_path = "tokenizer.json"
if not os.path.exists(tok_path):
    print("Training BPE tokenizer...", flush=True)
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=32000, special_tokens=["<pad>","<eos>","<bos>","<unk>","<mask>"], min_frequency=2)
    txts = []
    with open("data/train.jsonl") as f:
        for i, line in enumerate(f):
            if i >= 100000: break
            txts.append(json.loads(line).get("t",""))
    tokenizer.train_from_iterator(txts, tr)
    tokenizer.save(tok_path)
    print(f"BPE: {tokenizer.get_vocab_size()} tokens", flush=True)
else:
    print(f"Loading existing tokenizer", flush=True)

tok = Tokenizer.from_file(tok_path)
mask_id = tok.token_to_id("<mask>") or 4

# === DATASET ===
class DS(Dataset):
    def __init__(s, path, tok, ml=1024):
        s.samples, s.tok, s.ml = [], tok, ml
        with open(path) as f:
            for line in f:
                t = json.loads(line).get("t","")
                if len(t) > 30: s.samples.append(t)
        print(f"Dataset: {len(s.samples):,} samples", flush=True)
    def __len__(s): return len(s.samples)
    def __getitem__(s, i):
        ids = s.tok.encode(s.samples[i]).ids[:s.ml]
        p = s.ml - len(ids)
        return {"input_ids": torch.tensor(ids+[0]*p, dtype=torch.long),
                "attention_mask": torch.tensor([1]*len(ids)+[0]*p, dtype=torch.long)}

# === PARABOLIC GRADIENT SCHEDULE (SOPHIA XT) ===
class ParabolicSchedule:
    def __init__(self, optimizer, total_steps, warmup=0.05, peak_lr=2e-4, min_lr=1e-5):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup)
        self.peak_lr = peak_lr
        self.min_lr = min_lr
        self.step_count = 0
    def step(self):
        self.step_count += 1
        if self.step_count < self.warmup_steps:
            t = self.step_count / self.warmup_steps
            lr = self.min_lr + (self.peak_lr - self.min_lr) * (1 - math.exp(-5 * t))
        else:
            t = (self.step_count - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
            lr = self.min_lr + (self.peak_lr - self.min_lr) * (1 - t) ** 2
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr
        return lr
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]

# === MODEL SETUP ===
cfg = sophia_t1_base()
model = SophiaT1Model(cfg).cuda()
BS = 16
SEQ = 1024
EP = 15
ga = 2
LAMBDA_SPATIAL = 2.0

ds2 = DS("data/train.jsonl", tok, SEQ)
dl = DataLoader(ds2, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
opt = torch.optim.AdamW(model.parameters(), lr=2e-4, betas=(0.9, 0.95), weight_decay=0.1)
total_steps = len(dl) * EP // ga
sched = ParabolicSchedule(opt, total_steps, warmup=0.05, peak_lr=2e-4, min_lr=1e-5)
os.makedirs("checkpoints", exist_ok=True)

print(f"\n  CASSANDRA T1-BASE on A100 SXM 80GB")
print(f"  Params: {model.count_parameters()/1e6:.1f}M | Data: {len(ds2):,} | Epochs: {EP}")
print(f"  Batch: {BS}x{ga}={BS*ga} | Steps: {total_steps:,} | Seq: {SEQ}")
print(f"  Loss: L_diffusion + {LAMBDA_SPATIAL}x L_spatial")
print(f"  Schedule: Parabolic GD (SOPHIA XT)")
print(f"  Mask sampling: Beta(2,2) PDE-matched")
print(f"  VRAM: {torch.cuda.memory_allocated()/1024**3:.1f}GB", flush=True)

# === TRAINING ===
model.train()
opt.zero_grad()
gs, best = 0, 999
t0 = time.time()
scaler = torch.amp.GradScaler("cuda")

for ep in range(EP):
    el, es, el_s = 0, 0, 0
    for bi, batch in enumerate(dl):
        ids = batch["input_ids"].cuda()
        msk = batch["attention_mask"].cuda()
        mr = random.betavariate(2.0, 2.0) * 0.7 + 0.15
        with torch.amp.autocast("cuda"):
            bsz, sl = ids.shape
            rm = torch.rand(bsz, sl, device=ids.device) < mr
            rm = rm & msk.bool()
            tgt = ids.clone()
            ids[rm] = mask_id
            logits = model.forward(ids, msk)
            fl = logits[rm]
            ft = tgt[rm]
            if fl.numel() > 0:
                loss_per_token = F.cross_entropy(fl, ft, reduction='none')
                spatial_mask = (ft >= 32000).float()
                weights = 1.0 + spatial_mask * (LAMBDA_SPATIAL - 1.0)
                loss = (loss_per_token * weights).mean() / ga
                el_s += (loss_per_token * spatial_mask).sum().item() / max(spatial_mask.sum().item(), 1)
            else:
                loss = torch.tensor(0.0, device=ids.device, requires_grad=True)
        scaler.scale(loss).backward()
        el += loss.item() * ga
        es += 1
        if (bi + 1) % ga == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            opt.zero_grad()
            gs += 1
            if gs % 50 == 0:
                avg = el / es
                sps = gs / (time.time() - t0)
                eta = (total_steps - gs) / max(sps, 0.001) / 60
                cur_lr = sched.get_last_lr()[0]
                vram = torch.cuda.memory_allocated() / 1024**3
                print(f"  step {gs:>6d}/{total_steps} | loss {avg:.4f} | spatial {el_s/max(es,1):.4f} | lr {cur_lr:.2e} | {sps:.2f} s/s | ETA {eta:.0f}m | VRAM {vram:.1f}GB", flush=True)
    avg_ep = el / max(es, 1)
    avg_spatial = el_s / max(es, 1)
    print(f"\n  Epoch {ep+1}/{EP} | loss: {avg_ep:.4f} | spatial: {avg_spatial:.4f}", flush=True)
    if (ep + 1) % 3 == 0:
        torch.save({"model": model.state_dict(), "config": cfg, "loss": avg_ep, "epoch": ep+1},
                   f"checkpoints/epoch_{ep+1}.pt")
        print(f"  -> Checkpoint epoch {ep+1}", flush=True)
    if avg_ep < best:
        best = avg_ep
        torch.save({"model": model.state_dict(), "config": cfg, "loss": best, "epoch": ep+1},
                   "checkpoints/best.pt")
        print(f"  -> New best (loss={best:.4f})\n", flush=True)

torch.save({"model": model.state_dict(), "config": cfg, "loss": el/max(es,1)},
           "checkpoints/final.pt")
hrs = (time.time() - t0) / 3600
print(f"\n{'='*60}")
print(f"  DONE! {hrs:.2f} hours | Best loss: {best:.4f}")
print(f"  Checkpoints: checkpoints/")
print(f"{'='*60}", flush=True)
