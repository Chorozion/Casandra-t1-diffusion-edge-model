"""Pre-flight for pretrain_from_scratch: build model, forward, backward, save+reload."""
import os, sys, torch
sys.path.insert(0, "src")
from model.config import sophia_t1_base_q3
from model.sophia_t1 import SophiaT1Model

print("GPU:", torch.cuda.get_device_name(0))

print("[1] Build Q3-native model from scratch...")
cfg = sophia_t1_base_q3()
m = SophiaT1Model(cfg).cuda()
print("  Params:", round(m.count_parameters() / 1e6, 1), "M")

print("[2] Forward pass with timestep...")
x = torch.randint(0, cfg.vocab_size, (8, 1024)).cuda()
mask = torch.ones_like(x)
t = torch.rand(8).cuda()
m.train()
with torch.amp.autocast("cuda"):
    y = m(x, mask, t=t)
print("  logits:", tuple(y.shape), "max|y|=", round(y.abs().max().item(), 2))

print("[3] Backward pass at BS=8 SEQ=1024 (actual training config)...")
opt = torch.optim.AdamW(m.parameters(), lr=2e-4)
scaler = torch.amp.GradScaler("cuda")
with torch.amp.autocast("cuda"):
    loss = m.compute_diffusion_loss(x, mask, step_frac=0.5)
scaler.scale(loss).backward()
scaler.unscale_(opt)
scaler.step(opt)
scaler.update()
print("  Loss:", round(loss.item(), 4))
print("  VRAM peak:", round(torch.cuda.max_memory_allocated() / 1024**3, 2), "GB / 80GB")

print("[4] Save + reload round-trip (verify save robustness)...")
os.makedirs("checkpoints", exist_ok=True)
path = "checkpoints/smoke_test.pt"
torch.save({"model": m.state_dict(), "config": cfg, "loss": 0.0, "epoch": 0, "step": 0}, path)
back = torch.load(path, map_location="cpu", weights_only=False)
print("  Saved & reloaded:", len(back["model"]), "tensors,",
      round(os.path.getsize(path) / 1024**3, 2), "GB")
os.remove(path)

print("\nALL OK — READY TO LAUNCH PRETRAINING")
