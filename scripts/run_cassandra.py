# Cassandra T1 — Local Inference (Q3-enhanced quantile unmasking)
# Works on ep5 checkpoint — new generate() gives better quality without retraining.
# Run: python I:\sophiat1\run_cassandra.py

import sys, torch
sys.path.insert(0, "I:/sophiat1/src")
from model.sophia_t1 import SophiaT1Model
from model.config import sophia_t1_base, sophia_t1_base_q3
from tokenizers import Tokenizer

USE_Q3_ARCHITECTURE = True   # Loads ep5 into Q3-enhanced shell (AdaLN zero-init → no behavior change until trained)

print("Loading Cassandra T1 (epoch 5, loss 2.26)...")
ckpt = torch.load("I:/sophiat1/checkpoints/epoch5/cassandra_ep5_fp16.pt",
                  map_location="cuda", weights_only=False)
cfg = sophia_t1_base_q3() if USE_Q3_ARCHITECTURE else sophia_t1_base()
model = SophiaT1Model(cfg).cuda()
state = SophiaT1Model.remap_baseline_state_dict(ckpt["model"], use_adaln=cfg.use_adaln)
state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
result = model.load_state_dict(state, strict=False)
print(f"  Loaded epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f}")
if result.missing_keys:
    print(f"  {len(result.missing_keys)} Q3 additions zero-initialized (normal for ep5)")
model.eval()

tok = Tokenizer.from_file("I:/sophiat1/tokenizer.json")
mask_id = tok.token_to_id("<mask>") or 4
print("Ready. Type a message and press Enter. 'quit' to exit.\n")

SYS = "You are Cassandra T1, a diffusion language model by SOPHIA XT. Direct, helpful, honest."


def generate(prompt: str, max_tokens=100, num_steps=12):
    full = f"Q: {SYS}\n\n{prompt}\nA:"
    ids = torch.tensor([tok.encode(full).ids]).cuda()
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=max_tokens,
            num_steps=num_steps,
            temperature=0.8,
            top_p=0.9,
            beta=0.5,            # quantile offset τ = μ + 0.5σ
            rep_penalty=1.3,
            timestep_power=1.5,
        )
    raw = tok.decode(out[0].tolist())
    return raw.replace(chr(288), " ").replace(chr(266), "\n").strip()


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
        import traceback; traceback.print_exc()

print("Goodbye!")
