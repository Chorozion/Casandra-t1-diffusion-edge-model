"""LoRA fine-tune Cassandra v1 on LTMi-XT locus → (Q, A) training pairs.

Loci → training examples by mechanical breadcrumb→question template:
    Q = "Tell me about {breadcrumb[2] or [-1]}: {breadcrumb[0]}"
    A = locus.statement

Anchor-token masking: during training we mask only the answer tokens
(the model must predict the statement given the breadcrumb-derived
question). This is the same access pattern as inference-time forced
decoding, so train/inference are consistent.

LoRA rank=16 on q_proj/k_proj/v_proj/o_proj. 28 layers × 4 modules =
112 wrapped linear layers. Trainable params ≈ 7M (vs 1.3B base).
"""
from __future__ import annotations
import sys, json, time, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))

import torch
import torch.nn as nn
import torch.nn.functional as F
from cassandra_loader import load, decode

# ─── Config ───────────────────────────────────────────────────
LORA_RANK = 16
LORA_ALPHA = 32
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
SEQ_LEN = 192
BATCH = 2
LR = 1e-4
STEPS = 300
LOG_EVERY = 20
ADAPTER_PATH = ROOT / "lora_adapters" / "v1_ltmi_r16.pt"
ADAPTER_PATH.parent.mkdir(exist_ok=True)
SYS = "You are Cassandra T1, a diffusion language model by SOPHIA XT. Direct, helpful, honest."

# ─── LoRA module ─────────────────────────────────────────────
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        # LoRA in fp32 for stability
        self.A = nn.Linear(base.in_features, rank, bias=False)
        self.B = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)
        self.scale = alpha / rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # Cast LoRA path to base dtype for the add
        x_lora = self.A(x.to(self.A.weight.dtype))
        x_lora = self.B(x_lora) * self.scale
        return out + x_lora.to(out.dtype)


def wrap_lora(model: nn.Module, targets=TARGETS, rank=LORA_RANK, alpha=LORA_ALPHA):
    """In-place wrap matching child Linear modules with LoRA."""
    n_wrapped = 0
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                wrapped = LoRALinear(child, rank=rank, alpha=alpha)
                setattr(parent, child_name, wrapped)
                n_wrapped += 1
    return n_wrapped


def lora_state(model: nn.Module) -> dict:
    return {k: v for k, v in model.state_dict().items() if ".A." in k or ".B." in k}


def trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ─── Build training set from .ltmi bundles ─────────────────────
def build_pairs(bundle_paths: list[Path]) -> list[tuple[str, str]]:
    """Each locus → multiple (Q, A) variants for cheap data augmentation."""
    pairs: list[tuple[str, str]] = []
    for p in bundle_paths:
        bundle = json.loads(p.read_text(encoding="utf-8"))
        for locus in bundle["loci"]:
            bc = locus["breadcrumb"]
            stmt = locus["statement"]
            topic = bc[0] if bc else "this"
            concept = bc[2] if len(bc) >= 3 else (bc[-1] if bc else "this")

            # Template variants
            pairs.append((f"Tell me about {concept} in the context of {topic}.", stmt))
            pairs.append((f"What does the source say about {concept}?", stmt))
            pairs.append((f"Regarding {topic}: {concept}?", stmt))
    return pairs


def encode_pair(tok, q: str, a: str, seq_len: int, mask_id: int) -> tuple[list[int], list[int], int]:
    """Return (input_ids, target_ids, answer_start_pos).

    Format: "Q: {SYS}\n\n{q}\nA: {a}" padded to seq_len.
    During training we'll mask only the {a} tokens.
    """
    full = f"Q: {SYS}\n\n{q}\nA:"
    a_text = " " + a
    full_ids = tok.encode(full).ids
    a_ids = tok.encode(a_text).ids
    answer_start = len(full_ids)
    full_seq = full_ids + a_ids
    if len(full_seq) > seq_len:
        full_seq = full_seq[:seq_len]
    pad_len = seq_len - len(full_seq)
    target = full_seq + [mask_id] * pad_len  # target = clean
    return target, target.copy(), answer_start


# ─── Train loop ──────────────────────────────────────────────
def main():
    print(f"\n{'='*72}\nLoRA FINE-TUNE: v1 + LTMi-XT loci\n{'='*72}")
    torch.manual_seed(42)

    model, tok, _ = load("v1")  # fp16 base on CUDA
    cfg = model.config
    mask_id = cfg.mask_token_id

    # CRITICAL: freeze entire base model BEFORE wrapping. LoRA's nn.Linear
    # children default to requires_grad=True so they remain trainable.
    for p in model.parameters():
        p.requires_grad = False

    # Wrap with LoRA (LoRA params are fp32, base stays fp16)
    n_wrapped = wrap_lora(model)

    # Move LoRA params to GPU in fp32 (newly-created modules default to fp32 cpu)
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            mod.A = mod.A.to(device="cuda", dtype=torch.float32)
            mod.B = mod.B.to(device="cuda", dtype=torch.float32)

    n_trainable = trainable_params(model)
    print(f"[lora] wrapped {n_wrapped} linear layers")
    print(f"[lora] trainable params: {n_trainable/1e6:.2f}M (vs 1330M base)")

    # Build dataset
    bundles = [ROOT / f"{cid}.json" for cid in ("C1", "C2", "C3")]
    pairs = build_pairs(bundles)
    print(f"[data] {len(pairs)} training pairs from {len(bundles)} bundles")

    # Encode all pairs once
    examples = []
    for q, a in pairs:
        tgt, _, ans_start = encode_pair(tok, q, a, SEQ_LEN, mask_id)
        examples.append((tgt, ans_start))

    # Optimizer (LoRA only)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, betas=(0.9, 0.95), weight_decay=0.01,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=STEPS, eta_min=LR * 0.1)

    # Train
    model.train()
    losses = []
    t_start = time.time()
    for step in range(1, STEPS + 1):
        # Random batch
        idxs = torch.randint(0, len(examples), (BATCH,)).tolist()
        batch_targets = []
        batch_masks = []  # which positions to compute loss on (answer tokens only)
        for i in idxs:
            tgt, ans_start = examples[i]
            batch_targets.append(tgt)
            # Loss mask = True where position is in answer slot AND is real token (not pad)
            mask = [False] * SEQ_LEN
            for j in range(ans_start, SEQ_LEN):
                if tgt[j] != mask_id:
                    mask[j] = True
            batch_masks.append(mask)

        targets = torch.tensor(batch_targets, dtype=torch.long, device="cuda")
        loss_mask = torch.tensor(batch_masks, dtype=torch.bool, device="cuda")

        # Build masked input: replace answer tokens with mask_id (random ratio for diffusion)
        ratio = 0.5 + 0.4 * torch.rand(1).item()  # 0.5–0.9 mask ratio on answer
        rand_mask = torch.rand(*targets.shape, device="cuda") < ratio
        rand_mask = rand_mask & loss_mask
        masked_inputs = targets.clone()
        masked_inputs[rand_mask] = mask_id

        # Diffusion timestep = mask fraction
        gamma = rand_mask.float().sum(dim=-1) / loss_mask.float().sum(dim=-1).clamp_min(1)
        gamma = gamma.clamp(0.05, 0.95)
        t = gamma

        attention_mask = (targets != mask_id).long()  # attend to real tokens, not pads-as-mask
        # actually build attention from non-pad positions: count from start until ans_end
        attn = torch.zeros_like(targets)
        for b, (tgt, _) in enumerate([examples[i] for i in idxs]):
            real_len = sum(1 for x in tgt if x != mask_id)  # all real tokens
            attn[b, :real_len] = 1
        attention_mask = attn

        logits = model(masked_inputs, attention_mask=attention_mask, t=t)

        # Loss only on positions that were just masked AND are answer tokens
        loss_positions = rand_mask
        if loss_positions.sum() == 0:
            continue
        flat_logits = logits[loss_positions]
        flat_targets = targets[loss_positions]
        loss = F.cross_entropy(flat_logits.float(), flat_targets)

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optim.step()
        sched.step()

        losses.append(loss.item())
        if step % LOG_EVERY == 0 or step == 1:
            recent = sum(losses[-LOG_EVERY:]) / max(1, len(losses[-LOG_EVERY:]))
            elapsed = time.time() - t_start
            print(f"  step {step:>4} | loss={loss.item():.4f} | recent_avg={recent:.4f} | "
                  f"lr={sched.get_last_lr()[0]:.2e} | "
                  f"elapsed={elapsed:.1f}s | mem={torch.cuda.memory_allocated()/1e9:.1f}GB")

    total = time.time() - t_start
    final_avg = sum(losses[-50:]) / min(50, len(losses))
    print(f"\n[done] {STEPS} steps in {total:.1f}s ({STEPS/total:.1f} steps/s)")
    print(f"[done] starting loss ~{losses[0]:.3f} -> final-50-avg {final_avg:.3f}")

    # Save adapter
    adapter = lora_state(model)
    torch.save({
        "lora_state": adapter,
        "config": {"rank": LORA_RANK, "alpha": LORA_ALPHA, "targets": list(TARGETS)},
        "training": {
            "steps": STEPS, "batch": BATCH, "lr": LR, "seq_len": SEQ_LEN,
            "n_pairs": len(pairs), "loss_first": losses[0], "loss_final_avg": final_avg,
            "total_seconds": total,
        },
    }, ADAPTER_PATH)
    print(f"[save] adapter -> {ADAPTER_PATH} ({ADAPTER_PATH.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
