"""Train a LoRA-on-triple-attention variant.

Usage:
    python train_triple_lora.py --variant V1 --steps 500 --seed 42

Loads v2_ltmi_triple base (frozen), wraps every attn layer with
TripleAttentionLoRA(variant_config), trains only the LoRA + intervention
parameters. Saves the trained adapter state (small file, ~30-50 MB).
"""
from __future__ import annotations
import sys
import argparse
import gc
import json
import math
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "runner"))
sys.path.insert(0, str(ROOT / "cassandra_src"))

import torch
import torch.nn.functional as F

from cassandra_loader import load
from lora_finetune import encode_pair, SYS, SEQ_LEN  # noqa: F401
from continued_pretrain_t2 import build_pairs_with_lattice, lattice_for_breadcrumb
from triple_attention_lora import VARIANTS, wrap_model_with_lora


# ─── Config ────────────────────────────────────────────────────────────
WEIGHTS_DIR = ROOT / "weights"
ADAPTERS_DIR = WEIGHTS_DIR / "lora_adapters"
ADAPTERS_DIR.mkdir(exist_ok=True)
DATA_BUNDLES = [ROOT / f"{cid}.json" for cid in ("C1", "C2", "C3")]

DEFAULT_STEPS = 300  # LoRA needs fewer steps than full fine-tune
SEQ = 512
BATCH = 2
GA = 2
PEAK_LR = 5e-4  # higher LR for LoRA (only adapters train)
MIN_LR = 5e-5
WARMUP_STEPS = 30
LOG_EVERY = 25


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=list(VARIANTS.keys()), required=True)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warm-start", default="v2_ltmi_triple",
                        help="base checkpoint to LoRA on top of (default v2_ltmi_triple)")
    parser.add_argument("--bundles", nargs="*", default=None)
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()

    variant_factory = VARIANTS[args.variant]
    variant = variant_factory()
    bundles = [Path(b) for b in args.bundles] if args.bundles else DATA_BUNDLES

    out_name = f"lora_{args.variant.lower()}_seed{args.seed}_step{args.steps}"
    if args.tag:
        out_name += f"_{args.tag}"
    out_path = ADAPTERS_DIR / f"{out_name}.pt"

    print(f"\n{'='*72}\nTRIPLE-ATTENTION LoRA TRAINING\n{'='*72}")
    print(f"  variant     : {args.variant} ({variant})")
    print(f"  warm-start  : {args.warm_start}")
    print(f"  steps       : {args.steps}")
    print(f"  seed        : {args.seed}")
    print(f"  bundles     : {[b.name for b in bundles]}")
    print(f"  output      : {out_path}")

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16

    print(f"\n[setup] loading base model {args.warm_start}...")
    model, tok, mask_id = load(args.warm_start, dtype=dtype)

    # Freeze EVERYTHING
    for p in model.parameters():
        p.requires_grad = False

    # Wrap with LoRA + interventions
    print(f"[setup] wrapping {sum(1 for _ in model.layers)} layers with TripleAttentionLoRA...")
    model, n_trainable = wrap_model_with_lora(model, variant)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"[setup] trainable: {n_trainable/1e6:.2f}M / total {n_total/1e9:.2f}B "
          f"({100*n_trainable/n_total:.3f}%)")

    # Move new modules to GPU + dtype
    model = model.to(device="cuda", dtype=dtype)

    print(f"\n[data] building (Q, A, lattice) pairs from {len(bundles)} bundles...")
    pairs = build_pairs_with_lattice(bundles)
    print(f"  {len(pairs)} (Q, A, lattice) training triples")

    examples = []
    for q, a, lat in pairs:
        tgt, _, ans_start = encode_pair(tok, q, a, SEQ, mask_id)
        examples.append((tgt, ans_start, lat))

    # Optimizer over trainable params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(
        trainable_params,
        lr=PEAK_LR, betas=(0.9, 0.95), weight_decay=0.01, eps=1e-6,
    )

    def lr_at(step):
        if step < WARMUP_STEPS:
            return PEAK_LR * step / max(1, WARMUP_STEPS)
        progress = (step - WARMUP_STEPS) / max(1, args.steps - WARMUP_STEPS)
        return MIN_LR + 0.5 * (PEAK_LR - MIN_LR) * (1 + math.cos(math.pi * progress))

    def gate_temp_at(step):
        """V4: anneal gate temperature linearly from start to end over training."""
        if not variant.gate_temp_anneal:
            return 1.0
        progress = step / max(1, args.steps)
        return variant.gate_temp_start + progress * (variant.gate_temp_end - variant.gate_temp_start)

    print(f"\n[train] starting...")
    print(f"  initial cuda_mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    model.train()
    losses: list[float] = []
    aux_losses: list[float] = []
    t_start = time.time()
    optim.zero_grad()

    for step in range(1, args.steps + 1):
        # ── V4: set gate temp on every wrapper for this step ──
        if variant.gate_temp_anneal:
            current_temp = gate_temp_at(step)
            for layer in model.layers:
                if hasattr(layer.attn, "set_gate_temp"):
                    layer.attn.set_gate_temp(current_temp)

        idxs = torch.randint(0, len(examples), (BATCH,)).tolist()
        batch_targets = []
        batch_loss_masks = []
        batch_anchor_masks = []
        batch_lattice = []
        for i in idxs:
            tgt, ans_start, lat = examples[i]
            batch_targets.append(tgt)
            lm = [False] * SEQ
            for j in range(ans_start, SEQ):
                if tgt[j] != mask_id:
                    lm[j] = True
            batch_loss_masks.append(lm)
            am = [False] * SEQ
            for j in range(ans_start):
                if tgt[j] != mask_id:
                    am[j] = True
            batch_anchor_masks.append(am)
            batch_lattice.append(lat)

        targets = torch.tensor(batch_targets, dtype=torch.long, device="cuda")
        loss_mask = torch.tensor(batch_loss_masks, dtype=torch.bool, device="cuda")
        anchor_mask_t = torch.tensor(batch_anchor_masks, dtype=torch.bool, device="cuda")

        ratio = 0.5 + 0.4 * torch.rand(1).item()
        rand_mask = torch.rand(*targets.shape, device="cuda") < ratio
        rand_mask = rand_mask & loss_mask
        masked_inputs = targets.clone()
        masked_inputs[rand_mask] = mask_id

        gamma = rand_mask.float().sum(dim=-1) / loss_mask.float().sum(dim=-1).clamp_min(1)
        gamma = gamma.clamp(0.05, 0.95)
        t = gamma

        attn = torch.zeros_like(targets)
        for b, idx in enumerate(idxs):
            tgt_b = examples[idx][0]
            real_len = sum(1 for x in tgt_b if x != mask_id)
            attn[b, :real_len] = 1
        attention_mask = attn

        # LTMi priors
        anchor_scores_t = torch.zeros_like(targets, dtype=torch.float32)
        anchor_scores_t[anchor_mask_t] = 1.0
        anchor_lattice_t = torch.zeros(BATCH, SEQ, 3, dtype=torch.long, device="cuda")
        for b, (lx, ly, lz) in enumerate(batch_lattice):
            anchor_lattice_t[b, :, 0] = torch.where(
                anchor_mask_t[b],
                torch.tensor(int(lx), device="cuda", dtype=torch.long),
                torch.tensor(0, device="cuda", dtype=torch.long),
            )
            anchor_lattice_t[b, :, 1] = torch.where(
                anchor_mask_t[b],
                torch.tensor(int(ly), device="cuda", dtype=torch.long),
                torch.tensor(0, device="cuda", dtype=torch.long),
            )
            anchor_lattice_t[b, :, 2] = torch.where(
                anchor_mask_t[b],
                torch.tensor(int(lz), device="cuda", dtype=torch.long),
                torch.tensor(0, device="cuda", dtype=torch.long),
            )

        # Main forward
        logits = model(
            masked_inputs, attention_mask=attention_mask, t=t,
            anchor_mask=anchor_mask_t,
            anchor_scores=anchor_scores_t,
            anchor_lattice=anchor_lattice_t,
        )

        loss_positions = rand_mask
        if loss_positions.sum() == 0:
            continue
        flat_logits = logits[loss_positions]
        flat_targets = targets[loss_positions]
        main_loss = F.cross_entropy(flat_logits.float(), flat_targets)

        # ── V5: aux contrastive loss ──
        # Run a second forward WITHOUT lattice priors (anchor_lattice=zeros).
        # Force the divergence between the two outputs at locked positions.
        if variant.aux_contrastive:
            zero_lattice = torch.zeros_like(anchor_lattice_t)
            with torch.no_grad():
                pass  # forward below should still compute gradients
            logits_no_lattice = model(
                masked_inputs, attention_mask=attention_mask, t=t,
                anchor_mask=anchor_mask_t,
                anchor_scores=anchor_scores_t,
                anchor_lattice=zero_lattice,
            )
            # Force divergence: MSE between the two logit sets at masked positions
            # NEGATED so we maximize divergence (channel must do work)
            div = F.mse_loss(
                logits[loss_positions].float(),
                logits_no_lattice[loss_positions].float(),
            )
            aux_loss = -variant.aux_contrastive_weight * div
            total_loss = (main_loss + aux_loss) / GA
            aux_losses.append(aux_loss.item() * GA)
        else:
            total_loss = main_loss / GA

        total_loss.backward()

        current_lr = lr_at(step)
        if step % GA == 0:
            for g in optim.param_groups:
                g["lr"] = current_lr
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optim.step()
            optim.zero_grad()

        losses.append(main_loss.item())

        if step % LOG_EVERY == 0 or step == 1:
            recent = sum(losses[-LOG_EVERY:]) / max(1, len(losses[-LOG_EVERY:]))
            elapsed = time.time() - t_start
            eta = elapsed / max(1, step) * (args.steps - step)
            extras = f"temp={gate_temp_at(step):.2f}" if variant.gate_temp_anneal else ""
            if aux_losses:
                extras += f" aux={aux_losses[-1]:+.4f}"
            print(f"step {step:>4} | loss={main_loss.item():.4f} | "
                  f"recent_avg={recent:.4f} | lr={current_lr:.2e} | "
                  f"elapsed={elapsed:.0f}s | eta={eta:.0f}s | "
                  f"mem={torch.cuda.memory_allocated()/1e9:.1f}GB | {extras}")

    # ── Save adapter state ──
    # Only save TRAINABLE params (skip the frozen base weights inside LoRALinear).
    trainable_keys = {n for n, p in model.named_parameters() if p.requires_grad}
    adapter_state = {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
        if k in trainable_keys
    }
    save_payload = {
        "variant": args.variant,
        "variant_config": variant.__dict__,
        "warm_start": args.warm_start,
        "seed": args.seed,
        "steps": args.steps,
        "final_loss": losses[-1] if losses else None,
        "recent_avg_loss": sum(losses[-50:]) / max(1, len(losses[-50:])),
        "adapter_state": adapter_state,
    }
    torch.save(save_payload, out_path)

    loss_log_path = ROOT / "eval_out" / f"{out_name}_loss.json"
    loss_log_path.parent.mkdir(exist_ok=True)
    loss_log_path.write_text(json.dumps({
        "variant": args.variant,
        "seed": args.seed,
        "losses": losses,
        "aux_losses": aux_losses,
        "config": variant.__dict__,
    }, indent=2), encoding="utf-8")

    print(f"\n[train] done in {time.time()-t_start:.0f}s")
    print(f"  saved adapter → {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"  saved loss log → {loss_log_path}")


if __name__ == "__main__":
    main()
