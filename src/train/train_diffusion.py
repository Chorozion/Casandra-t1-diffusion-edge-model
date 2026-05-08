"""
Sophia T1 — Diffusion Training Loop
Purpose-built for masked diffusion LM. No HuggingFace Trainer overhead.

Usage:
  python src/train/train_diffusion.py --size small --data data/processed/train.jsonl
  python src/train/train_diffusion.py --size base --data data/processed/train.jsonl

Training T1-Small on 3090 Ti: ~8GB VRAM, ~2-4 hours for 1K samples
Training T1-Base on 3090 Ti: ~14GB VRAM, ~4-8 hours for 1K samples
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.config import sophia_t1_small, sophia_t1_base, SophiaT1Config
from model.sophia_t1 import SophiaT1Model


class TextDataset(Dataset):
    """Simple tokenized text dataset from JSONL."""

    def __init__(self, data_path: str, max_len: int = 1024):
        self.samples = []
        self.max_len = max_len

        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                text = record.get("output", "") or record.get("text", "")
                if len(text) > 50:
                    self.samples.append(text)

        print(f"Loaded {len(self.samples)} text samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        # Simple character-level tokenization for now
        # TODO: Replace with BPE tokenizer
        tokens = [min(ord(c), 32767) for c in text[:self.max_len]]
        # Pad to max_len
        pad_len = self.max_len - len(tokens)
        tokens = tokens + [32767] * pad_len  # pad_token_id
        mask = [1] * (self.max_len - pad_len) + [0] * pad_len
        return {
            "input_ids": torch.tensor(tokens, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }


def train(
    model: SophiaT1Model,
    dataset: TextDataset,
    config: SophiaT1Config,
    output_dir: str,
    epochs: int = 10,
    batch_size: int = 4,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    log_every: int = 10,
    save_every: int = 500,
    grad_accum: int = 4,
    max_grad_norm: float = 1.0,
):
    device = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    # Cosine schedule with warmup
    total_steps = len(dataloader) * epochs // grad_accum
    def lr_schedule(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    import math
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    # Training loop
    global_step = 0
    total_loss = 0.0
    best_loss = float('inf')
    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"  SOPHIA T1 — Diffusion Training")
    print(f"{'='*60}")
    print(f"  Params:     {model.count_parameters()/1e6:.1f}M")
    print(f"  Samples:    {len(dataset)}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch:      {batch_size} x {grad_accum} = {batch_size * grad_accum} effective")
    print(f"  Steps:      {total_steps}")
    print(f"  LR:         {lr}")
    print(f"  VRAM:       {torch.cuda.memory_allocated()/1024**3:.1f} GB")
    print(f"{'='*60}\n")

    model.train()
    optimizer.zero_grad()

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # Variable mask ratio: uniform [0.15, 0.85] for curriculum
            mask_ratio = 0.15 + 0.7 * torch.rand(1).item()

            # Forward + diffusion loss
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = model.compute_diffusion_loss(
                    input_ids, attention_mask, mask_ratio=mask_ratio
                )
                loss = loss / grad_accum

            loss.backward()
            total_loss += loss.item() * grad_accum
            epoch_loss += loss.item() * grad_accum
            epoch_steps += 1

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % log_every == 0:
                    avg_loss = total_loss / (global_step * grad_accum / log_every) if global_step > 0 else total_loss
                    elapsed = time.time() - start_time
                    steps_per_sec = global_step / max(elapsed, 1)
                    eta = (total_steps - global_step) / max(steps_per_sec, 0.001)
                    current_lr = scheduler.get_last_lr()[0]
                    vram = torch.cuda.memory_allocated() / 1024**3

                    recent_loss = epoch_loss / max(epoch_steps, 1)
                    print(
                        f"  step {global_step:>5d}/{total_steps} | "
                        f"loss {recent_loss:.4f} | "
                        f"lr {current_lr:.2e} | "
                        f"{steps_per_sec:.1f} steps/s | "
                        f"ETA {eta/60:.0f}m | "
                        f"VRAM {vram:.1f}GB"
                    )

                if global_step % save_every == 0:
                    ckpt_path = os.path.join(output_dir, f"step_{global_step}.pt")
                    torch.save({
                        "step": global_step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        "loss": epoch_loss / max(epoch_steps, 1),
                    }, ckpt_path)
                    print(f"  -> Saved checkpoint: {ckpt_path}")

        # End of epoch
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  Epoch {epoch+1}/{epochs} done | avg loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_path = os.path.join(output_dir, "best.pt")
            torch.save({
                "step": global_step,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "config": config,
                "loss": best_loss,
            }, best_path)
            print(f"  -> New best! Saved: {best_path}\n")

    # Final save
    final_path = os.path.join(output_dir, "final.pt")
    torch.save({
        "step": global_step,
        "model": model.state_dict(),
        "config": config,
        "loss": epoch_loss / max(epoch_steps, 1),
    }, final_path)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Total time: {elapsed/3600:.1f} hours")
    print(f"  Best loss:  {best_loss:.4f}")
    print(f"  Model:      {final_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Sophia T1 Diffusion Training")
    parser.add_argument("--size", choices=["small", "base"], default="small")
    parser.add_argument("--data", type=str, default="I:/sophiat1/data/processed/train.jsonl")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    config = sophia_t1_small() if args.size == "small" else sophia_t1_base()
    output_dir = args.output or f"I:/sophiat1/checkpoints/t1-{args.size}"

    print(f"Loading T1-{args.size.title()} ({config.num_params_millions:.0f}M params)...")
    model = SophiaT1Model(config).to("cuda").bfloat16()

    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cuda")
        model.load_state_dict(ckpt["model"])

    dataset = TextDataset(args.data, max_len=args.max_len)

    train(
        model=model,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        grad_accum=args.grad_accum,
    )


if __name__ == "__main__":
    main()
