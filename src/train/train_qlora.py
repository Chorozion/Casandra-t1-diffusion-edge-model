"""
Sophia T1 — QLoRA Fine-tuning Script
Distills Gemma 4 knowledge into LLaDA 8B diffusion model.

Usage:
  python src/train/train_qlora.py --data data/processed/train.jsonl

Hardware: RTX 3090 Ti (24GB VRAM), 94GB RAM
Estimated time: 12-20 hours for 50K samples, 3 epochs
"""

import os
import json
import argparse
import torch
from pathlib import Path
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import torch.nn.functional as F


class DiffusionSFTTrainer(SFTTrainer):
    """Custom trainer that computes masked diffusion loss for LLaDA.
    Single forward pass only — mask tokens before feeding to model."""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        device = input_ids.device
        bsz, seq_len = input_ids.shape

        # Mask ~40% of tokens (fixed ratio for speed/stability)
        mask_token_id = 126336
        rand_mask = torch.rand(bsz, seq_len, device=device) < 0.4
        if attention_mask is not None:
            rand_mask = rand_mask & attention_mask.bool()

        original_ids = input_ids.clone()
        input_ids[rand_mask] = mask_token_id

        # Single forward pass with masked input
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Loss only on masked positions
        flat_logits = logits[rand_mask]
        flat_targets = original_ids[rand_mask]

        if flat_logits.numel() > 0:
            loss = F.cross_entropy(flat_logits, flat_targets)
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)

        return (loss, outputs) if return_outputs else loss

# ── Constants ──────────────────────────────────────────────────────

BASE_MODEL = "I:/sophiat1/models/LLaDA-8B-Instruct"
OUTPUT_DIR = "I:/sophiat1/checkpoints/sophia-t1-v0.1"
DEFAULT_DATA = "I:/sophiat1/data/processed/train.jsonl"


def load_training_data(data_path: str) -> Dataset:
    """Load JSONL training data into a HuggingFace Dataset."""
    records = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            # Format as instruction-following conversation
            system = record.get("system", "You are Sophia T1, a helpful AI assistant.")
            instruction = record.get("instruction", "")
            inp = record.get("input", "")
            output = record.get("output", "")

            if inp:
                prompt = f"{instruction}\n\n{inp}"
            else:
                prompt = instruction

            # Format as chat template
            text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{output}"
            records.append({"text": text})

    print(f"Loaded {len(records)} training samples from {data_path}")
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Sophia T1 QLoRA Training")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Training data JSONL")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-len", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=128, help="LoRA alpha")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    print("=" * 60)
    print("  SOPHIA T1 — QLoRA Training Pipeline")
    print("=" * 60)
    print(f"  Base model:  {BASE_MODEL}")
    print(f"  Data:        {args.data}")
    print(f"  Output:      {args.output}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Batch:       {args.batch_size} x {args.grad_accum} grad_accum = {args.batch_size * args.grad_accum} effective")
    print(f"  LoRA:        r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"  Max length:  {args.max_len}")
    print(f"  GPU:         {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:        {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    # ── 1. Load tokenizer ──────────────────────────────────────────
    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── 2. Quantization config (4-bit NF4) ─────────────────────────
    print("[2/5] Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # ── 3. Load model ──────────────────────────────────────────────
    print("[3/5] Loading LLaDA 8B in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

    # ── 4. LoRA config ─────────────────────────────────────────────
    print("[4/5] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    trainable, total = model.get_nb_trainable_parameters()
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ── 5. Load data ───────────────────────────────────────────────
    print("[5/5] Loading training data...")
    dataset = load_training_data(args.data)

    # Split 95/5 for train/eval
    split = dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # ── Training arguments ─────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=250,
        save_steps=500,
        save_total_limit=3,
        dataloader_num_workers=4,
        gradient_checkpointing=False,
        optim="paged_adamw_8bit",
        report_to="none",
        run_name="sophia-t1-qlora",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # ── Trainer ────────────────────────────────────────────────────
    trainer = DiffusionSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # ── Train ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Starting training...")
    print("=" * 60 + "\n")

    if args.resume:
        trainer.train(resume_from_checkpoint=args.resume)
    else:
        trainer.train()

    # ── Save ───────────────────────────────────────────────────────
    print("\nSaving final model...")
    trainer.save_model(os.path.join(args.output, "final"))
    tokenizer.save_pretrained(os.path.join(args.output, "final"))

    print(f"\nTraining complete! Model saved to: {args.output}/final")
    print("To merge LoRA weights: python src/train/merge_lora.py")


if __name__ == "__main__":
    main()
