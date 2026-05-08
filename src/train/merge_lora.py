"""
Sophia T1 — Merge LoRA weights into base model
Run after training to create a standalone model.

Usage:
  python src/train/merge_lora.py --checkpoint checkpoints/sophia-t1-v0.1/final
"""

import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "I:/sophiat1/models/LLaDA-8B-Instruct"
OUTPUT = "I:/sophiat1/models/sophia-t1-merged"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default=OUTPUT)
    args = parser.parse_args()

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )

    print(f"Loading LoRA from {args.checkpoint}...")
    model = PeftModel.from_pretrained(model, args.checkpoint)

    print("Merging weights...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {args.output}...")
    model.save_pretrained(args.output, safe_serialization=True)

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.save_pretrained(args.output)

    print("Done! Merged model ready for inference or GGUF conversion.")


if __name__ == "__main__":
    main()
