"""Evaluate a trained LoRA adapter on the n=172 expanded held-out corpus.

Loads v2_ltmi_triple base, wraps with TripleAttentionLoRA(variant), applies
the trained adapter state, then runs the standard multidomain forced+unforced
eval and dumps results as jsonl.

Usage:
    python eval_with_lora.py --adapter weights/lora_adapters/lora_v1_seed42_step300.pt
    python eval_with_lora.py --all-in-dir weights/lora_adapters
"""
from __future__ import annotations
import argparse
import gc
import json
import sys
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

from cassandra_loader import load
from triple_attention_lora import VARIANTS, VariantConfig, wrap_model_with_lora
import run_t2_multidomain_eval as eval_mod


def load_adapter(adapter_path: Path) -> tuple[str, VariantConfig, dict, int]:
    """Returns (variant_name, variant_config, state_dict, seed)."""
    payload = torch.load(adapter_path, map_location="cpu", weights_only=False)
    variant_name = payload["variant"]
    cfg_dict = payload["variant_config"]
    cfg = VariantConfig(**cfg_dict)
    return variant_name, cfg, payload["adapter_state"], payload.get("seed", 0)


def build_wrapped_model(variant: VariantConfig, base_arm: str = "v2_ltmi_triple"):
    """Load base and wrap with TripleAttentionLoRA configured per variant."""
    model, tok, mask_id = load(base_arm, dtype=torch.bfloat16)
    for p in model.parameters():
        p.requires_grad = False
    model, n_trainable = wrap_model_with_lora(model, variant)
    model = model.to(device="cuda", dtype=torch.bfloat16)
    return model, tok, mask_id, n_trainable


def load_adapter_into_model(model, adapter_state: dict, dtype=torch.bfloat16):
    """Copy adapter params into the wrapped model. Adapter keys are the
    trainable param names; unmatched keys are reported."""
    model_state_dict = model.state_dict()
    matched = 0
    skipped = []
    for key, val in adapter_state.items():
        if key in model_state_dict:
            target = model_state_dict[key]
            model_state_dict[key].copy_(val.to(target.device, target.dtype))
            matched += 1
        else:
            skipped.append(key)
    print(f"  [adapter] loaded {matched}/{len(adapter_state)} keys; "
          f"skipped {len(skipped)}")
    if skipped:
        print(f"  [adapter] first skip: {skipped[:3]}")


def use_extended_corpus():
    """Point eval_mod at the n=172 expanded query set."""
    # v2 file has _metadata / _generated_at top-level keys that the eval
    # iterator would mistake for query lists. eval_baseline_n172.py writes
    # the cleaned copy; prefer it, else strip metadata inline.
    cleaned = ROOT / "queries_heldout_extended_v2_clean.json"
    if not cleaned.exists():
        import json as _json
        src = ROOT / "queries_heldout_extended_v2.json"
        data = _json.loads(src.read_text(encoding="utf-8"))
        clean = {k: v for k, v in data.items() if isinstance(v, list)}
        cleaned.write_text(_json.dumps(clean, indent=2, ensure_ascii=False), encoding="utf-8")
    eval_mod.QUERIES_EXT = cleaned
    print(f"  [eval] using cleaned corpus: {cleaned.name}")


def eval_one_adapter(adapter_path: Path, base_arm: str = "v2_ltmi_triple"):
    variant_name, variant, adapter_state, seed = load_adapter(adapter_path)
    print(f"\n{'-'*72}")
    print(f"EVAL {adapter_path.name}: variant={variant_name} seed={seed}")
    print(f"{'-'*72}")

    model, tok, mask_id, n_trainable = build_wrapped_model(variant, base_arm)
    load_adapter_into_model(model, adapter_state)
    model.eval()

    arm_label = f"{variant_name.lower()}_seed{seed}"
    supply_ltmi = bool(getattr(model.config, "attention_use_ltmi_priors", False))
    rows = eval_mod.evaluate_checkpoint(model, tok, arm_label,
                                         supply_ltmi_priors=supply_ltmi)

    out_path = ROOT / "eval_out" / f"lora_eval_{arm_label}_step{adapter_path.stem.split('step')[-1]}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            r["adapter_file"] = adapter_path.name
            r["variant"] = variant_name
            r["seed"] = seed
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  [eval] saved {len(rows)} rows → {out_path.name}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=str, default=None,
                        help="single adapter .pt path")
    parser.add_argument("--all-in-dir", type=str, default=None,
                        help="directory of adapter .pt files; eval all matching")
    parser.add_argument("--base-arm", default="v2_ltmi_triple")
    parser.add_argument("--pattern", default="lora_*_step300.pt",
                        help="glob pattern when --all-in-dir is used")
    args = parser.parse_args()

    use_extended_corpus()
    torch.manual_seed(eval_mod.SEED)

    if args.adapter:
        eval_one_adapter(Path(args.adapter), args.base_arm)
    elif args.all_in_dir:
        adapters = sorted(Path(args.all_in_dir).glob(args.pattern))
        print(f"[main] found {len(adapters)} adapters matching {args.pattern!r}")
        for i, ap in enumerate(adapters, 1):
            print(f"\n[main] {i}/{len(adapters)}: {ap.name}")
            try:
                eval_one_adapter(ap, args.base_arm)
            except Exception as e:
                print(f"  FAIL: {type(e).__name__}: {e}")
    else:
        parser.error("specify --adapter or --all-in-dir")


if __name__ == "__main__":
    main()
