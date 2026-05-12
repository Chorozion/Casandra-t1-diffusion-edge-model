"""Drive all 18 LoRA training runs: 6 variants × 3 seeds.

Sequential. Each run logs to its own file. Total ~2.7 hours expected.
"""
from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path("D:/cassandra-eval")
ADAPTERS_DIR = ROOT / "weights" / "lora_adapters"
LOG_DIR = ROOT / "eval_out" / "lora_train_logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6"]
SEEDS = [42, 1337, 0xC0FFEE]
STEPS = 300


def already_trained(variant: str, seed: int, steps: int) -> bool:
    name = f"lora_{variant.lower()}_seed{seed}_step{steps}.pt"
    return (ADAPTERS_DIR / name).exists()


def run_one(variant: str, seed: int) -> dict:
    log_path = LOG_DIR / f"lora_{variant.lower()}_seed{seed}_step{STEPS}.log"
    if already_trained(variant, seed, STEPS):
        print(f"  [skip] {variant} seed={seed} already trained")
        return {"variant": variant, "seed": seed, "status": "skipped"}

    t0 = time.time()
    cmd = [
        sys.executable, str(ROOT / "runner" / "train_triple_lora.py"),
        "--variant", variant,
        "--seed", str(seed),
        "--steps", str(STEPS),
    ]
    with log_path.open("w", encoding="utf-8") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = time.time() - t0
    return {
        "variant": variant,
        "seed": seed,
        "status": "ok" if result.returncode == 0 else f"fail rc={result.returncode}",
        "elapsed_s": round(elapsed, 1),
        "log": str(log_path),
    }


def main():
    grand_start = time.time()
    total = len(VARIANTS) * len(SEEDS)
    print(f"\n{'='*72}\nTRAIN ALL LoRA VARIANTS\n{'='*72}")
    print(f"  variants: {VARIANTS}")
    print(f"  seeds   : {SEEDS}")
    print(f"  steps   : {STEPS}")
    print(f"  total   : {total} runs")
    print(f"  logs    : {LOG_DIR}")
    print()

    results = []
    for v_idx, variant in enumerate(VARIANTS):
        for s_idx, seed in enumerate(SEEDS):
            i = v_idx * len(SEEDS) + s_idx + 1
            print(f"[{i}/{total}] {variant} seed={seed}... ", end="", flush=True)
            r = run_one(variant, seed)
            results.append(r)
            elapsed_grand = time.time() - grand_start
            eta = (elapsed_grand / i) * (total - i)
            print(f"{r['status']} ({r.get('elapsed_s', '—')}s) "
                  f"| grand={elapsed_grand:.0f}s eta={eta:.0f}s")

    print(f"\n{'='*72}\nALL DONE in {time.time()-grand_start:.0f}s\n{'='*72}")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_skip = sum(1 for r in results if r["status"] == "skipped")
    n_fail = total - n_ok - n_skip
    print(f"  ok: {n_ok}, skipped: {n_skip}, failed: {n_fail}")

    # Summary
    print(f"\n{'variant':<8}{'seed':<10}{'status':<10}{'elapsed':<10}")
    for r in results:
        seed_str = str(r["seed"])
        print(f"{r['variant']:<8}{seed_str:<10}{r['status']:<10}{r.get('elapsed_s', '—')}s")


if __name__ == "__main__":
    main()
