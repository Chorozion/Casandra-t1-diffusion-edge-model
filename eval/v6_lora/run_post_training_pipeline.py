"""End-to-end pipeline: waits for 18 trained adapters, then runs baseline
eval, eval over all 18 adapters, bootstrap analysis, and prints verdict.

Idempotent: each step checks for existing outputs and skips if done.
Safe to re-run.
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
EVAL_OUT = ROOT / "eval_out"

VARIANTS = ["V1", "V2", "V3", "V4", "V5", "V6"]
SEEDS = [42, 1337, 12648430]
STEPS = 300

POLL_INTERVAL_S = 120  # how often to check adapter files


def expected_adapter_paths() -> list[Path]:
    return [
        ADAPTERS_DIR / f"lora_{v.lower()}_seed{s}_step{STEPS}.pt"
        for v in VARIANTS for s in SEEDS
    ]


def expected_eval_paths() -> list[Path]:
    return [
        EVAL_OUT / f"lora_eval_{v.lower()}_seed{s}_step{STEPS}.jsonl"
        for v in VARIANTS for s in SEEDS
    ]


def wait_for_training():
    """Block until all 18 adapter .pt files exist."""
    expected = expected_adapter_paths()
    while True:
        present = [p for p in expected if p.exists()]
        n_done = len(present)
        if n_done >= len(expected):
            print(f"[pipeline] all {n_done}/{len(expected)} adapters present")
            return
        print(f"[pipeline] waiting for training... {n_done}/{len(expected)} adapters present")
        time.sleep(POLL_INTERVAL_S)


def run_step(label: str, cmd: list[str]):
    print(f"\n{'='*72}\n[pipeline] {label}\n{'='*72}")
    print(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.time() - t0
    print(f"[pipeline] {label} done in {elapsed:.0f}s  rc={result.returncode}")
    return result.returncode == 0


def main():
    print(f"\n{'='*72}\nPOST-TRAINING PIPELINE\n{'='*72}")
    print(f"Steps: wait_training → baseline_eval → 18×lora_eval → bootstrap")
    print()

    wait_for_training()

    # Step 1: baseline eval (n=172) if not done
    baseline_path = EVAL_OUT / "lora_baseline_v2_ltmi_triple_n172.jsonl"
    if baseline_path.exists():
        print(f"[pipeline] baseline eval exists, skipping: {baseline_path.name}")
    else:
        ok = run_step(
            "Baseline eval (v2_ltmi_triple on n=172)",
            [sys.executable, str(ROOT / "runner" / "eval_baseline_n172.py")],
        )
        if not ok:
            print("[pipeline] FAIL — baseline eval failed; abort")
            sys.exit(1)

    # Step 2: eval each of 18 adapters that don't have results yet
    eval_paths = expected_eval_paths()
    for ap in expected_adapter_paths():
        ep = EVAL_OUT / f"lora_eval_{ap.stem.replace('lora_', '').split('_step')[0]}_step{STEPS}.jsonl"
        if ep.exists():
            print(f"[pipeline] {ep.name} exists, skipping")
            continue
        ok = run_step(
            f"LoRA eval — {ap.name}",
            [
                sys.executable, str(ROOT / "runner" / "eval_with_lora.py"),
                "--adapter", str(ap),
            ],
        )
        if not ok:
            print(f"[pipeline] WARN — eval failed for {ap.name}; continuing")

    # Step 3: bootstrap + verdict
    ok = run_step(
        "Bootstrap analysis + pre-registered verdict",
        [sys.executable, str(ROOT / "runner" / "bootstrap_lora_variants.py")],
    )

    # Final summary
    verdict_md = ROOT / "LORA_VERDICT.md"
    if verdict_md.exists():
        print(f"\n{'='*72}\nVERDICT\n{'='*72}")
        print(verdict_md.read_text(encoding="utf-8"))
    else:
        print(f"\n[pipeline] no verdict produced — check bootstrap logs")


if __name__ == "__main__":
    main()
