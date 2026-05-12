"""T2.5-random continued pretraining — uniform-random per-locus lattice coords.

Tests whether the SPECIFIC coord-to-locus mapping matters, or whether
only the existence of a deterministic per-locus mapping matters.

Same as continued_pretrain_t2_5_pca.py but with random coords loaded from
lattice_coords_random.json instead of lattice_coords_pca.json.

Output: cassandra_t2_5_random_ltmi-triple_step{N}.pt
"""
from __future__ import annotations

import json
import sys
import shutil
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path("D:/cassandra-eval")
COORDS_PATH = ROOT / "lattice_coords_random.json"

print(f"[t2.5-random] loading coords from {COORDS_PATH}...")
coords_data = json.loads(COORDS_PATH.read_text(encoding="utf-8"))
RANDOM_COORDS = coords_data["coords"]
print(f"  loaded {len(RANDOM_COORDS)} locus → random-coord mappings "
      f"(seed={hex(coords_data['seed'])})")

sys.path.insert(0, str(Path(__file__).parent))
import continued_pretrain_t2 as t2

_original_lattice_for_breadcrumb = t2.lattice_for_breadcrumb

def random_lattice_for_breadcrumb(breadcrumb, dim=t2.LATTICE_DIM):
    if not breadcrumb or len(breadcrumb) != 4:
        return _original_lattice_for_breadcrumb(breadcrumb, dim)
    key = "/".join(breadcrumb).lower()
    if key in RANDOM_COORDS:
        x, y, z = RANDOM_COORDS[key]
        return (int(x), int(y), int(z))
    return _original_lattice_for_breadcrumb(breadcrumb, dim)

t2.lattice_for_breadcrumb = random_lattice_for_breadcrumb
print(f"[t2.5-random] monkey-patched lattice_for_breadcrumb → uniform-random per locus")

_orig_main = t2.main

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--attention", default="ltmi-triple",
                        choices=["single", "triple", "ltmi-triple"])
    parser.add_argument("--steps", type=int, default=t2.DEFAULT_STEPS)
    parser.add_argument("--bundles", nargs="*", default=None)
    parser.add_argument("--warm", default="v1.5")
    parser.add_argument("--ltmi-gate-init", type=float, default=0.1)
    args = parser.parse_args()

    sys.argv = [
        "continued_pretrain_t2_5_random.py",
        "--attention", args.attention,
        "--steps", str(args.steps),
        "--warm", args.warm,
        "--ltmi-gate-init", str(args.ltmi_gate_init),
    ]
    if args.bundles:
        sys.argv.extend(["--bundles", *args.bundles])

    print(f"[t2.5-random] firing t2.main() with random-coord-patched lattice...")
    _orig_main()

    # Copy + remove to avoid destroying T2 BLAKE2b checkpoint (lesson learned)
    for ckpt in (t2.WEIGHTS_DIR).glob(f"cassandra_t2_{args.attention}_step*.pt"):
        new_name = ckpt.with_name(ckpt.name.replace("cassandra_t2_", "cassandra_t2_5_random_"))
        if not new_name.exists():
            shutil.copy2(ckpt, new_name)
            print(f"[t2.5-random] copied: {ckpt.name} → {new_name.name}")
        else:
            print(f"[t2.5-random] skip copy: {new_name.name} exists")
        ckpt.unlink()
        print(f"[t2.5-random] cleared default-named: {ckpt.name}")

    loss_src = ROOT / "eval_out" / f"continued_pretrain_t2_{args.attention}_loss.json"
    loss_dst = ROOT / "eval_out" / f"continued_pretrain_t2_5_random_{args.attention}_loss.json"
    if loss_src.exists() and not loss_dst.exists():
        shutil.copy2(loss_src, loss_dst)
        loss_src.unlink()
        print(f"[t2.5-random] moved loss log → {loss_dst.name}")


if __name__ == "__main__":
    main()
