"""Precompute uniform-random 3D lattice coords per locus.

For the random-coord ablation: tests whether the SPECIFIC coord-to-locus
mapping matters for T2 downstream perplexity, or whether only the
existence of a deterministic per-locus mapping matters.

Each locus gets a coord sampled uniformly from {0..63}^3, deterministic
per locus (i.e., same coord across all training steps for a given locus).
This is methodologically equivalent to BLAKE2b in protocol (per-locus
deterministic random) but uses a fresh uniform sampler instead of the
breadcrumb-keyed hash.

Output: lattice_coords_random.json with same shape as lattice_coords_pca_all.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path("D:/cassandra-eval")
BUNDLES = [ROOT / f"{cid}.json" for cid in ("C1", "C2", "C3", "C5", "C6", "C7")]
OUT = ROOT / "lattice_coords_random.json"
LATTICE_DIM = 64
SEED = 0xC0FFEE  # fixed seed → deterministic across runs


def locus_id(locus):
    bc = locus.get("breadcrumb") or []
    return "/".join(bc).lower()


def main():
    print(f"[random-coords] start {time.strftime('%H:%M:%S')}  seed={SEED:#x}")
    print(f"  bundles: {[b.name for b in BUNDLES]}")

    all_loci = []
    for bp in BUNDLES:
        bundle = json.loads(bp.read_text(encoding="utf-8"))
        for locus in bundle["loci"]:
            all_loci.append({"bundle": bp.name, "id": locus_id(locus)})
    print(f"  loci: {len(all_loci)}")

    rng = np.random.RandomState(SEED)
    coords = rng.randint(0, LATTICE_DIM, size=(len(all_loci), 3), dtype=np.int64)

    mapping = {L["id"]: [int(c) for c in coords[i]] for i, L in enumerate(all_loci)}

    unique = len(set(tuple(c) for c in coords))
    print(f"  unique cells: {unique}/{len(coords)} ({100*unique/len(coords):.1f}%)")

    out = {
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "lattice_dim": LATTICE_DIM,
        "n_loci": len(all_loci),
        "n_unique_cells": unique,
        "seed": SEED,
        "method": "uniform_random_per_locus_deterministic_with_fixed_seed",
        "coords": mapping,
    }
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[random-coords] saved → {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
