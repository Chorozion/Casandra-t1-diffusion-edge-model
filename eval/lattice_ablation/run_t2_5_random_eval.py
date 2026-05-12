"""T2.5-random vs T2 BLAKE2b vs v1.5 — held-out comparison.

Final lattice-channel ablation. Trains and evaluates with random per-locus
coords (fresh uniform sample, deterministic seed). If random-coord arm has
same downstream metrics as BLAKE2b arm, the lattice channel is empirically
content-free — only the existence of a deterministic per-locus mapping
matters, not the mapping itself.

Mirrors run_t2_5_pca_eval.py — same monkey-patch pattern, different coords.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path("D:/cassandra-eval")
RANDOM_COORDS_PATH = ROOT / "lattice_coords_random.json"

print(f"[t2.5-random-eval] loading random coords from {RANDOM_COORDS_PATH}...")
RANDOM_DATA = json.loads(RANDOM_COORDS_PATH.read_text(encoding="utf-8"))
RANDOM_COORDS = RANDOM_DATA["coords"]
print(f"  loaded {len(RANDOM_COORDS)} locus → random-coord mappings "
      f"(seed={hex(RANDOM_DATA['seed'])})")

sys.path.insert(0, str(Path(__file__).parent))
import run_t2_multidomain_eval as eval_mod

_original_locus_lattice = eval_mod._locus_lattice
_USE_RANDOM_FOR_CURRENT_ARM = False


def random_aware_locus_lattice(locus):
    if _USE_RANDOM_FOR_CURRENT_ARM:
        bc = locus.get("breadcrumb") or []
        key = "/".join(bc).lower()
        if key in RANDOM_COORDS:
            x, y, z = RANDOM_COORDS[key]
            return (int(x), int(y), int(z))
    return _original_locus_lattice(locus)


eval_mod._locus_lattice = random_aware_locus_lattice
print(f"[t2.5-random-eval] monkey-patched _locus_lattice → random-aware (toggle per arm)")


def main():
    import torch
    from cassandra_loader import load

    print(f"\n{'='*72}\nT2.5-RANDOM vs T2 BLAKE2b — LATTICE-CHANNEL ABLATION\n{'='*72}")
    print(f"  bundles : {list(eval_mod.BUNDLES.keys())}")
    print(f"  queries : {eval_mod.QUERIES_EXT.name}")
    torch.manual_seed(eval_mod.SEED)

    log_path = eval_mod.OUT_DIR / "t2_5_random_comparison.jsonl"
    log_f = log_path.open("w", encoding="utf-8")
    all_rows = []

    arms = ["v1.5", "v2_ltmi_triple", "v2_5_random"]
    for arm in arms:
        print(f"\n{'-'*72}\nARM: {arm}\n{'-'*72}")
        global _USE_RANDOM_FOR_CURRENT_ARM
        _USE_RANDOM_FOR_CURRENT_ARM = (arm == "v2_5_random")
        print(f"  use_random_coords: {_USE_RANDOM_FOR_CURRENT_ARM}")

        model, tok, _ = load(arm, dtype=torch.bfloat16)
        model.eval()
        supply_ltmi = bool(getattr(model.config, "attention_use_ltmi_priors", False))
        print(f"  supply_ltmi_priors: {supply_ltmi}")
        rows = eval_mod.evaluate_checkpoint(model, tok, arm,
                                             supply_ltmi_priors=supply_ltmi)
        all_rows.extend(rows)
        for r in rows:
            log_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            log_f.flush()
        del model
        torch.cuda.empty_cache()

    log_f.close()

    summary = eval_mod.summarize(all_rows)
    print(f"\n{'='*72}\nT2.5-RANDOM SUMMARY\n{'='*72}")
    print(f"{'arm':<22}{'corpus':<8}{'force':<10}{'n':<5}"
          f"{'overlap':<10}{'eng_ratio':<11}{'anchor_pres':<12}")
    for arm in arms:
        for corp in ("C5", "C6", "C7", "ALL"):
            for force_label in ("forced", "unforced"):
                key = f"{arm}/{corp}/{force_label}"
                if key in summary:
                    s = summary[key]
                    overlap = s.get("corpus_overlap_avg", 0.0)
                    eng = s.get("english_ratio_avg", 0.0)
                    anchor = s.get("anchor_preservation_avg")
                    n = s.get("n", 0)
                    anchor_s = f"{anchor:.3f}" if isinstance(anchor, float) else "—"
                    print(f"{arm:<22}{corp:<8}{force_label:<10}{n:<5}"
                          f"{overlap:<10.3f}{eng:<11.3f}{anchor_s:<12}")

    out_path = eval_mod.OUT_DIR / "t2_5_random_comparison_summary.json"
    out_path.write_text(json.dumps({
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arms": arms,
        "summary": summary,
    }, indent=2), encoding="utf-8")
    print(f"\n[t2.5-random-eval] summary → {out_path}")


if __name__ == "__main__":
    main()
