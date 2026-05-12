"""Bootstrap CI on T2.5-random vs T2 BLAKE2b vs v1.5.

Reads t2_5_random_comparison.jsonl, computes paired-by-query bootstrap on:
  v2_5_random − v2_ltmi_triple   (THE load-bearing question: does the
                                   specific coord-to-locus mapping matter?)
  v2_5_random − v1.5             (any T2 effect at all)
  v2_ltmi_triple − v1.5          (replicates prior T2 result)

If v2_5_random ≈ v2_ltmi_triple on all metrics: lattice channel content-free.

Output: t2_5_random_bootstrap_summary.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path("D:/cassandra-eval")
LOG_PATH = ROOT / "eval_out" / "t2_5_random_comparison.jsonl"
OUT_PATH = ROOT / "eval_out" / "t2_5_random_bootstrap_summary.json"
N_BOOT = 2000
SEED = 0xC0FFEE


def load_rows():
    return [json.loads(line) for line in LOG_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def by_arm_metric(rows, arm, force_label, metric):
    out = {}
    want_forced = (force_label == "forced")
    for r in rows:
        if r.get("arm") != arm:
            continue
        if bool(r.get("force_anchor", False)) != want_forced:
            continue
        key = (r.get("corpus"), r.get("query"))
        val = r.get(metric)
        if val is not None:
            out[key] = float(val)
    return out


def bootstrap_paired_diff(a_dict, b_dict, n_boot=N_BOOT, seed=SEED):
    common_keys = sorted(set(a_dict.keys()) & set(b_dict.keys()))
    if not common_keys:
        return None
    a = [a_dict[k] for k in common_keys]
    b = [b_dict[k] for k in common_keys]
    n = len(a)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        sa = sum(a[i] for i in idx) / n
        sb = sum(b[i] for i in idx) / n
        diffs.append(sa - sb)
    diffs.sort()
    point = sum(a) / n - sum(b) / n
    lo = diffs[int(0.025 * n_boot)]
    hi = diffs[int(0.975 * n_boot)]
    p_above_zero = sum(1 for d in diffs if d > 0) / n_boot
    return {
        "n_paired": n,
        "point": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "p_above_zero": p_above_zero,
    }


def main():
    print(f"[bootstrap-random] loading {LOG_PATH}...")
    rows = load_rows()
    print(f"  {len(rows)} rows")

    arms = ["v1.5", "v2_ltmi_triple", "v2_5_random"]
    force_labels = ["forced", "unforced"]
    metrics = ["corpus_overlap", "english_ratio"]

    print("\n[bootstrap-random] per-arm means:")
    for arm in arms:
        for fl in force_labels:
            for m in metrics:
                vals = list(by_arm_metric(rows, arm, fl, m).values())
                if vals:
                    mean = sum(vals) / len(vals)
                    print(f"  {arm:<18} {fl:<9} {m:<16} n={len(vals):<4} mean={mean:.4f}")

    pairs = [
        ("v2_5_random", "v2_ltmi_triple"),  # THE QUESTION
        ("v2_5_random", "v1.5"),
        ("v2_ltmi_triple", "v1.5"),
    ]

    summary = {"pairwise": {}}
    print("\n[bootstrap-random] paired-by-query CIs (n_boot=2000):")
    for fl in force_labels:
        for m in metrics:
            for a, b in pairs:
                a_d = by_arm_metric(rows, a, fl, m)
                b_d = by_arm_metric(rows, b, fl, m)
                stats = bootstrap_paired_diff(a_d, b_d)
                if stats is None:
                    continue
                key = f"{fl}/{m}/{a}_minus_{b}"
                summary["pairwise"][key] = stats
                sig = "✓" if stats["ci_lo"] > 0 else ("✗" if stats["ci_hi"] < 0 else "~")
                print(f"  {fl:<9} {m:<14} {a:<18} - {b:<18}  "
                      f"point={stats['point']:+.4f}  "
                      f"CI[{stats['ci_lo']:+.4f}, {stats['ci_hi']:+.4f}]  "
                      f"p>0={stats['p_above_zero']:.3f}  {sig}")

    print("\n" + "=" * 78)
    print("VERDICT — does specific lattice-coord-to-locus mapping matter?")
    print("=" * 78)
    passes = 0
    total = 0
    for fl in force_labels:
        for m in metrics:
            key = f"{fl}/{m}/v2_5_random_minus_v2_ltmi_triple"
            if key in summary["pairwise"]:
                s = summary["pairwise"][key]
                ci_excludes_zero = s["ci_lo"] > 0 or s["ci_hi"] < 0
                total += 1
                if ci_excludes_zero:
                    passes += 1
                print(f"  {fl} {m}: random - BLAKE2b = {s['point']:+.4f}  "
                      f"CI[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}]  "
                      f"differs CI-sig: {'YES' if ci_excludes_zero else 'no'}")
    print(f"\nLattice-mapping-specificity claim: {passes}/{total} metrics CI-sig differ")
    if passes == 0:
        print("VERDICT: random coords give the SAME downstream behavior as BLAKE2b.")
        print("         The lattice channel carries no semantic information in our T2 setup.")
        print("         Only the existence of a deterministic per-locus tag matters.")
    else:
        print(f"VERDICT: random coords DO differ CI-significantly from BLAKE2b on {passes} metric(s).")

    OUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[bootstrap-random] saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
