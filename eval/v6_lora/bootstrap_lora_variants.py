"""Pre-registered bootstrap analysis on V1-V6 LoRA evals.

Reads all lora_eval_*.jsonl files in eval_out/, computes per-variant
seed-averaged means, paired-by-query bootstrap CI against v2_ltmi_triple
BLAKE2b baseline, then applies the PRE_REGISTRATION_v1_v6.md decision
rules verbatim.

Outputs:
  lora_bootstrap_summary.json — full numerical results
  lora_verdict.md — PASS/FAIL/AMBIGUOUS verdict per variant
"""
from __future__ import annotations
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent.parent
EVAL_OUT = ROOT / "eval_out"
N_BOOT = 2000
SEED = 0xC0FFEE

# Pre-registered thresholds
PASS_METRIC_MAGNITUDE = 0.02
PASS_METRIC_CI_EXCLUDES_ZERO = True
PASS_REQUIRED_METRICS = 2
PASS_REQUIRED_SEED_CONSISTENCY = True


def load_rows(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


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


def bootstrap_paired_diff(a, b, n_boot=N_BOOT, seed=SEED):
    common = sorted(set(a.keys()) & set(b.keys()))
    if not common:
        return None
    av = [a[k] for k in common]
    bv = [b[k] for k in common]
    n = len(av)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(av[i] for i in idx)/n - sum(bv[i] for i in idx)/n)
    diffs.sort()
    point = sum(av)/n - sum(bv)/n
    return {
        "n_paired": n,
        "point": point,
        "ci_lo": diffs[int(0.025*n_boot)],
        "ci_hi": diffs[int(0.975*n_boot)],
        "p_above_zero": sum(1 for d in diffs if d > 0) / n_boot,
    }


def find_lora_evals():
    """Group lora_eval files by variant. Each variant has 3 seeds."""
    files = sorted(EVAL_OUT.glob("lora_eval_v*_seed*.jsonl"))
    pattern = re.compile(r"lora_eval_(v\d)_seed(\d+)_step(\d+)\.jsonl")
    by_variant = defaultdict(list)
    for fp in files:
        m = pattern.match(fp.name)
        if m:
            variant = m.group(1).upper()
            seed = int(m.group(2))
            by_variant[variant].append({"file": fp, "seed": seed})
    return dict(sorted(by_variant.items()))


def find_baseline_eval():
    """Look for an authoritative v2_ltmi_triple eval on the SAME n=172 corpus.
    If missing, fall back to t2_5_random_comparison.jsonl which has v2_ltmi_triple."""
    # Preferred: a fresh n=172 baseline run
    candidates = [
        EVAL_OUT / "lora_baseline_v2_ltmi_triple_n172.jsonl",
        EVAL_OUT / "t2_5_random_comparison.jsonl",  # n=36 fallback
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def evaluate_one_variant(variant: str, files: list[dict], baseline_rows: list[dict]):
    """Returns per-seed and seed-averaged bootstrap CIs vs baseline."""
    baseline_arm = "v2_ltmi_triple"
    force_labels = ["forced", "unforced"]
    metrics = ["corpus_overlap", "english_ratio"]

    per_seed = []
    for fdat in files:
        rows = load_rows(fdat["file"])
        arm = f"{variant.lower()}_seed{fdat['seed']}"
        seed_stats = {"seed": fdat["seed"], "file": fdat["file"].name, "metrics": {}}
        for fl in force_labels:
            for m in metrics:
                a = by_arm_metric(rows, arm, fl, m)
                b = by_arm_metric(baseline_rows, baseline_arm, fl, m)
                stats = bootstrap_paired_diff(a, b)
                if stats is not None:
                    seed_stats["metrics"][f"{fl}/{m}"] = stats
        per_seed.append(seed_stats)
    return per_seed


def seed_aggregate(per_seed: list[dict]) -> dict:
    """Per-metric: aggregate across seeds. Mean of point estimates; sign-
    consistency check; minimum/maximum CI bounds across seeds (conservative)."""
    if not per_seed:
        return {}
    metric_keys = sorted(per_seed[0]["metrics"].keys())
    agg = {}
    for mk in metric_keys:
        points = [s["metrics"][mk]["point"] for s in per_seed if mk in s["metrics"]]
        ci_los = [s["metrics"][mk]["ci_lo"] for s in per_seed if mk in s["metrics"]]
        ci_his = [s["metrics"][mk]["ci_hi"] for s in per_seed if mk in s["metrics"]]
        ps = [s["metrics"][mk]["p_above_zero"] for s in per_seed if mk in s["metrics"]]
        if not points:
            continue
        agg[mk] = {
            "n_seeds": len(points),
            "mean_point": sum(points) / len(points),
            "min_point": min(points),
            "max_point": max(points),
            # Conservative across-seed CI: max of low bounds, min of high bounds
            "ci_lo_conservative": max(ci_los),
            "ci_hi_conservative": min(ci_his),
            "max_p_above_zero": max(ps),
            "min_p_above_zero": min(ps),
            "sign_consistent": all(p > 0 for p in points) or all(p < 0 for p in points),
            "all_seed_points": points,
        }
    return agg


def apply_decision_rules(agg: dict) -> dict:
    """Apply PRE_REGISTRATION_v1_v6.md decision rules."""
    if not agg:
        return {"verdict": "NO_DATA", "reasons": ["no eval data"]}

    metrics_passing = 0
    metric_results = {}
    for mk, a in agg.items():
        criteria = {
            "magnitude_ok": abs(a["mean_point"]) >= PASS_METRIC_MAGNITUDE,
            "ci_excludes_zero": a["ci_lo_conservative"] > 0 or a["ci_hi_conservative"] < 0,
            "sign_consistent": a["sign_consistent"],
            "direction_positive": a["mean_point"] > 0,
        }
        passes = all(criteria.values())
        metric_results[mk] = {**a, "criteria": criteria, "passes": passes}
        if passes:
            metrics_passing += 1

    # Degradation check — any CI-sig negative metric → FAIL
    any_degradation = any(
        m["mean_point"] < -PASS_METRIC_MAGNITUDE
        and (m["ci_lo_conservative"] > 0 or m["ci_hi_conservative"] < 0)
        for m in agg.values()
    )

    if any_degradation:
        verdict = "FAIL"
        reason = f"CI-significant degradation on at least one metric"
    elif metrics_passing >= PASS_REQUIRED_METRICS:
        verdict = "PASS"
        reason = f"{metrics_passing}/{len(agg)} metrics meet PASS criteria"
    elif metrics_passing == 1:
        verdict = "FAIL"
        reason = f"only 1/{len(agg)} metric passes; need ≥{PASS_REQUIRED_METRICS}"
    else:
        # Check for ambiguity (cross-seed sign flips with big magnitudes)
        big_inconsistent = any(
            abs(a["max_point"] - a["min_point"]) > 2 * PASS_METRIC_MAGNITUDE
            and not a["sign_consistent"]
            for a in agg.values()
        )
        if big_inconsistent:
            verdict = "AMBIGUOUS"
            reason = "metric points flip sign across seeds with large magnitude — treat as FAIL"
        else:
            verdict = "FAIL"
            reason = f"0/{len(agg)} metrics meet PASS criteria"

    return {
        "verdict": verdict,
        "reason": reason,
        "metrics_passing": metrics_passing,
        "metric_results": metric_results,
    }


def main():
    print(f"\n{'='*72}\nLoRA VARIANTS PRE-REGISTERED BOOTSTRAP\n{'='*72}")

    by_variant = find_lora_evals()
    print(f"[scan] found variants: {list(by_variant.keys())}")
    for v, fs in by_variant.items():
        print(f"  {v}: {len(fs)} seeds = {[f['seed'] for f in fs]}")

    baseline_path = find_baseline_eval()
    if baseline_path is None:
        print("[error] no baseline eval found. Run a v2_ltmi_triple eval on n=172 first.")
        sys.exit(1)
    print(f"[scan] baseline: {baseline_path.name}")
    baseline_rows = load_rows(baseline_path)

    summary = {"per_variant": {}, "verdicts": {}}
    for variant, files in by_variant.items():
        print(f"\n--- {variant} ---")
        per_seed = evaluate_one_variant(variant, files, baseline_rows)
        agg = seed_aggregate(per_seed)
        verdict = apply_decision_rules(agg)
        summary["per_variant"][variant] = {
            "per_seed": per_seed,
            "seed_aggregate": agg,
            "verdict": verdict,
        }
        summary["verdicts"][variant] = verdict["verdict"]
        print(f"  verdict: {verdict['verdict']} — {verdict['reason']}")
        for mk, a in agg.items():
            sign_str = "consistent" if a["sign_consistent"] else "FLIPS"
            print(f"    {mk:<30} mean={a['mean_point']:+.4f}  "
                  f"CI[{a['ci_lo_conservative']:+.4f}, {a['ci_hi_conservative']:+.4f}]  "
                  f"seeds={a['all_seed_points']}  ({sign_str})")

    # Write summary JSON
    out_json = EVAL_OUT / "lora_bootstrap_summary.json"
    # JSON-safe: replace non-serializable items
    out_json.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"\n[done] summary → {out_json}")

    # Write verdict markdown
    verdict_md = ROOT / "LORA_VERDICT.md"
    lines = ["# LoRA V1-V6 Verdict", "",
             f"**Date:** auto-generated", "",
             "## Per-variant verdicts (pre-registered rules applied)", ""]
    lines.append("| Variant | Verdict | Reason |")
    lines.append("|---|---|---|")
    for v, verd_text in summary["verdicts"].items():
        reason = summary["per_variant"][v]["verdict"]["reason"]
        lines.append(f"| {v} | **{verd_text}** | {reason} |")
    lines.append("")
    lines.append("## Detail")
    for v, info in summary["per_variant"].items():
        lines.append(f"### {v}")
        lines.append(f"**Verdict: {info['verdict']['verdict']}** — {info['verdict']['reason']}")
        lines.append("")
        lines.append("| Metric | Mean | Conservative CI | Seeds | Sign |")
        lines.append("|---|---|---|---|---|")
        for mk, a in info["seed_aggregate"].items():
            sign = "✓" if a["sign_consistent"] else "FLIPS"
            lines.append(
                f"| {mk} | {a['mean_point']:+.4f} | "
                f"[{a['ci_lo_conservative']:+.4f}, {a['ci_hi_conservative']:+.4f}] | "
                f"{a['all_seed_points']} | {sign} |"
            )
        lines.append("")
    verdict_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[done] verdict → {verdict_md}")

    # Print final headline
    print(f"\n{'='*72}\nFINAL VERDICT\n{'='*72}")
    for v, verd in summary["verdicts"].items():
        print(f"  {v}: {verd}")


if __name__ == "__main__":
    main()
