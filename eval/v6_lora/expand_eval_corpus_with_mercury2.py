"""Phase 0: expand eval corpus to n>=200 by generating diverse held-out
queries via Mercury 2 (Inception API).

For each existing locus in C5/C6/C7, generate 4 alternative-phrasing
queries that target the locus's statement WITHOUT leaking the exact
expected keywords. Append to existing 12 per corpus → target 60 per
corpus, 180 total over C5+C6+C7. Add 1 new corpus (C8) for topic
diversity.

Output: queries_heldout_extended_v2.json (back-compat: includes original
12 per corpus plus the new queries).

Leakage discipline:
  - Mercury 2 sees the STATEMENT, breadcrumb, expected_keywords. It is
    INSTRUCTED to generate questions whose answers contain the keywords
    but whose question text does NOT contain them verbatim.
  - Each generated query is filtered against the expected_keywords list;
    any overlap above 1 keyword is rejected (forces real generalization).
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = Path("D:/cassandra-eval")
# Inception Labs API key — set via environment variable.
# Get a key at https://platform.inceptionlabs.ai/
API_KEY = os.environ.get("INCEPTION_API_KEY", "")
URL = "https://api.inceptionlabs.ai/v1/chat/completions"

if not API_KEY:
    print(
        "[expand] ERROR: INCEPTION_API_KEY environment variable is required.\n"
        "  Set it before running:\n"
        "    PowerShell:  $env:INCEPTION_API_KEY = 'sk_...'\n"
        "    bash:        export INCEPTION_API_KEY=sk_...",
        file=sys.stderr,
    )
    sys.exit(1)
EXISTING = ROOT / "queries_heldout_extended.json"
OUTPUT = ROOT / "queries_heldout_extended_v2.json"

QUERIES_PER_LOCUS_TARGET = 4  # 12 loci × 4 new = 48, +12 existing = 60 per corpus
CORPORA = ["C5", "C6", "C7"]


SYSTEM_PROMPT = "You output JSON only."


def query_mercury(messages, max_tokens=1200):
    body = {
        "model": "mercury-2",
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        URL,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def parse_json_block(text: str) -> dict | None:
    """Extract first balanced {...} JSON object from text. Handles nested
    brackets inside arrays."""
    if not text:
        return None
    # Try direct parse first (the model usually outputs clean JSON)
    text_stripped = text.strip()
    if text_stripped.startswith("```"):
        # Strip markdown fences
        text_stripped = re.sub(r"^```(?:json)?\s*", "", text_stripped)
        text_stripped = re.sub(r"\s*```$", "", text_stripped)
    try:
        return json.loads(text_stripped)
    except json.JSONDecodeError:
        pass
    # Balanced-bracket scan: find first `{`, then track depth
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def queries_overlap_keywords(query: str, keywords: list[str]) -> int:
    """Count how many expected keywords appear (case-insensitive substring)
    in the generated query. Aggressive filter — anything > 1 rejected."""
    q_lower = query.lower()
    hits = 0
    for kw in keywords:
        kw_lower = kw.lower().strip()
        if len(kw_lower) < 3:
            continue  # skip ultra-short tokens
        if kw_lower in q_lower:
            hits += 1
    return hits


def generate_for_locus(bundle_id: str, locus: dict, expected_keywords: list[str]):
    user_prompt = (
        f"Generate 4 diverse natural-language questions whose correct answer is "
        f"the following fact. Vary the phrasing across the four questions. "
        f"Avoid using the exact answer keywords in the question text where possible.\n\n"
        f"Fact: {locus.get('statement', '')}\n"
        f"Topic: {' > '.join(locus.get('breadcrumb', []))}\n"
        f"Answer keywords to avoid in question (best-effort): {expected_keywords}\n\n"
        f'Output exactly this JSON structure: {{"queries": ["q1", "q2", "q3", "q4"]}}'
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    result = query_mercury(messages)
    if "choices" not in result:
        return None, f"no choices: {json.dumps(result)[:200]}"
    text = result["choices"][0]["message"]["content"]
    parsed = parse_json_block(text)
    if parsed is None or "queries" not in parsed:
        return None, f"failed to parse: {text[:200]}"
    qs = parsed["queries"]
    # Filter for leakage
    accepted = []
    rejected = []
    for q in qs:
        if not isinstance(q, str) or len(q) < 8:
            rejected.append((q, "too short"))
            continue
        hits = queries_overlap_keywords(q, expected_keywords)
        if hits > 1:
            rejected.append((q, f"keyword overlap {hits}"))
            continue
        accepted.append(q.strip())
    return accepted, rejected


def main():
    existing = json.loads(EXISTING.read_text(encoding="utf-8"))
    print(f"[expand] starting; existing queries per corpus: "
          f"{ {k: len(v) for k, v in existing.items()} }")

    bundles = {}
    for cid in CORPORA:
        bundles[cid] = json.loads((ROOT / f"{cid}.json").read_text(encoding="utf-8"))

    expanded = {cid: list(existing.get(cid, [])) for cid in CORPORA}
    metadata = {"generated_per_locus": [], "rejected": []}

    for cid in CORPORA:
        print(f"\n[expand] --- corpus {cid} ---")
        loci = bundles[cid]["loci"]
        existing_queries_in_corpus = {q["q"] for q in expanded[cid]}

        for i, locus in enumerate(loci):
            # Find existing query that targets this locus to source expected_keywords
            existing_q_for_locus = None
            for eq in existing.get(cid, []):
                # Use breadcrumb final concept as proxy for "targets this locus"
                bc_final = (locus.get("breadcrumb") or ["?"])[-1].lower()
                if bc_final in eq["q"].lower() or any(
                    kw.lower() in eq["q"].lower()
                    for kw in eq.get("expect_keywords", [])
                ):
                    existing_q_for_locus = eq
                    break
            expected = (
                existing_q_for_locus["expect_keywords"]
                if existing_q_for_locus
                else []
            )
            if not expected:
                # Derive keywords from breadcrumb + first 3 nouns of statement
                bc = locus.get("breadcrumb", [])
                expected = [bc[-1]] if bc else []

            print(f"  locus {i+1}/{len(loci)} bc={locus.get('breadcrumb', ['?'])[-1]}", end=" ")
            try:
                accepted, rejected = generate_for_locus(cid, locus, expected)
            except urllib.error.HTTPError as e:
                err_body = e.read().decode(errors="replace")[:200]
                print(f"HTTP {e.code}: {err_body}")
                continue
            except Exception as e:
                print(f"FAIL {type(e).__name__}: {e}")
                continue

            if accepted is None:
                print(f"PARSE_FAIL: {rejected[:150]}")
                continue

            added = 0
            for q in accepted:
                if q in existing_queries_in_corpus:
                    continue
                expanded[cid].append({
                    "q": q,
                    "expect_keywords": expected,
                    "source_locus_id": locus.get("id"),
                    "generated_by": "mercury-2",
                })
                existing_queries_in_corpus.add(q)
                added += 1
            metadata["generated_per_locus"].append({
                "corpus": cid,
                "locus_id": locus.get("id"),
                "accepted": len(accepted),
                "added": added,
                "rejected_count": len(rejected) if isinstance(rejected, list) else 0,
            })
            print(f"+{added} queries")

        print(f"[expand] {cid} now has {len(expanded[cid])} queries")

    out = {**expanded, "_metadata": metadata, "_generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    OUTPUT.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[expand] saved → {OUTPUT}")
    print(f"  totals: {{ {', '.join(f'{c}: {len(expanded[c])}' for c in CORPORA)} }}")
    print(f"  grand total queries: {sum(len(expanded[c]) for c in CORPORA)}")


if __name__ == "__main__":
    main()
