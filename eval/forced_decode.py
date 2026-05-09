"""Forced-anchor diffusion decoding.

Same masked-diffusion loop as model.generate(), but with the option to
pre-fill specific answer-slot positions with tokens from a retrieved
LTMi-XT locus. Locked positions are excluded from is_masked at init,
so the iterative unmasking only fills the gaps around them.

Inference-only — no weight updates. The broken model only has to
predict connective tissue between known-correct anchor tokens.
"""
from __future__ import annotations
import json
from pathlib import Path

import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────
# LTMi-XT bundle retrieval (keyword overlap, no LLM)
# ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    "the","a","an","is","are","was","were","be","been","being","of","to","in","on","at","by",
    "for","with","as","and","or","but","not","this","that","these","those","it","its","what",
    "which","who","whom","how","why","do","does","did","done","have","has","had","can","could",
    "will","would","should","may","might","i","you","they","them","their","our","my","me",
    "company","companies","each","also","into","from","over","than",
}

import re

def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower())
            if t and t not in STOPWORDS and len(t) >= 2}


def load_bundle(path: str | Path) -> list[dict]:
    """Returns list of loci with breadcrumb + statement."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return data["loci"]


def retrieve_top1(loci: list[dict], query: str) -> dict | None:
    """Best locus for query by keyword overlap on (breadcrumb + statement)."""
    qt = _tokens(query)
    if not qt:
        return None
    best, best_score = None, -1.0
    for l in loci:
        haystack = " ".join(l["breadcrumb"]) + " " + l["statement"]
        ht = _tokens(haystack)
        if not ht:
            continue
        score = len(qt & ht) / len(qt)
        if score > best_score:
            best_score, best = score, l
    return best


# ──────────────────────────────────────────────────────────────────
# Forced-anchor decoding
# ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_with_anchors(
    model,
    prompt_ids: torch.LongTensor,
    answer_len: int,
    locked_positions: dict[int, int] | None = None,
    *,
    num_steps: int = 12,
    temperature: float = 0.8,
    top_p: float = 0.9,
    beta: float = 0.5,
    rep_penalty: float = 1.3,
    timestep_power: float = 1.5,
):
    """
    Args:
        prompt_ids: [1, P] prompt tokens (Q: ... A:)
        answer_len: number of answer slots to fill
        locked_positions: {pos_in_answer_slot: token_id}; these stay fixed.

    Returns:
        seq: [1, P+answer_len] full sequence (locked positions preserved)
        was_masked_init: [answer_len] bool mask of which positions were initially mask
    """
    cfg = model.config
    device = prompt_ids.device
    bsz = prompt_ids.shape[0]
    prompt_len = prompt_ids.shape[1]
    locked_positions = locked_positions or {}

    # Build answer slot: locked positions get their token, rest get mask_token_id
    answer_slot = torch.full((bsz, answer_len), cfg.mask_token_id, dtype=torch.long, device=device)
    for pos, tok_id in locked_positions.items():
        if 0 <= pos < answer_len:
            answer_slot[:, pos] = tok_id

    seq = torch.cat([prompt_ids, answer_slot], dim=1)
    attention_mask = torch.ones_like(seq)
    is_masked = seq == cfg.mask_token_id  # locked positions are NOT mask
    initial_mask = is_masked.clone()

    placed_counts = torch.zeros(bsz, cfg.vocab_size, device=device)

    for step in range(num_steps):
        t_val = (1.0 - step / max(num_steps - 1, 1)) ** timestep_power
        t = torch.full((bsz,), t_val, device=device, dtype=torch.float32)

        logits = model.forward(seq, attention_mask, t=t)

        if rep_penalty > 1.0 and placed_counts.sum() > 0:
            scale = 1.0 + (rep_penalty - 1.0) * torch.clamp(placed_counts, 0, 3) / 3.0
            logits = logits - scale.unsqueeze(1).log() * (scale > 1).float().unsqueeze(1)

        scaled = logits / max(temperature, 1e-3)
        sorted_logits, sorted_idx = scaled.sort(descending=True, dim=-1)
        cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        cutoff = cum_probs > top_p
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        probs = F.softmax(sorted_logits, dim=-1)
        flat = probs.view(-1, probs.size(-1))
        sampled_sorted = torch.multinomial(flat, 1).view(*probs.shape[:-1], 1)
        sampled = sorted_idx.gather(-1, sampled_sorted).squeeze(-1)

        conf = model._confidence_from_logits(logits)
        conf = torch.where(is_masked, conf, torch.full_like(conf, -1.0))

        if step == num_steps - 1:
            seq = torch.where(is_masked, sampled, seq)
            for b in range(bsz):
                for v in sampled[b][is_masked[b]].unique():
                    placed_counts[b, v] += (sampled[b][is_masked[b]] == v).sum().float()
            is_masked = torch.zeros_like(is_masked)
            break

        for b in range(bsz):
            c = conf[b][is_masked[b]]
            if c.numel() == 0:
                continue
            mu, sigma = c.mean(), c.std().clamp_min(1e-6)
            thresh = mu + beta * sigma
            to_commit_mask = (conf[b] >= thresh) & is_masked[b]
            if to_commit_mask.sum() == 0:
                best = conf[b].argmax()
                to_commit_mask = torch.zeros_like(is_masked[b])
                to_commit_mask[best] = True
            for pos in to_commit_mask.nonzero(as_tuple=True)[0]:
                v = sampled[b, pos].item()
                seq[b, pos] = v
                placed_counts[b, v] += 1
            is_masked[b] = is_masked[b] & ~to_commit_mask

    return seq[:, prompt_len:], initial_mask[:, prompt_len:]
