"""
Sophia T1 — Masked Diffusion Language Model  (Q3-enhanced)

Core architecture unchanged from baseline:
  - RoPE positional encoding (128K context)
  - Grouped Query Attention (GQA) with sliding window + global tokens
  - SwiGLU FFN
  - RMSNorm

Q3 additions (all OFF by default → ep5 ckpt loads identically):
  - AdaLN: per-layer timestep-conditioned scale+shift (zero-init → no-op at load)
  - Learned confidence head (small MLP on final hidden state)
  - Span masking + mask-ratio curriculum in compute_diffusion_loss
  - Quantile-adaptive generate() (replaces fixed-topk unmasking)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import SophiaT1Config


# ────────────────────────────────────────────────────────────
#  Normalization + position
# ────────────────────────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


def rotary_embedding(dim: int, seq_len: int, theta: float = 500000.0, device=None):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rotary(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs[:x.shape[-2], :]
    freqs = freqs.unsqueeze(0).unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).flatten(-2)
    return out.type_as(x)


# ────────────────────────────────────────────────────────────
#  Q3 additions — timestep embedding + AdaLN
# ────────────────────────────────────────────────────────────
class TimestepEmbedding(nn.Module):
    """Sinusoidal embedding → MLP → per-layer conditioning signal."""

    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: [B] float in [0, 1]
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)  # [B, hidden]


class AdaLN(nn.Module):
    """
    Adaptive LayerNorm: h = RMSNorm(x) * (1 + scale(t)) + shift(t)
    Final linear is ZERO-INIT so at load time this is an exact no-op;
    training lets the conditioning emerge.
    """

    def __init__(self, hidden_size: int, cond_dim: int, rms_eps: float = 1e-6):
        super().__init__()
        self.norm = RMSNorm(hidden_size, rms_eps)
        self.proj = nn.Linear(cond_dim, 2 * hidden_size)
        # Zero-init → scale=0, shift=0 → output = RMSNorm(x) (identical to baseline)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor]) -> torch.Tensor:
        x = self.norm(x)
        if t_emb is None:
            return x
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        # Broadcast over sequence: [B, 1, H]
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ────────────────────────────────────────────────────────────
#  Attention + FFN (unchanged)
# ────────────────────────────────────────────────────────────
class SlidingWindowAttention(nn.Module):
    def __init__(self, config: SophiaT1Config):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.sliding_window = config.sliding_window
        self.global_tokens = config.global_tokens

        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

        self.kv_repeat = config.num_heads // config.num_kv_heads

    def forward(self, x, freqs, mask=None):
        bsz, seq_len, _ = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = apply_rotary(q, freqs)
        k = apply_rotary(k, freqs)

        if self.kv_repeat > 1:
            k = k.repeat_interleave(self.kv_repeat, dim=1)
            v = v.repeat_interleave(self.kv_repeat, dim=1)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        attn = attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.o_proj(attn)


class SwiGLU(nn.Module):
    def __init__(self, config: SophiaT1Config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Pre-norm block. When use_adaln=True, RMSNorms become AdaLN conditioned on t."""

    def __init__(self, config: SophiaT1Config):
        super().__init__()
        self.use_adaln = config.use_adaln
        if self.use_adaln:
            self.attn_norm = AdaLN(config.hidden_size, config.timestep_embed_dim, config.rms_norm_eps)
            self.ffn_norm  = AdaLN(config.hidden_size, config.timestep_embed_dim, config.rms_norm_eps)
        else:
            self.attn_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
            self.ffn_norm  = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attn = SlidingWindowAttention(config)
        self.ffn  = SwiGLU(config)

    def forward(self, x, freqs, mask=None, t_emb=None):
        if self.use_adaln:
            x = x + self.attn(self.attn_norm(x, t_emb), freqs, mask)
            x = x + self.ffn(self.ffn_norm(x, t_emb))
        else:
            x = x + self.attn(self.attn_norm(x), freqs, mask)
            x = x + self.ffn(self.ffn_norm(x))
        return x


# ────────────────────────────────────────────────────────────
#  Model
# ────────────────────────────────────────────────────────────
class SophiaT1Model(nn.Module):
    def __init__(self, config: SophiaT1Config):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Q3 additions (optional)
        if config.use_adaln:
            self.t_embed = TimestepEmbedding(config.timestep_embed_dim, config.timestep_embed_dim)
        if config.use_confidence_head:
            # Small MLP: 4 features → 1 confidence score
            # features: [max_prob, entropy, margin, logit_argmax]
            self.conf_head = nn.Sequential(
                nn.Linear(4, 32),
                nn.SiLU(),
                nn.Linear(32, 1),
            )
            # Zero-init final layer → starts as constant ~0.5 after sigmoid
            nn.init.zeros_(self.conf_head[-1].weight)
            nn.init.zeros_(self.conf_head[-1].bias)

        self.apply(self._init_weights)
        # Re-zero AdaLN projections after global init (belt-and-suspenders)
        if config.use_adaln:
            for layer in self.layers:
                nn.init.zeros_(layer.attn_norm.proj.weight)
                nn.init.zeros_(layer.attn_norm.proj.bias)
                nn.init.zeros_(layer.ffn_norm.proj.weight)
                nn.init.zeros_(layer.ffn_norm.proj.bias)
            nn.init.zeros_(self.conf_head[-1].weight) if config.use_confidence_head else None

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)

    # ─── Forward ─────────────────────────────────────────────
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        t: Optional[torch.Tensor] = None,           # [B] diffusion step in [0,1]
        return_hidden: bool = False,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        x = self.embed(input_ids)
        freqs = rotary_embedding(self.config.head_dim, seq_len, self.config.rope_theta, device)

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].bool()
            attn_mask = attn_mask.expand(bsz, 1, seq_len, seq_len)
            attn_mask = torch.where(attn_mask, 0.0, float('-inf')).to(x.dtype)

        # Timestep embedding
        t_emb = None
        if self.config.use_adaln and t is not None:
            t_emb = self.t_embed(t)

        for layer in self.layers:
            x = layer(x, freqs, attn_mask, t_emb)

        x = self.norm(x)

        if self.lm_head is not None:
            logits = self.lm_head(x)
        else:
            logits = F.linear(x, self.embed.weight)

        if return_hidden:
            return logits, x
        return logits

    # ─── Q3 features → confidence ────────────────────────────
    @torch.no_grad()
    def _confidence_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Four-feature composite confidence. [B, S, V] → [B, S]"""
        probs = F.softmax(logits, dim=-1)
        top2 = probs.topk(2, dim=-1).values
        max_prob = top2[..., 0]
        margin = top2[..., 0] - top2[..., 1]
        entropy = -(probs.clamp_min(1e-9) * probs.clamp_min(1e-9).log()).sum(-1)
        max_logit = logits.max(dim=-1).values
        if self.config.use_confidence_head and hasattr(self, "conf_head"):
            feats = torch.stack([max_prob, entropy, margin, max_logit / 10.0], dim=-1)
            return torch.sigmoid(self.conf_head(feats).squeeze(-1))
        # Hand-crafted composite (outperforms pure max-prob even untrained)
        return 0.5 * max_prob + 0.3 * margin + 0.2 * (1.0 - entropy / math.log(self.config.vocab_size))

    # ─── Training loss (span masking + curriculum + SNR) ─────
    def compute_diffusion_loss(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        mask_ratio: Optional[float] = None,
        step_frac: float = 1.0,
        use_snr_weight: bool = False,
    ) -> torch.Tensor:
        """
        Masked-diffusion CE loss.
        - span_mask_prob of batch elements get contiguous-span masking.
        - mask_ratio curriculum: if step_frac<1.0, clamp range tighter early.
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device
        cfg = self.config

        # Curriculum: first 10% of training → ratio in (0, 0.6); else (min, max)
        if step_frac < 0.1:
            r_lo, r_hi = cfg.mask_ratio_min, min(cfg.mask_ratio_max, 0.6)
        else:
            r_lo, r_hi = cfg.mask_ratio_min, cfg.mask_ratio_max

        if mask_ratio is None:
            mask_ratio = r_lo + (r_hi - r_lo) * torch.rand(1, device=device).item()

        # ---- Build mask per sample: span OR random ----
        rand_mask = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
        use_span = torch.rand(bsz, device=device) < cfg.span_mask_prob
        # Random-mask portion
        pure_random = torch.rand(bsz, seq_len, device=device) < mask_ratio
        rand_mask = torch.where(use_span.unsqueeze(-1), rand_mask, pure_random)

        # Span-mask portion
        if use_span.any():
            # Geometric spans with mean=cfg.span_mean_length
            p_geom = 1.0 / cfg.span_mean_length
            target_total = int(seq_len * mask_ratio)
            for b in range(bsz):
                if not use_span[b]:
                    continue
                masked_so_far = 0
                while masked_so_far < target_total:
                    span_len = max(1, int(torch.distributions.Geometric(
                        torch.tensor(p_geom, device=device)).sample().item()) + 1)
                    start = torch.randint(0, max(1, seq_len - span_len), (1,), device=device).item()
                    rand_mask[b, start:start + span_len] = True
                    masked_so_far += span_len

        if attention_mask is not None:
            rand_mask = rand_mask & attention_mask.bool()

        targets = input_ids.clone()
        masked_ids = input_ids.clone()
        masked_ids[rand_mask] = cfg.mask_token_id

        # Timestep = fraction of tokens masked (Q3: γ(t) = mask ratio)
        gamma = rand_mask.float().mean(dim=-1).clamp(0.05, 0.95)  # [B]
        t = gamma  # just the mask fraction per sample

        logits = self.forward(masked_ids, attention_mask, t=t)

        flat_logits = logits[rand_mask]
        flat_targets = targets[rand_mask]

        if flat_logits.numel() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        per_tok = F.cross_entropy(flat_logits, flat_targets, reduction="none")

        if use_snr_weight:
            # Per-sample weight w(t) = (1-γ)/γ, broadcast to masked positions
            # rm is [B,S]; map each masked position to its sample's weight
            sample_idx = torch.arange(bsz, device=device).unsqueeze(1).expand(bsz, seq_len)[rand_mask]
            snr_w = ((1.0 - gamma) / gamma.clamp_min(1e-3))[sample_idx]
            snr_w = snr_w / snr_w.mean().clamp_min(1e-6)  # normalize
            return (per_tok * snr_w).mean()

        return per_tok.mean()

    # ─── Generation: quantile-adaptive unmasking ─────────────
    @torch.no_grad()
    def generate(
        self,
        prompt_ids: torch.LongTensor,
        max_new_tokens: int = 128,
        num_steps: int = 12,
        temperature: float = 0.8,
        top_p: float = 0.9,
        beta: Optional[float] = None,          # quantile offset τ = μ + β·σ
        rep_penalty: float = 1.3,
        timestep_power: float = 1.5,           # α for step→t mapping (Q3 §3.2.4)
    ) -> torch.LongTensor:
        """
        Q3 quantile-adaptive unmasking:
          - Compute confidence at every masked position.
          - Commit positions whose confidence ≥ μ + β·σ (adaptive, not fixed topk).
          - Commit everything on final step.
          - Bonus: rep_penalty on previously placed tokens.
          - Timestep passed to model (if use_adaln) via α-power schedule.
        """
        device = prompt_ids.device
        bsz = prompt_ids.shape[0]
        prompt_len = prompt_ids.shape[1]
        if beta is None:
            beta = self.config.unmask_beta

        mask_tokens = torch.full(
            (bsz, max_new_tokens), self.config.mask_token_id,
            dtype=torch.long, device=device,
        )
        seq = torch.cat([prompt_ids, mask_tokens], dim=1)
        attention_mask = torch.ones_like(seq)
        is_masked = seq == self.config.mask_token_id

        placed_counts = torch.zeros(bsz, self.config.vocab_size, device=device)

        for step in range(num_steps):
            # α-power timestep: more iterations spent at high-noise regimes
            t_val = (1.0 - step / max(num_steps - 1, 1)) ** timestep_power
            t = torch.full((bsz,), t_val, device=device, dtype=torch.float32)

            logits = self.forward(seq, attention_mask, t=t)

            # Repetition penalty: downweight already-placed tokens
            if rep_penalty > 1.0 and placed_counts.sum() > 0:
                scale = 1.0 + (rep_penalty - 1.0) * torch.clamp(placed_counts, 0, 3) / 3.0
                logits = logits - scale.unsqueeze(1).log() * (scale > 1).float().unsqueeze(1)

            # Top-p nucleus + temperature sampling
            scaled = logits / max(temperature, 1e-3)
            sorted_logits, sorted_idx = scaled.sort(descending=True, dim=-1)
            cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Keep tokens until cum_probs >= top_p
            cutoff = cum_probs > top_p
            # Don't drop the first token
            cutoff[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
            # Sample
            probs = F.softmax(sorted_logits, dim=-1)
            flat = probs.view(-1, probs.size(-1))
            sampled_sorted = torch.multinomial(flat, 1).view(*probs.shape[:-1], 1)
            sampled = sorted_idx.gather(-1, sampled_sorted).squeeze(-1)   # [B, S]
            sample_probs = probs.gather(-1, sampled_sorted).squeeze(-1)  # [B, S]

            # Confidence
            conf = self._confidence_from_logits(logits)  # [B, S]
            conf = torch.where(is_masked, conf, torch.full_like(conf, -1.0))

            if step == num_steps - 1:
                # Final: commit everything still masked
                seq = torch.where(is_masked, sampled, seq)
                # Update placed tracking
                for b in range(bsz):
                    for v in sampled[b][is_masked[b]].unique():
                        placed_counts[b, v] += (sampled[b][is_masked[b]] == v).sum().float()
                is_masked = torch.zeros_like(is_masked)
                break

            # Adaptive quantile threshold per sample
            for b in range(bsz):
                c = conf[b][is_masked[b]]
                if c.numel() == 0:
                    continue
                mu, sigma = c.mean(), c.std().clamp_min(1e-6)
                thresh = mu + beta * sigma
                # Always commit at least one to avoid stalls
                to_commit_mask = (conf[b] >= thresh) & is_masked[b]
                if to_commit_mask.sum() == 0:
                    # Commit the single best
                    best = conf[b].argmax()
                    to_commit_mask = torch.zeros_like(is_masked[b])
                    to_commit_mask[best] = True
                # Apply
                positions = to_commit_mask.nonzero(as_tuple=True)[0]
                for pos in positions:
                    v = sampled[b, pos].item()
                    seq[b, pos] = v
                    placed_counts[b, v] += 1
                is_masked[b] = is_masked[b] & ~to_commit_mask

        return seq[:, prompt_len:]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def remap_baseline_state_dict(state: dict, use_adaln: bool) -> dict:
        """
        Remap a baseline (non-AdaLN) ep5 checkpoint into the AdaLN-wrapped model.
        Old key:  layers.N.attn_norm.weight
        New key:  layers.N.attn_norm.norm.weight   (when use_adaln=True)
        Same for ffn_norm.
        Safe to call on Q3 state dicts too — it only rewrites keys that match the old pattern.
        """
        if not use_adaln:
            return state
        import re
        out = {}
        pat = re.compile(r"^(layers\.\d+\.)(attn_norm|ffn_norm)\.weight$")
        for k, v in state.items():
            m = pat.match(k)
            if m:
                out[f"{m.group(1)}{m.group(2)}.norm.weight"] = v
            else:
                out[k] = v
        return out


if __name__ == "__main__":
    from .config import sophia_t1_small, sophia_t1_base, sophia_t1_base_q3

    # Baseline (ep5 compatible)
    print("=" * 50)
    cfg = sophia_t1_small()
    m = SophiaT1Model(cfg)
    print(f"T1-Small baseline: {m.count_parameters() / 1e6:.1f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 128))
    logits = m(x)
    print(f"  Forward: {x.shape} -> {logits.shape}")
    loss = m.compute_diffusion_loss(x)
    print(f"  Loss: {loss.item():.4f}")

    # Q3-enhanced
    print("=" * 50)
    cfg_q3 = sophia_t1_small()
    cfg_q3.use_adaln = True
    cfg_q3.use_confidence_head = True
    m_q3 = SophiaT1Model(cfg_q3)
    print(f"T1-Small + Q3: {m_q3.count_parameters() / 1e6:.1f}M (extra = AdaLN + conf head)")
    t = torch.rand(2)
    logits_q3 = m_q3(x, t=t)
    print(f"  Forward with t: {x.shape} -> {logits_q3.shape}")
    loss_q3 = m_q3.compute_diffusion_loss(x, step_frac=0.5)
    print(f"  Loss: {loss_q3.item():.4f}")
