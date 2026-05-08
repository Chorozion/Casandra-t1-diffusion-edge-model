"""
Sophia T1 — Model Configuration
A lightweight masked diffusion language model designed for:
  - 12GB VRAM inference / mobile deployment
  - 128K context via sliding window + global attention
  - Parallel token generation in 8-16 diffusion steps
  - PDE-based noise schedule for fast convergence

Q3-inspired additions (all default-safe for ep5 backward compat):
  - Optional AdaLN timestep conditioning
  - Optional learned confidence head
  - Span masking ratio for training
  - Mask ratio curriculum range
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SophiaT1Config:
    """
    Two model sizes:
      - T1-Small: 1.5B params (~1.5GB INT4) — phone/edge
      - T1-Base:  3B params (~3GB INT4) — 12GB GPU
    """

    # ── Architecture ───────────────────────────────────────────
    vocab_size: int = 32768        # BPE vocab (compact — not 256K like Gemma)
    hidden_size: int = 2048        # T1-Base. T1-Small: 1536
    num_layers: int = 28           # T1-Base. T1-Small: 20
    num_heads: int = 16            # T1-Base. T1-Small: 12
    num_kv_heads: int = 4          # Grouped Query Attention (4:1 ratio)
    intermediate_size: int = 5632  # FFN dim. T1-Small: 4096
    head_dim: int = 128            # Per-head dimension

    # ── Context & Attention ────────────────────────────────────
    max_seq_len: int = 131072      # 128K context
    sliding_window: int = 4096     # Local attention window
    global_tokens: int = 256       # Global attention tokens (CLS-like)

    # ── Diffusion ──────────────────────────────────────────────
    mask_token_id: int = 32766     # Special [MASK] token
    pad_token_id: int = 32767      # Special [PAD] token
    num_denoise_steps_train: int = 32
    num_denoise_steps_inference: int = 12
    noise_schedule: str = "pde_cosine"

    # ── Q3-inspired diffusion improvements ─────────────────────
    use_adaln: bool = False               # AdaLN timestep conditioning (Q3 §3.3.2)
    use_confidence_head: bool = False     # Learned confidence (Q3 §3.2.5)
    timestep_embed_dim: int = 256         # Timestep embedding dim
    span_mask_prob: float = 0.15          # Fraction of batch getting span masks (Q3 §3.2.7)
    span_mean_length: float = 3.0         # Geometric mean for span length
    mask_ratio_min: float = 0.15          # Lower bound for curriculum
    mask_ratio_max: float = 0.85          # Upper bound for curriculum
    unmask_temperature: float = 0.8       # Generation temp
    unmask_beta: float = 0.5              # Quantile offset τ = μ + β·σ (Q3 §3.2.4)

    # ── Normalization & Regularization ─────────────────────────
    rms_norm_eps: float = 1e-6
    rope_theta: float = 500000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    # ── Training ───────────────────────────────────────────────
    dtype: str = "bfloat16"
    init_std: float = 0.02

    @property
    def num_params_millions(self) -> float:
        embed = self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)
        per_layer = (
            self.hidden_size * (self.num_heads + 2 * self.num_kv_heads) * self.head_dim +
            self.num_heads * self.head_dim * self.hidden_size +
            3 * self.hidden_size * self.intermediate_size +
            2 * self.hidden_size
        )
        total = embed + self.num_layers * per_layer
        return total / 1e6


def sophia_t1_small() -> SophiaT1Config:
    return SophiaT1Config(
        hidden_size=1536, num_layers=20, num_heads=12, num_kv_heads=4,
        intermediate_size=4096, head_dim=128,
    )


def sophia_t1_base() -> SophiaT1Config:
    return SophiaT1Config(
        hidden_size=2048, num_layers=28, num_heads=16, num_kv_heads=4,
        intermediate_size=5632, head_dim=128,
    )


def sophia_t1_base_q3() -> SophiaT1Config:
    """T1-Base with Q3 improvements enabled — for continuation training from ep5."""
    cfg = sophia_t1_base()
    cfg.use_adaln = True
    cfg.use_confidence_head = True
    return cfg


if __name__ == "__main__":
    small = sophia_t1_small()
    base = sophia_t1_base()
    print(f"T1-Small: {small.num_params_millions:.0f}M params")
    print(f"T1-Base:  {base.num_params_millions:.0f}M params")
