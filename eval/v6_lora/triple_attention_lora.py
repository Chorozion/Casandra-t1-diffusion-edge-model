"""LoRA adapter + variant-specific interventions for AnchorAwareTripleAttention.

Five interventions, toggleable per variant:

  V1 baseline:        plain LoRA on Q/K/V/O of all 3 paths
  V2 learned_projection: MLP from concat(lx,ly,lz) → contribution to Path-3 K
  V3 multi_resolution:  coarse 4³ + medium 16³ + fine 64³ embedding tables
  V4 gate_temp_anneal:  softmax temperature on the path-mix gate, annealed
                        from τ=5 → τ=1 over training
  V5 aux_contrastive:   aux loss forcing lattice-conditioned logits to differ
                        from un-conditioned logits at locked positions

V6 stack = V2 + V3 + V4 + V5 simultaneously.

Architecture: we ATTACH this module to an existing AnchorAwareTripleAttention
instance (which is then frozen). The wrapper holds:
  - LoRA A/B matrices for each Q/K/V/O projection (8 LoRAs per attn layer)
  - V2 lattice MLP
  - V3 multi-resolution embedding tables
  - Optionally trainable LTMi gate that overrides the frozen base gate

In forward, we re-implement triple attention but call the LoRA-adjusted
linear layers. This adds compute overhead vs the base implementation but is
cheap relative to the attention itself.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────
# Variant configuration
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class VariantConfig:
    """All toggleable interventions. False/None values mean the intervention
    is inactive — V1 baseline has lora_rank=16 and everything else off.
    """
    variant_name: str = "V1"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.0
    # LoRA target subset — which projections get adapters
    # Default: all 8 (q/k/v/o for content + anchor paths)
    lora_targets: tuple[str, ...] = (
        "q_content", "k_content", "v_content", "o_content",
        "q_anchor", "k_anchor", "v_anchor", "o_anchor",
    )
    # V2: learned MLP on lattice coords → K contribution
    learned_lattice_projection: bool = False
    lattice_mlp_hidden: int = 256
    # V3: multi-resolution lattice (coarse + medium + fine)
    multi_resolution_lattice: bool = False
    multi_res_dims: tuple[int, ...] = (4, 16, 64)  # coarse, medium, fine
    # V4: gate softmax temperature annealing
    gate_temp_anneal: bool = False
    gate_temp_start: float = 5.0
    gate_temp_end: float = 1.0
    # V5: aux contrastive loss
    aux_contrastive: bool = False
    aux_contrastive_weight: float = 0.1
    # LTMi gate trainable override (always on for variants that interact with lattice)
    train_ltmi_gate: bool = True
    ltmi_gate_init: float = 0.5  # warmer than base 0.0/0.1


def variant_v1() -> VariantConfig:
    return VariantConfig(variant_name="V1")


def variant_v2() -> VariantConfig:
    return VariantConfig(
        variant_name="V2",
        learned_lattice_projection=True,
    )


def variant_v3() -> VariantConfig:
    return VariantConfig(
        variant_name="V3",
        multi_resolution_lattice=True,
    )


def variant_v4() -> VariantConfig:
    return VariantConfig(
        variant_name="V4",
        gate_temp_anneal=True,
    )


def variant_v5() -> VariantConfig:
    return VariantConfig(
        variant_name="V5",
        aux_contrastive=True,
    )


def variant_v6() -> VariantConfig:
    return VariantConfig(
        variant_name="V6",
        learned_lattice_projection=True,
        multi_resolution_lattice=True,
        gate_temp_anneal=True,
        aux_contrastive=True,
    )


VARIANTS = {
    "V1": variant_v1,
    "V2": variant_v2,
    "V3": variant_v3,
    "V4": variant_v4,
    "V5": variant_v5,
    "V6": variant_v6,
}


# ─────────────────────────────────────────────────────────────────────────
# LoRA wrapper for nn.Linear
# ─────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """LoRA: y = base(x) + (B @ A @ x) * (alpha / rank).

    The base module is frozen; only A and B are trained. A is init with
    Kaiming (small random); B is init to zero so the LoRA contribution is
    zero at step 0 (training starts from the base model's behavior).
    """

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_features = base.in_features
        out_features = base.out_features
        self.rank = rank
        self.scale = alpha / max(1, rank)

        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scale
        return base_out + lora_out


# ─────────────────────────────────────────────────────────────────────────
# Intervention modules
# ─────────────────────────────────────────────────────────────────────────
class LatticeMLP(nn.Module):
    """V2: learned projection (lx, ly, lz) → contribution to Path-3 K.

    Takes the lattice-axis embedding sums and runs them through a small
    MLP. Output is added (gated by ltmi_gate) into k3_pre at anchor
    positions, replacing the simple `lx + ly + lz` of the base module.
    """

    def __init__(self, in_dim: int, hidden: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        # Init last layer to zero — at step 0, MLP output is zero, so the
        # base behavior (no contribution from lattice MLP) is preserved.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, lattice_sum: torch.Tensor) -> torch.Tensor:
        return self.net(lattice_sum)


class MultiResLatticeEmb(nn.Module):
    """V3: multi-resolution lattice embeddings.

    For each axis, three embedding tables at different resolutions: 4 / 16 / 64.
    Coords are quantized down: coarse = coord // 16, medium = coord // 4,
    fine = coord. All three embeddings are summed per axis, then per-axis
    sums are summed across axes. Output shape: [B, S, kv_heads * head_dim].

    All tables initialized to zero (preserves base behavior at step 0).
    """

    def __init__(self, fine_dim: int, embed_dim: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        out_dim = num_kv_heads * head_dim
        self.fine_dim = fine_dim  # 64
        self.medium_dim = max(2, fine_dim // 4)  # 16
        self.coarse_dim = max(2, fine_dim // 16)  # 4
        self.coarse = nn.ModuleList([nn.Embedding(self.coarse_dim, out_dim) for _ in range(3)])
        self.medium = nn.ModuleList([nn.Embedding(self.medium_dim, out_dim) for _ in range(3)])
        self.fine = nn.ModuleList([nn.Embedding(self.fine_dim, out_dim) for _ in range(3)])
        for table_list in (self.coarse, self.medium, self.fine):
            for t in table_list:
                nn.init.zeros_(t.weight)

    def forward(self, lattice: torch.Tensor) -> torch.Tensor:
        """lattice: [B, S, 3] long. Returns [B, S, out_dim]."""
        coarse_div = self.fine_dim // self.coarse_dim
        medium_div = self.fine_dim // self.medium_dim
        coarse = lattice // coarse_div
        medium = lattice // medium_div
        fine = lattice
        total = torch.zeros(
            *lattice.shape[:-1], self.coarse[0].embedding_dim,
            dtype=self.coarse[0].weight.dtype, device=lattice.device,
        )
        for axis in range(3):
            total = total + self.coarse[axis](coarse[..., axis].clamp(0, self.coarse_dim - 1))
            total = total + self.medium[axis](medium[..., axis].clamp(0, self.medium_dim - 1))
            total = total + self.fine[axis](fine[..., axis].clamp(0, self.fine_dim - 1))
        return total


# ─────────────────────────────────────────────────────────────────────────
# The LoRA + interventions wrapper for AnchorAwareTripleAttention
# ─────────────────────────────────────────────────────────────────────────
class TripleAttentionLoRA(nn.Module):
    """Replaces a frozen AnchorAwareTripleAttention's forward with a LoRA-
    adjusted + intervention-augmented version. The base module's weights
    are referenced (frozen); the wrapper holds all trainable parameters.

    To use: at model load time, for each layer L with attn=AnchorAwareTripleAttention,
    wrap it as: L.attn = TripleAttentionLoRA(L.attn, variant_config).
    """

    def __init__(self, base_attn, variant: VariantConfig):
        super().__init__()
        self.base = base_attn
        self.variant = variant
        for p in self.base.parameters():
            p.requires_grad = False

        # Cached references for forward
        self.num_heads = base_attn.num_heads
        self.num_kv_heads = base_attn.num_kv_heads
        self.head_dim = base_attn.head_dim
        self.hidden_size = base_attn.hidden_size
        self.kv_repeat = base_attn.kv_repeat
        self.use_ltmi_priors = base_attn.use_ltmi_priors
        self.lattice_dim = base_attn.lattice_dim

        # ── LoRA on projections ──
        self.loras = nn.ModuleDict()
        for target in variant.lora_targets:
            if hasattr(base_attn, target):
                module = getattr(base_attn, target)
                if isinstance(module, nn.Linear):
                    self.loras[target] = LoRALinear(
                        module, variant.lora_rank, variant.lora_alpha,
                        variant.lora_dropout,
                    )

        # ── Trainable LTMi gate override ──
        if variant.train_ltmi_gate and self.use_ltmi_priors:
            self.ltmi_gate_override = nn.Parameter(
                torch.tensor([variant.ltmi_gate_init], dtype=torch.float32)
            )
        else:
            self.ltmi_gate_override = None

        # ── V2: learned lattice projection MLP ──
        if variant.learned_lattice_projection and self.use_ltmi_priors:
            self.lattice_mlp = LatticeMLP(
                in_dim=self.num_kv_heads * self.head_dim,
                hidden=variant.lattice_mlp_hidden,
                out_dim=self.num_kv_heads * self.head_dim,
            )
        else:
            self.lattice_mlp = None

        # ── V3: multi-resolution lattice ──
        if variant.multi_resolution_lattice and self.use_ltmi_priors:
            self.multi_res_lattice = MultiResLatticeEmb(
                fine_dim=self.lattice_dim,
                embed_dim=self.num_kv_heads * self.head_dim,
                num_kv_heads=self.num_kv_heads,
                head_dim=self.head_dim,
            )
        else:
            self.multi_res_lattice = None

        # ── V4: gate temp (controlled per step from train loop) ──
        self.register_buffer("gate_temp", torch.tensor(1.0, dtype=torch.float32))

    def _proj(self, target: str, x: torch.Tensor) -> torch.Tensor:
        """Use LoRA-adjusted projection if available, else fall back to base."""
        if target in self.loras:
            return self.loras[target](x)
        return getattr(self.base, target)(x)

    def forward(
        self,
        x: torch.Tensor,
        freqs: torch.Tensor,
        mask: torch.Tensor | None = None,
        t_emb: torch.Tensor | None = None,
        anchor_mask: torch.Tensor | None = None,
        anchor_scores: torch.Tensor | None = None,
        anchor_lattice: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from model.triple_attention import apply_rotary
        bsz, seq_len, _ = x.shape

        # Path 1 — content
        q1 = self._proj("q_content", x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = self._proj("k_content", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v1 = self._proj("v_content", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q1 = apply_rotary(q1, freqs)
        k1 = apply_rotary(k1, freqs)
        if self.kv_repeat > 1:
            k1 = k1.repeat_interleave(self.kv_repeat, dim=1)
            v1 = v1.repeat_interleave(self.kv_repeat, dim=1)
        p1_attn = F.scaled_dot_product_attention(q1, k1, v1, attn_mask=mask)
        p1 = self._proj("o_content", p1_attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))

        # Path 2 — timestep (no LoRA, use base)
        if t_emb is not None:
            q2 = self.base.q_time(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            q2 = apply_rotary(q2, freqs)
            k2, v2 = self.base.tkv(t_emb, seq_len)
            p2_attn = F.scaled_dot_product_attention(q2, k2, v2, attn_mask=None)
            p2 = self.base.o_time(p2_attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))
        else:
            p2 = torch.zeros_like(p1)

        # Path 3 — anchor + lattice
        anchor_present_per_batch = (
            anchor_mask.any(dim=-1) if anchor_mask is not None else None
        )
        if anchor_present_per_batch is not None and anchor_present_per_batch.any().item():
            q3 = self._proj("q_anchor", x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k3_pre = self._proj("k_anchor", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v3 = self._proj("v_anchor", x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # Lattice contribution to k3
            if self.use_ltmi_priors and anchor_lattice is not None:
                lattice_emb = self._compute_lattice_emb(anchor_lattice, anchor_mask, k3_pre.dtype)
                gate_val = (
                    torch.sigmoid(self.ltmi_gate_override).to(lattice_emb.dtype)
                    if self.ltmi_gate_override is not None
                    else self.base.ltmi_gate.to(lattice_emb.dtype)
                )
                k3_pre = k3_pre + gate_val * lattice_emb

            q3 = apply_rotary(q3, freqs)
            k3 = apply_rotary(k3_pre, freqs)
            if self.kv_repeat > 1:
                k3 = k3.repeat_interleave(self.kv_repeat, dim=1)
                v3 = v3.repeat_interleave(self.kv_repeat, dim=1)

            anchor_attn_bias = torch.where(
                anchor_mask.unsqueeze(1).unsqueeze(2),
                torch.zeros((), dtype=q3.dtype, device=q3.device),
                torch.full((), float("-inf"), dtype=q3.dtype, device=q3.device),
            )

            if self.use_ltmi_priors and anchor_scores is not None:
                # score_bias via base.relevance_proj (frozen — could add LoRA later)
                score_bias = self.base.relevance_proj(anchor_scores.unsqueeze(-1).to(q3.dtype))
                score_bias = score_bias.permute(0, 2, 1).unsqueeze(2)
                anchor_attn_bias = anchor_attn_bias + score_bias
            if mask is not None:
                if mask.dtype == torch.bool:
                    base = torch.where(
                        mask,
                        torch.zeros((), dtype=q3.dtype, device=q3.device),
                        torch.full((), float("-inf"), dtype=q3.dtype, device=q3.device),
                    )
                else:
                    base = mask.to(q3.dtype)
                anchor_attn_bias = base + anchor_attn_bias

            row_has_anchor = anchor_present_per_batch.view(bsz, 1, 1, 1)
            safe_bias = torch.where(
                row_has_anchor,
                anchor_attn_bias,
                torch.zeros_like(anchor_attn_bias),
            )
            p3_attn = F.scaled_dot_product_attention(q3, k3, v3, attn_mask=safe_bias)
            p3 = self._proj("o_anchor", p3_attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))
            p3 = p3 * row_has_anchor.view(bsz, 1, 1).to(p3.dtype)
        else:
            p3 = torch.zeros_like(p1)

        # Gate
        anchor_feat = (
            anchor_mask.any(dim=-1, keepdim=True).to(x.dtype)
            if anchor_mask is not None
            else torch.zeros(bsz, 1, dtype=x.dtype, device=x.device)
        )
        anchor_feat = anchor_feat.unsqueeze(1).expand(bsz, seq_len, 1)
        gate_input = torch.cat([x, anchor_feat], dim=-1)
        gate_logits = self.base.gate(gate_input)
        # V4: apply temperature
        if self.variant.gate_temp_anneal:
            gate_logits = gate_logits / self.gate_temp.to(gate_logits.dtype).clamp_min(0.1)
        gates = F.softmax(gate_logits, dim=-1)

        out = gates[..., 0:1] * p1 + gates[..., 1:2] * p2 + gates[..., 2:3] * p3
        return out

    def _compute_lattice_emb(
        self, anchor_lattice: torch.Tensor, anchor_mask: torch.Tensor, dtype: torch.dtype
    ) -> torch.Tensor:
        bsz, seq_len, _ = anchor_lattice.shape

        # Base path: use base module's lattice embeddings (lx + ly + lz)
        lx = self.base.lattice_x_emb(anchor_lattice[..., 0])
        ly = self.base.lattice_y_emb(anchor_lattice[..., 1])
        lz = self.base.lattice_z_emb(anchor_lattice[..., 2])
        lattice_sum = lx + ly + lz  # [B, S, kv*D]

        # V2: pass through learned MLP
        if self.lattice_mlp is not None:
            lattice_sum = lattice_sum + self.lattice_mlp(lattice_sum)

        # V3: add multi-resolution contribution
        if self.multi_res_lattice is not None:
            multi = self.multi_res_lattice(anchor_lattice)
            lattice_sum = lattice_sum + multi

        # Reshape to attention layout
        out = lattice_sum.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Mask out non-anchor positions
        anchor_kv_mask = anchor_mask.unsqueeze(1).unsqueeze(-1).to(out.dtype)
        return out * anchor_kv_mask

    def set_gate_temp(self, temp: float):
        """Called from train loop to anneal V4 gate temperature."""
        self.gate_temp.fill_(float(temp))


def wrap_model_with_lora(model, variant: VariantConfig) -> tuple[nn.Module, int]:
    """Wrap every layer's attn with TripleAttentionLoRA. Returns the model and
    the count of trainable params added."""
    n_wrapped = 0
    for layer in model.layers:
        base_attn = layer.attn
        # Skip if not triple-attention
        if not hasattr(base_attn, "q_content"):
            continue
        layer.attn = TripleAttentionLoRA(base_attn, variant)
        n_wrapped += 1
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model, n_trainable
