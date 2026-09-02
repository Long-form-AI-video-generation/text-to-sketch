"""Transformer encoder blocks using PyTorch scaled dot-product attention."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from models.sketchformer.config import SketchformerConfig


def _activation(name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    raise ValueError(f"Unsupported activation: {name}")


class SDPAAttention(nn.Module):
    """Multi-head attention backed by ``scaled_dot_product_attention``."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float,
        *,
        query_dim: int | None = None,
        key_dim: int | None = None,
        value_dim: int | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = float(dropout)

        self.q_proj = nn.Linear(query_dim or d_model, d_model)
        self.k_proj = nn.Linear(key_dim or d_model, d_model)
        self.v_proj = nn.Linear(value_dim or d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        *,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch_size = query.shape[0]

        q = self._split_heads(self.q_proj(query), batch_size)
        k, v = self.project_key_value(key, value)
        return self._attend(query, q, k, v, attention_mask, is_causal=is_causal)

    def project_key_value(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project reusable key/value states for cached decoding."""

        batch_size = key.shape[0]
        return (
            self._split_heads(self.k_proj(key), batch_size),
            self._split_heads(self.v_proj(value), batch_size),
        )

    def forward_cached(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        static_key_value: bool = False,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Attend one or more new queries and return projected key/value cache."""

        batch_size = query.shape[0]
        q = self._split_heads(self.q_proj(query), batch_size)
        if static_key_value and cache is not None:
            k, v = cache
        else:
            new_k, new_v = self.project_key_value(key, value)
            if cache is not None and not static_key_value:
                k = torch.cat((cache[0], new_k), dim=2)
                v = torch.cat((cache[1], new_v), dim=2)
            else:
                k, v = new_k, new_v
        output = self._attend(query, q, k, v, attention_mask, is_causal=False)
        return output, (k, v)

    def _attend(
        self,
        query: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: torch.Tensor | None,
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        batch_size = query.shape[0]
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch_size, query.shape[1], self.d_model)
        return self.out_proj(attended)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, x.shape[1], self.num_heads, self.head_dim)
        return x.transpose(1, 2)


class FeedForward(nn.Module):
    """Position-wise transformer feed-forward block."""

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.norm_first = config.norm_first
        self.self_attn = SDPAAttention(config.d_model, config.num_heads, config.dropout)
        self.ffn = FeedForward(
            config.d_model,
            config.dim_feedforward,
            config.dropout,
            config.activation,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.norm_first:
            norm_x = self.norm1(x)
            x = x + self.dropout1(self.self_attn(norm_x, norm_x, norm_x, attention_mask))
            x = x + self.dropout2(self.ffn(self.norm2(x)))
            return x

        attn = self.self_attn(x, x, x, attention_mask)
        x = self.norm1(x + self.dropout1(attn))
        x = self.norm2(x + self.dropout2(self.ffn(x)))
        return x


class StrokeEncoder(nn.Module):
    """Stack of SDPA transformer encoder blocks."""

    def __init__(self, config: SketchformerConfig) -> None:
        super().__init__()
        self.gradient_checkpointing = config.gradient_checkpointing
        self.layers = nn.ModuleList(
            [EncoderBlock(config) for _ in range(config.num_encoder_layers)]
        )
        self.final_norm = (
            nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
            if config.use_final_norm
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                x = checkpoint(layer, x, attention_mask, use_reentrant=False)
            else:
                x = layer(x, attention_mask)
        return self.final_norm(x)


class AttentionPool(nn.Module):
    """Mask-aware attention pooling from sequence states to one latent vector."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int,
        *,
        mode: str = "projected",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.mode = mode
        if mode == "projected":
            self.score = nn.Linear(d_model, 1)
            self.projection = nn.Linear(d_model, latent_dim)
        elif mode == "tf_self_attn_v1":
            hidden = int(hidden_dim or latent_dim)
            self.W_attn = nn.Parameter(torch.empty(d_model, hidden))
            self.b_attn = nn.Parameter(torch.zeros(hidden))
            self.V_attn = nn.Parameter(torch.empty(hidden, 1))
            nn.init.normal_(self.W_attn)
            nn.init.uniform_(self.V_attn)
        else:
            raise ValueError("mode must be one of: projected, tf_self_attn_v1")

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.mode == "projected":
            scores = self.score(x).squeeze(-1)
        else:
            scores = torch.tanh(x @ self.W_attn + self.b_attn) @ self.V_attn
            scores = scores.squeeze(-1)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return self.projection(pooled) if self.mode == "projected" else pooled
