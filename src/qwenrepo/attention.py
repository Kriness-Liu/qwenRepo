"""Small, inspectable Grouped-Query Attention shape helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class GqaSpec:
    hidden_size: int
    query_heads: int
    key_value_heads: int
    head_dim: int
    query_heads_per_kv_head: int


def gqa_spec(config) -> GqaSpec:
    query_heads = int(config.num_attention_heads)
    key_value_heads = int(config.num_key_value_heads)
    if query_heads % key_value_heads != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    hidden_size = int(config.hidden_size)
    return GqaSpec(
        hidden_size=hidden_size,
        query_heads=query_heads,
        key_value_heads=key_value_heads,
        head_dim=hidden_size // query_heads,
        query_heads_per_kv_head=query_heads // key_value_heads,
    )


def repeat_kv(hidden_states: torch.Tensor, repetitions: int) -> torch.Tensor:
    """Expand [B, Hkv, S, D] to [B, Hq, S, D] without changing values."""

    if hidden_states.ndim != 4:
        raise ValueError("hidden_states must have shape [batch, kv_heads, sequence, head_dim]")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if repetitions == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repetitions, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repetitions, sequence, head_dim)


def scaled_dot_product_gqa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    is_causal: bool = True,
) -> torch.Tensor:
    """Reference GQA by explicitly repeating K/V heads before SDPA."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key and value must be four-dimensional")
    if key.shape != value.shape:
        raise ValueError("key and value shapes must match")
    if query.shape[0] != key.shape[0] or query.shape[-1] != key.shape[-1]:
        raise ValueError("batch size and head dimension must match")
    if query.shape[1] % key.shape[1] != 0:
        raise ValueError("query heads must be divisible by key/value heads")
    repetitions = query.shape[1] // key.shape[1]
    return functional.scaled_dot_product_attention(
        query,
        repeat_kv(key, repetitions),
        repeat_kv(value, repetitions),
        is_causal=is_causal,
    )
