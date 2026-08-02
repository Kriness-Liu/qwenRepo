"""KV cache inspection and size accounting."""

from __future__ import annotations

from collections.abc import Iterator

import torch


def iter_cache_tensors(cache) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if hasattr(cache, "layers"):
        for layer in cache.layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is not None and values is not None:
                yield keys, values
        return
    for layer in cache:
        if len(layer) < 2:
            raise ValueError("legacy cache layers must contain key and value tensors")
        yield layer[0], layer[1]


def cache_nbytes(cache) -> int:
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in iter_cache_tensors(cache)
    )


def cache_seq_length(cache) -> int:
    if hasattr(cache, "get_seq_length"):
        return int(cache.get_seq_length())
    first_key, _ = next(iter(iter_cache_tensors(cache)))
    return int(first_key.shape[-2])


def expected_cache_nbytes(config, batch_size: int, sequence_length: int, dtype: torch.dtype) -> int:
    if batch_size <= 0 or sequence_length < 0:
        raise ValueError("batch_size must be positive and sequence_length non-negative")
    head_dim = int(config.hidden_size) // int(config.num_attention_heads)
    element_size = torch.empty((), dtype=dtype).element_size()
    return (
        2
        * int(config.num_hidden_layers)
        * batch_size
        * int(config.num_key_value_heads)
        * sequence_length
        * head_dim
        * element_size
    )
