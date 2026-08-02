"""Experiment 01: Qwen GQA head mapping and SDPA equivalence."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as functional
from transformers import AutoConfig

from qwenrepo.attention import gqa_spec, scaled_dot_product_gqa
from qwenrepo.modeling import resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    args = parser.parse_args()
    config = AutoConfig.from_pretrained(resolve_model_path(args.model_path), local_files_only=True)
    spec = gqa_spec(config)
    print(spec)

    torch.manual_seed(7)
    query = torch.randn(2, spec.query_heads, 8, spec.head_dim)
    key = torch.randn(2, spec.key_value_heads, 8, spec.head_dim)
    value = torch.randn_like(key)
    explicit = scaled_dot_product_gqa(query, key, value)
    native = functional.scaled_dot_product_attention(
        query, key, value, is_causal=True, enable_gqa=True
    )
    torch.testing.assert_close(explicit, native)
    print("Q/K/V:", tuple(query.shape), tuple(key.shape), tuple(value.shape))
    print("PASS: explicit KV repetition equals native enable_gqa=True.")


if __name__ == "__main__":
    main()
