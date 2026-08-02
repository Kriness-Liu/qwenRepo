"""Qwen architecture and inference-performance learning helpers."""

from .attention import GqaSpec, gqa_spec, repeat_kv, scaled_dot_product_gqa
from .benchmark import benchmark_grid, benchmark_single_case
from .cache import cache_nbytes, cache_seq_length, expected_cache_nbytes
from .modeling import load_model_and_tokenizer, resolve_model_path

__all__ = [
    "GqaSpec",
    "benchmark_grid",
    "benchmark_single_case",
    "cache_nbytes",
    "cache_seq_length",
    "expected_cache_nbytes",
    "gqa_spec",
    "load_model_and_tokenizer",
    "repeat_kv",
    "resolve_model_path",
    "scaled_dot_product_gqa",
]
