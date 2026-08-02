from __future__ import annotations

import types
import unittest

import torch

from qwenrepo.cache import cache_nbytes, cache_seq_length, expected_cache_nbytes


class CacheTests(unittest.TestCase):
    def test_legacy_cache_size_and_sequence_length(self) -> None:
        cache = ((torch.empty(2, 3, 7, 5), torch.empty(2, 3, 7, 5)),) * 4
        expected = 4 * 2 * 2 * 3 * 7 * 5 * 4
        self.assertEqual(cache_nbytes(cache), expected)
        self.assertEqual(cache_seq_length(cache), 7)

    def test_formula_uses_kv_heads_not_query_heads(self) -> None:
        config = types.SimpleNamespace(
            hidden_size=896,
            num_attention_heads=14,
            num_key_value_heads=2,
            num_hidden_layers=24,
        )
        actual = expected_cache_nbytes(config, 2, 128, torch.float16)
        expected = 2 * 24 * 2 * 2 * 128 * 64 * 2
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
