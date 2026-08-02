from __future__ import annotations

import unittest

import torch
import torch.nn.functional as functional

from qwenrepo.attention import repeat_kv, scaled_dot_product_gqa


class AttentionTests(unittest.TestCase):
    def test_repeat_kv_preserves_head_groups(self) -> None:
        tensor = torch.arange(2 * 2 * 3 * 4).reshape(2, 2, 3, 4)
        repeated = repeat_kv(tensor, 3)
        self.assertEqual(repeated.shape, (2, 6, 3, 4))
        torch.testing.assert_close(repeated[:, 0], repeated[:, 1])
        torch.testing.assert_close(repeated[:, 1], repeated[:, 2])
        torch.testing.assert_close(repeated[:, 3], repeated[:, 4])

    def test_explicit_gqa_matches_native_sdpa(self) -> None:
        torch.manual_seed(7)
        query = torch.randn(2, 4, 5, 8)
        key = torch.randn(2, 2, 5, 8)
        value = torch.randn_like(key)
        actual = scaled_dot_product_gqa(query, key, value)
        expected = functional.scaled_dot_product_attention(
            query, key, value, is_causal=True, enable_gqa=True
        )
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
