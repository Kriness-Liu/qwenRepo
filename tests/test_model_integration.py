from __future__ import annotations

import os
import unittest

import torch

from qwenrepo.cache import cache_nbytes, cache_seq_length
from qwenrepo.modeling import forward_last_logits, load_model_and_tokenizer


@unittest.skipUnless(
    os.environ.get("QWEN_MODEL_PATH") and torch.cuda.is_available(),
    "set QWEN_MODEL_PATH and provide CUDA for model integration tests",
)
class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model, cls.tokenizer, cls.device = load_model_and_tokenizer(
            os.environ["QWEN_MODEL_PATH"]
        )

    @torch.inference_mode()
    def test_cached_decode_matches_full_recompute(self) -> None:
        encoded = self.tokenizer("hello CUDA", return_tensors="pt").to(self.device)
        prefill = forward_last_logits(
            self.model, **encoded, use_cache=True, return_dict=True
        )
        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        combined = torch.cat((encoded.input_ids, next_token), dim=1)
        full = forward_last_logits(
            self.model, input_ids=combined, use_cache=False, return_dict=True
        )
        incremental = forward_last_logits(
            self.model,
            input_ids=next_token,
            attention_mask=torch.ones_like(combined),
            past_key_values=prefill.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        torch.testing.assert_close(
            incremental.logits[:, -1], full.logits[:, -1], rtol=2e-2, atol=2e-2
        )
        self.assertEqual(cache_seq_length(incremental.past_key_values), combined.shape[1])
        self.assertGreater(cache_nbytes(incremental.past_key_values), 0)


if __name__ == "__main__":
    unittest.main()
