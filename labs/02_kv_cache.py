"""Experiment 02: validate incremental decoding and account KV cache bytes."""

from __future__ import annotations

import argparse

import torch

from qwenrepo.cache import cache_nbytes, cache_seq_length, expected_cache_nbytes
from qwenrepo.modeling import forward_last_logits, load_model_and_tokenizer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--prompt", default="请用一句话解释 GPU。")
    args = parser.parse_args()
    model, tokenizer, device = load_model_and_tokenizer(args.model_path)
    encoded = tokenizer(args.prompt, return_tensors="pt").to(device)

    prefill = forward_last_logits(model, **encoded, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    next_token = prefill.logits[:, -1:].argmax(dim=-1)
    combined_ids = torch.cat((encoded.input_ids, next_token), dim=1)
    full = forward_last_logits(model, input_ids=combined_ids, use_cache=False, return_dict=True)
    incremental = forward_last_logits(
        model,
        input_ids=next_token,
        attention_mask=torch.ones_like(combined_ids),
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    torch.testing.assert_close(
        incremental.logits[:, -1], full.logits[:, -1], rtol=2e-2, atol=2e-2
    )
    actual = cache_nbytes(incremental.past_key_values)
    expected = expected_cache_nbytes(
        model.config,
        batch_size=combined_ids.shape[0],
        sequence_length=combined_ids.shape[1],
        dtype=next(model.parameters()).dtype,
    )
    assert actual == expected, (actual, expected)
    print("cache type:", type(incremental.past_key_values).__name__)
    print("cache sequence length:", cache_seq_length(incremental.past_key_values))
    print("cache bytes actual/expected:", actual, expected)
    print("PASS: cached decode logits match full recomputation.")


if __name__ == "__main__":
    main()
