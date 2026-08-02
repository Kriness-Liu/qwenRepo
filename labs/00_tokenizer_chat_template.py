"""Experiment 00: inspect Chat Template, token IDs and prompt boundaries."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from qwenrepo.modeling import resolve_model_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        resolve_model_path(args.model_path), local_files_only=True
    )
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "解释KV Cache。"},
    ]
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoded = tokenizer(rendered, return_tensors="pt")
    print(rendered)
    print("input_ids shape:", tuple(encoded.input_ids.shape))
    print("first tokens:", encoded.input_ids[0, :16].tolist())
    print("round trip:", tokenizer.decode(encoded.input_ids[0]))


if __name__ == "__main__":
    main()
