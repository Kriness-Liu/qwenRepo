"""Experiment 03: model-only TTFT, TPOT and decode throughput."""

from __future__ import annotations

import argparse
import json

from qwenrepo.benchmark import benchmark_single_case
from qwenrepo.modeling import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--decode-tokens", type=int, default=16)
    args = parser.parse_args()
    model, _, _ = load_model_and_tokenizer(args.model_path)
    result = benchmark_single_case(
        model,
        batch_size=args.batch_size,
        prompt_length=args.prompt_length,
        decode_tokens=args.decode_tokens,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
