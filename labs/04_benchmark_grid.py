"""Experiment 04: sweep batch size and context length.

Run from the repository root after ``python -m pip install -e .``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from qwenrepo.benchmark import benchmark_grid
from qwenrepo.modeling import load_model_and_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    args = parser.parse_args()

    model, _, device = load_model_and_tokenizer(args.model_path)
    if device.type != "cuda":
        raise SystemExit("CUDA is required for this experiment")
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    cases = benchmark_grid(
        model,
        args.batch_sizes,
        args.prompt_lengths,
        decode_tokens=args.decode_tokens,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(cases, indent=2), encoding="utf-8")
    print(json.dumps(cases, indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
