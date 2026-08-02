"""Run the Qwen batch/context sweep and save a reproducible JSON report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from qwenrepo.benchmark import benchmark_grid
from qwenrepo.modeling import load_model_and_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 128, 512])
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results/latest.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, _, device = load_model_and_tokenizer(args.model_path)
    if device.type != "cuda":
        raise SystemExit("CUDA is required for the benchmark grid")
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
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "dtype": str(next(model.parameters()).dtype),
            "model_type": model.config.model_type,
        },
        "methodology": {
            "timer": "CUDA Event",
            "ttft_scope": "model-only prefill; tokenization excluded",
            "tpot_scope": "sequential cached decode after first token",
            "synthetic_token_ids": True,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
