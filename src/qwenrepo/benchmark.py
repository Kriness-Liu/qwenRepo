"""Model-only TTFT, TPOT, throughput and memory measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import fmean, median

import torch

from .cache import cache_nbytes
from .modeling import forward_last_logits


@dataclass(frozen=True)
class LatencyStats:
    samples: int
    p50_ms: float
    p95_ms: float
    mean_ms: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("values must not be empty")
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _time_cuda(operation, *, warmup: int, iterations: int) -> LatencyStats:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return LatencyStats(
        samples=len(samples),
        p50_ms=_percentile(samples, 50),
        p95_ms=_percentile(samples, 95),
        mean_ms=fmean(samples),
    )


@torch.inference_mode()
def _decode_steps(model, input_ids: torch.Tensor, decode_tokens: int):
    prefill = forward_last_logits(model, input_ids=input_ids, use_cache=True, return_dict=True)
    cache = prefill.past_key_values
    next_token = prefill.logits[:, -1:].argmax(dim=-1)
    samples: list[float] = []
    total_length = input_ids.shape[1]
    for _ in range(max(0, decode_tokens - 1)):
        attention_mask = torch.ones(
            input_ids.shape[0], total_length + 1, device=input_ids.device, dtype=torch.long
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = forward_last_logits(
            model,
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
        cache = output.past_key_values
        next_token = output.logits[:, -1:].argmax(dim=-1)
        total_length += 1
    return samples, cache


@torch.inference_mode()
def benchmark_single_case(
    model,
    *,
    batch_size: int,
    prompt_length: int,
    decode_tokens: int = 16,
    warmup: int = 2,
    iterations: int = 10,
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for model-only timing")
    device = next(model.parameters()).device
    input_ids = torch.randint(
        0, int(model.config.vocab_size), (batch_size, prompt_length), device=device
    )

    prefill_stats = _time_cuda(
        lambda: forward_last_logits(model, input_ids=input_ids, use_cache=True, return_dict=True),
        warmup=warmup,
        iterations=iterations,
    )
    # Decode executes a different shape from prefill (query length is one).  Warm
    # it independently so lazy kernel/module initialization is not charged to TPOT.
    _decode_steps(model, input_ids, min(decode_tokens, 2))
    torch.cuda.synchronize()
    baseline_memory = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    decode_samples, cache = _decode_steps(model, input_ids, decode_tokens)
    peak_delta = max(0, torch.cuda.max_memory_allocated(device) - baseline_memory)
    decode_mean = fmean(decode_samples) if decode_samples else 0.0
    generated_after_prefill = len(decode_samples) * batch_size
    throughput = (
        generated_after_prefill / (sum(decode_samples) / 1000.0) if decode_samples else 0.0
    )
    return {
        "batch_size": batch_size,
        "prompt_length": prompt_length,
        "decode_tokens": decode_tokens,
        "ttft_model_only_ms": prefill_stats.p50_ms,
        "prefill": prefill_stats.to_dict(),
        "tpot_ms": median(decode_samples) if decode_samples else 0.0,
        "decode_mean_ms": decode_mean,
        "decode_p95_ms": _percentile(decode_samples, 95) if decode_samples else 0.0,
        "decode_tokens_per_second": throughput,
        "kv_cache_bytes": cache_nbytes(cache),
        "peak_runtime_delta_bytes": peak_delta,
    }


def benchmark_grid(model, batch_sizes, prompt_lengths, **kwargs) -> list[dict]:
    return [
        benchmark_single_case(
            model,
            batch_size=batch_size,
            prompt_length=prompt_length,
            **kwargs,
        )
        for batch_size in batch_sizes
        for prompt_length in prompt_lengths
    ]
