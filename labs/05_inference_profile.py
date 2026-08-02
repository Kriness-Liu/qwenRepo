"""Experiment 05: export a PyTorch Profiler trace for prefill and decode."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from qwenrepo.modeling import forward_last_logits, load_model_and_tokenizer


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("results/qwen_profile.trace.json"))
    args = parser.parse_args()
    model, _, device = load_model_and_tokenizer(args.model_path)
    input_ids = torch.randint(
        0, int(model.config.vocab_size), (1, args.prompt_length), device=device
    )
    # Keep one-time module/kernel loading out of the trace so the report reflects
    # the steady-state prefill/decode path rather than framework initialization.
    warmup = forward_last_logits(model, input_ids=input_ids, use_cache=True, return_dict=True)
    warmup_token = warmup.logits[:, -1:].argmax(dim=-1)
    forward_last_logits(
        model,
        input_ids=warmup_token,
        attention_mask=torch.ones(1, args.prompt_length + 1, device=device, dtype=torch.long),
        past_key_values=warmup.past_key_values,
        use_cache=True,
        return_dict=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        with record_function("qwen_prefill"):
            prefill = forward_last_logits(
                model, input_ids=input_ids, use_cache=True, return_dict=True
            )
        next_token = prefill.logits[:, -1:].argmax(dim=-1)
        with record_function("qwen_decode"):
            forward_last_logits(
                model,
                input_ids=next_token,
                attention_mask=torch.ones(1, args.prompt_length + 1, device=device, dtype=torch.long),
                past_key_values=prefill.past_key_values,
                use_cache=True,
                return_dict=True,
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.output))
    print(
        profiler.key_averages(group_by_input_shape=True).table(
            sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
            row_limit=20,
        )
    )
    print("saved:", args.output)


if __name__ == "__main__":
    main()
