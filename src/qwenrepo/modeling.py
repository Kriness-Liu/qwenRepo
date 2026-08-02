"""Portable loading and forward helpers across Transformers 4.x/5.x."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct"


def resolve_model_path(model_path: str | Path | None = None) -> Path:
    path = DEFAULT_MODEL_PATH if model_path is None else Path(model_path)
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"model path does not exist: {resolved}; place Qwen2.5-0.5B-Instruct "
            "under models/ or pass --model-path"
        )
    return resolved


def _transformers_major() -> int:
    text = transformers.__version__.split(".", 1)[0]
    return int(text) if text.isdigit() else 4


def load_model_and_tokenizer(
    model_path: str | Path | None = None,
    *,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
):
    path = resolve_model_path(model_path)
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    selected_dtype = dtype
    if selected_dtype is None:
        selected_dtype = torch.float16 if selected_device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    load_kwargs = {"local_files_only": True}
    if _transformers_major() >= 5:
        load_kwargs["dtype"] = selected_dtype
    else:
        load_kwargs["torch_dtype"] = selected_dtype
    model = AutoModelForCausalLM.from_pretrained(path, **load_kwargs)
    model = model.to(selected_device).eval()
    return model, tokenizer, selected_device


def forward_last_logits(model, **kwargs):
    """Request only the final logits when the installed model supports it."""

    if "logits_to_keep" in inspect.signature(model.forward).parameters:
        kwargs.setdefault("logits_to_keep", 1)
    return model(**kwargs)
