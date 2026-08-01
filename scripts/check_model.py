from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct")


def main():
    print("========== Environment ==========")

    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用，请先检查 PyTorch CUDA 环境。")

    print("GPU:", torch.cuda.get_device_name(0))

    print(
        "Total GPU memory:",
        torch.cuda.get_device_properties(0).total_memory / 1024**3,
        "GiB",
    )

    print("\n========== Loading tokenizer ==========")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print("Tokenizer loaded.")

    print("\n========== Loading model ==========")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16,
        local_files_only=True,
    )

    print("Model loaded into CPU memory.")

    model = model.to("cuda")

    print("Model moved to GPU.")

    num_params = sum(p.numel() for p in model.parameters())

    print("\n========== Model information ==========")

    print("Parameters:", num_params)
    print("Parameters (B):", num_params / 1e9)

    print(
        "CUDA allocated:",
        torch.cuda.memory_allocated() / 1024**3,
        "GiB",
    )

    print(
        "CUDA reserved:",
        torch.cuda.memory_reserved() / 1024**3,
        "GiB",
    )


if __name__ == "__main__":
    main()
