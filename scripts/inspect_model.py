from pathlib import Path

from transformers import AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct")


def main():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    print(model)


if __name__ == "__main__":
    main()
