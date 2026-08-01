from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = str(REPO_ROOT / "models" / "Qwen2.5-0.5B-Instruct")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Device:", device)

    # -------------------------------
    # 1. 加载 tokenizer
    # -------------------------------
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        local_files_only=True,
    )

    # -------------------------------
    # 2. 加载模型
    # -------------------------------
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        local_files_only=True,
    )

    model = model.to(device)
    model.eval()

    # -------------------------------
    # 3. 构造对话
    # -------------------------------
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {
            "role": "user",
            "content": "请用简单的话解释什么是GPU。",
        },
    ]

    # -------------------------------
    # 4. Chat Template + Tokenizer
    # -------------------------------
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(device)

    print("\n========== Input ==========")

    print("input_ids:")
    print(inputs["input_ids"])

    print("\ninput_ids shape:")
    print(inputs["input_ids"].shape)

    print("\ninput_ids device:")
    print(inputs["input_ids"].device)

    # -------------------------------
    # 5. 模型生成
    # -------------------------------
    print("\n========== Generating ==========")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 输入本身也包含在 outputs 里
    # 所以把原来的 prompt token 切掉
    generated_tokens = outputs[
        0,
        inputs["input_ids"].shape[1]:
    ]

    # -------------------------------
    # 6. Token -> Text
    # -------------------------------
    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    print("\n========== Qwen Answer ==========")

    print(answer)


if __name__ == "__main__":
    main()
