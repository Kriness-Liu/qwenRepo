# qwenRepo：从模型结构到推理性能

这是一个以本地 `Qwen2.5-0.5B-Instruct` 为对象的 AI Infra 学习仓库。代码不追求重新实现完整推理引擎，而是用可验证的小实验拆解：GQA、KV Cache、Prefill/Decode、TTFT/TPOT、吞吐与显存，以及 PyTorch Profiler 的性能归因。

## 你能学到什么

- 从 Q/K/V 张量形状理解 Grouped Query Attention（GQA）及 KV Head 共享。
- 验证带 KV Cache 的逐 Token Decode 与完整重算在数值上等价。
- 推导并实测 KV Cache 显存：`2 × layers × batch × kv_heads × sequence × head_dim × element_size`。
- 区分 Prefill 与 Decode：前者通常更偏计算密集，后者通常更受访存和调度开销影响。
- 使用 CUDA Event 测量模型侧 TTFT、TPOT 和 Decode throughput，避免把 CPU 异步提交时间误当作 GPU 执行时间。
- 使用 PyTorch Profiler 和 Chrome Trace 分析算子、张量形状、CPU/GPU 时间及显存分配。

## 目录

```text
qwenRepo/
├─ src/qwenrepo/       # 可复用的模型、GQA、KV Cache 和 benchmark 工具
├─ labs/               # 按学习顺序组织的实验
├─ scripts/            # 环境检查、推理、批量基准入口
├─ tests/              # 数值正确性、公式与真实模型集成测试
├─ results/            # 可复现结果说明；latest.json 不提交
└─ models/             # 本地权重，不纳入 Git
```

## 环境

建议使用独立环境。先按显卡和驱动安装 PyTorch，再安装其余依赖和当前仓库：

```powershell
conda create -n qwenrepo python=3.12 -y
conda activate qwenrepo
python -m pip install -r requirements.txt
python -m pip install -e .
```

将模型放在 `models/Qwen2.5-0.5B-Instruct/`，也可以通过每个脚本的 `--model-path` 指定路径。仓库仅保存实验代码，不重新分发模型权重。

## 推荐学习顺序

```powershell
python labs/00_tokenizer_chat_template.py
python labs/01_attention_gqa.py
python labs/02_kv_cache.py
python labs/03_prefill_decode.py --batch-size 1 --prompt-length 128
python labs/04_benchmark_grid.py --batch-sizes 1 2 --prompt-lengths 32 128 512
python labs/05_inference_profile.py --prompt-length 128
python -m unittest discover -s tests -v
```

真实模型集成测试默认跳过；显式提供本地模型路径后运行：

```powershell
$env:QWEN_MODEL_PATH="$PWD\models\Qwen2.5-0.5B-Instruct"
python -m unittest tests.test_model_integration -v
```

## 指标口径

- `ttft_model_only_ms`：仅测一次 Prefill forward 的 GPU 时间，不包含 tokenizer、排队、网络与服务框架。
- `tpot_ms`：使用 KV Cache 串行 Decode 时，每个后续 Token 的 GPU 时间中位数。
- `decode_tokens_per_second`：批内产生的 Token 数除以 Decode GPU 总时间，并非在线服务端到端吞吐。
- `kv_cache_bytes`：实际 Cache Tensor 的存储字节数。
- `peak_runtime_delta_bytes`：该 Case 相对测量前已分配显存的峰值增量。

使用合成 Token ID 是为了稳定控制 Batch Size 和 Context Length；结果适合比较形状趋势，不等价于线上业务压测。

## 当前边界

- 当前覆盖单机单卡、Hugging Face Transformers 推理链路，不包含 vLLM、TensorRT-LLM 或在线 Serving。
- 尚未覆盖 PagedAttention、Continuous Batching、量化、Tensor Parallel 和多机通信。
- 所有性能结论都应注明 GPU、PyTorch/Transformers/CUDA 版本、dtype、shape、预热和采样次数。
