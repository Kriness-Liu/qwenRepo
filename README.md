# qwenRepo

一个从 `Qwen2.5-0.5B-Instruct` 出发学习大语言模型结构、推理链路与性能特征的实验仓库。当前阶段先建立可解释、可运行的单卡推理基线，再逐步拆解 Transformer 组件、KV Cache、Prefill/Decode 与推理性能分析。

## 学习目标

- 理解 Tokenizer、Chat Template、Token ID 与文本生成之间的关系。
- 观察 Qwen2.5 的 Embedding、RMSNorm、RoPE、GQA、SwiGLU 和 LM Head 结构。
- 理解从 Prompt 到 Tokenize、Forward、Logits、Decode 的完整推理链路。
- 分析参数、激活与 KV Cache 的显存占用，以及 Prefill/Decode 的性能差异。
- 逐步建立延迟、吞吐、显存和正确性可复现的评测方法。

## 当前内容

| 文件 | 内容 |
|---|---|
| `scripts/check_model.py` | 检查CUDA环境，加载本地模型并统计参数量与显存占用 |
| `scripts/inspect_model.py` | 打印模型模块树，观察Qwen2.5网络结构 |
| `scripts/first_inference.py` | 完成Chat Template、Tokenize、Generate与Decode推理链路 |
| `labs/` | 后续按主题补充的可修改实验 |

## 目录结构

```text
qwenRepo/
├─ scripts/       # 环境检查、模型检查与基础推理脚本
├─ labs/          # 按学习顺序组织的实验
├─ models/        # 本地模型权重，不纳入Git
├─ requirements.txt
└─ README.md
```

## 环境准备

建议使用独立环境：

```powershell
conda create -n qwenrepo python=3.12 -y
conda activate qwenrepo
```

先根据本机GPU和CUDA驱动，从PyTorch官方安装选择器获取对应命令，再安装其余依赖：

```powershell
python -m pip install -r requirements.txt
```

## 准备模型

将 `Qwen/Qwen2.5-0.5B-Instruct` 下载到：

```text
models/Qwen2.5-0.5B-Instruct/
```

`models/` 已加入 `.gitignore`，约1 GB的权重和本机缓存不会进入Git历史。三个脚本都从仓库根目录解析模型路径，不依赖某台电脑的绝对路径。

## 运行顺序

在仓库根目录执行：

```powershell
python scripts/check_model.py
python scripts/inspect_model.py
python scripts/first_inference.py
```

建议每次实验都记录：运行环境、输入shape和dtype、预期结果、实测结果、显存变化及结论边界。

## 学习路线

1. Tokenizer、Chat Template与生成结果切片。
2. Transformer Block、GQA、RoPE、RMSNorm与SwiGLU。
3. Attention张量shape与Mask传播。
4. KV Cache及Prefill/Decode阶段差异。
5. Greedy、Sampling、Top-k与Top-p生成策略。
6. PyTorch Profiler下的算子耗时与显存归因。
7. Batch Size、上下文长度、吞吐与延迟权衡。
8. 量化、推理引擎和服务化作为后续扩展。

## 当前边界

- 当前只覆盖本地单卡推理学习，不包含预训练、微调或分布式训练。
- 尚未实现vLLM、TensorRT-LLM、量化或在线服务性能对比。
- 模型权重来自上游项目，仓库只保存学习代码，不重新分发权重。
