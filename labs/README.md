# Labs

本目录用于按顺序补充大模型机制实验。每个实验应尽量做到：先预测结果，只改变一个变量，使用断言或Reference验证，并记录环境与结论边界。

建议顺序：

1. `00_tokenizer_chat_template.py`：Tokenizer、Chat Template与Token切片。
2. `01_model_architecture.py`：Embedding、Transformer Block、RMSNorm与LM Head。
3. `02_attention_shapes.py`：Q/K/V、GQA、Mask与Attention张量shape。
4. `03_rope.py`：RoPE位置编码的旋转过程。
5. `04_kv_cache.py`：无Cache与使用KV Cache的结果和开销对比。
6. `05_prefill_decode.py`：区分Prefill与逐Token Decode。
7. `06_generation_sampling.py`：Greedy、Temperature、Top-k与Top-p。
8. `07_inference_profile.py`：延迟、吞吐、显存与Profiler分析。
