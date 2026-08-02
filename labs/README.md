# Labs 学习顺序

每个实验都遵循四步：先预测结果，只改变一个变量，用 reference 或断言检查正确性，最后记录结论成立的环境和边界。

1. `00_tokenizer_chat_template.py`：Tokenizer、Chat Template 与 Token ID。
2. `01_attention_gqa.py`：Q/K/V Head 映射、GQA 与 SDPA 数值等价性。
3. `02_kv_cache.py`：增量 Decode、完整重算与 KV Cache 显存公式。
4. `03_prefill_decode.py`：模型侧 TTFT、TPOT、吞吐与运行时显存。
5. `04_benchmark_grid.py`：Batch Size/Context Length 扫描与结果持久化。
6. `05_inference_profile.py`：PyTorch Profiler、标注区间和 Chrome Trace。

建议不要只看输出：修改 Batch Size、Prompt Length 和 Decode Tokens，解释曲线变化，再对照 Profiler 判断瓶颈来自算子执行、访存、Kernel 调度还是 CPU/GPU 空隙。
