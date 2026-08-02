# Results

Committed result files must include the model, GPU, PyTorch/Transformers versions, dtype, timing scope, warm-up/iteration counts and all benchmark inputs. `latest.json` and profiler traces are ignored because they are local rerun artifacts.

For this repository:

- TTFT means model-only prefill latency; tokenization and network latency are excluded.
- TPOT is the median cached decode latency after the first token.
- Decode throughput counts generated tokens across the whole batch.
- KV Cache bytes are counted from the actual key/value tensors, not from reserved CUDA memory.
