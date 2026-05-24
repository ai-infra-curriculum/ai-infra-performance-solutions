# Tensor Parallel — Solution

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 2 \
  --port 8000
```

Requires NVLink-connected GPUs for reasonable performance.
