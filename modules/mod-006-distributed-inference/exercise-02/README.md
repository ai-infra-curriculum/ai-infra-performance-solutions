# Pipeline Parallel — Solution

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --port 8000
```

Requires multi-node IB fabric for inter-node activations.
