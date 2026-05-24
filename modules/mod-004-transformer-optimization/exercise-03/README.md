# vLLM Prefix Caching — Solution

```bash
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --enable-prefix-caching --port 8000
```

See [engineer-solutions/mod-110 ex-03 BENCHMARKS.md](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/blob/main/modules/mod-110-llm-infrastructure/exercise-03-vllm-deep-dive/BENCHMARKS.md) for measured 2-3× throughput gain on shared-prompt workloads.
