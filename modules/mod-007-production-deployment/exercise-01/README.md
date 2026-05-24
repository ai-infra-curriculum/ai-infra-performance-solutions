# Framework Selection — Solution

| Model | Framework | Why |
|---|---|---|
| Mistral-7B-Instruct | vLLM | autoregressive; continuous batching wins big |
| ResNet-50 | Triton | high-QPS, request/response, dynamic batching |
| Sentence Transformer | Triton or FastAPI | low compute per request; latency-sensitive |
| Llama-3-70B | vLLM with TP=4 | LLM at scale |
| Stable Diffusion | Triton | multi-stage pipeline; controlled batching |
| Custom RAG composition | Ray Serve | multi-step graph; framework-aware |
