# HPA on Custom Metric — Solution

`hpa.yaml` scales on `vllm_num_requests_waiting`. Requires:
- Prometheus scraping vLLM `/metrics`
- Prometheus Adapter exposing the metric to k8s API
- Fast scale-up (0 stabilization) + slow scale-down (10 min)
