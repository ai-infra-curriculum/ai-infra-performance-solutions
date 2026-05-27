# SOLUTION — Distributed Inference

> Read this *after* you have served a model that doesn't fit on
> one GPU. This document explains *why* the distributed inference
> stack is shaped the way it is and which parallelism strategy to
> reach for first.

## What this module is really teaching

Once a model exceeds one GPU's memory (Llama-3 70B in FP16 needs
140 GB; an H100 has 80), you have to choose a parallelism strategy.
The choices look like a menu, but most are wrong for most
workloads:

| Strategy | When it wins | When it loses |
|---|---|---|
| **Tensor parallel (TP)** | Single-host inference, low latency required | Cross-host: NVLink → NVL switch → InfiniBand kills perf |
| **Pipeline parallel (PP)** | Cross-host, training | Inference: pipeline bubbles destroy latency |
| **Expert parallel (EP)** | MoE models only | Dense models: no benefit |
| **Replica scaling (HPA)** | Most real production | When model doesn't fit on 1 device, can't help alone |

The right production answer for a 70B model in 2026 is usually
**TP within a node + HPA across nodes**. The module exists to
make that conclusion feel earned, not asserted.

## Architectural decisions and *why*

### Decision 1: Tensor parallel first — because it's the production default

Exercise 01 (tensor parallel) implements TP across 4 H100s on one
NVLink-connected host. The reference solution uses vLLM's built-in
TP support (`tensor_parallel_size=4`). The measured latency is
typically 60-70% of single-GPU latency at 4x the model capacity —
a real win.

The "why" of TP-within-a-node: NVLink between H100s on the same
host runs at 900 GB/s per link, fast enough that the all-reduce
after each transformer block is amortized across the layer
computation. Once you cross hosts (NVL switch → InfiniBand at
50 GB/s), the all-reduce becomes the bottleneck and TP
catastrophically loses.

**Anti-pattern to avoid**: configuring `tensor_parallel_size=8`
across two 4-GPU hosts. The all-reduce traffic crosses the slow
link every layer, and you end up at 30% of the speed of TP=4 on
one host.

### Decision 2: Pipeline parallel for completeness, but flagged as "rarely right for inference"

Exercise 02 (pipeline parallel) is in the curriculum because PP is
the only way to scale very large models across many hosts during
**training** (where the bubble cost is amortized over the backward
pass). For inference, PP introduces a "pipeline bubble" — the time
between when stage N+1 finishes its work and starts on the next
request — that translates directly to user-facing latency.

The reference solution measures both throughput and latency under
PP and shows the latency penalty (typically 1.5-2x worse than TP)
clearly. The teaching: PP is a training tool. For inference, use
TP within a node and replicas across nodes.

### Decision 3: Custom HPA metric — because CPU utilization is the wrong signal

Exercise 03 (custom HPA metric) replaces Kubernetes' default
CPU-based HPA with a custom metric: **queue depth at the inference
gateway**. The reason: GPU inference saturates the GPU (so GPU
utilization spikes early) while CPU stays low. CPU-based HPA never
triggers. Queue-depth-based HPA scales the moment requests start
queueing, which is the symptom that matters to users.

The reference solution uses Prometheus + the HPA external metrics
adapter to expose `inference_queue_depth` and configures the HPA
to keep it under 5. This is the **standard production pattern** in
2026, but most internet tutorials still show CPU-based HPA. The
exercise exists to flag the gap.

**Anti-pattern to avoid**: scaling on `GPU utilization > 80%`.
At 80% you're already throttled; users see latency spikes for the
30-60 seconds it takes to spin up a new replica. Scale on a
*leading* indicator (queue depth, p95 latency) instead.

### Decision 4: Prefix-aware routing as the cache-hit-rate optimizer

Exercise 04 (prefix-aware routing) pins requests with similar
prompt prefixes to the same replica so the KV cache hit rate
stays high. Without it, replicas are picked round-robin and the
prefix cache is wasted on every replica.

The implementation uses a **consistent hash** over the first K
tokens of the prompt to choose a replica. For chat workloads with
a shared system prompt, this drives the prefix cache hit rate
from ~5% (round-robin) to ~95% (prefix-routed). The throughput
improvement is 2-5x on long shared prompts.

The reference solution emits the cache hit rate as a metric, then
shows the latency CDF before and after the routing change. Without
those metrics, the deployment is "vibes-based" and you can't tune.

### Decision 5: Cold-start mitigation — the operational reality check

Exercise 05 (cold-start mitigation) addresses what happens when a
new replica spins up: 30-90 seconds of loading a 70B model from
S3, then a JIT/CUDA-graph warm-up that takes another 30-60s. For
that window, traffic routed to the new replica gets terrible
latency.

The reference solution implements:
1. **Pre-warmed pod template** with model weights baked into a
   PVC mounted read-only.
2. **Readiness probe gated on a warm-up health check** that runs a
   real inference end-to-end before the pod accepts traffic.
3. **Surge capacity buffer** — HPA keeps N+1 replicas so a new pod
   warming up doesn't gate new traffic.

This is the "operational tax" of LLM inference that pure
performance engineering ignores. Mod-007 (production deployment)
extends this with spot-instance resilience.

## Trade-offs we deliberately accepted

### NCCL as the only collective library

The reference solutions use NCCL. RCCL (AMD), oneCCL (Intel), and
Gloo (CPU) exist but are not relevant for the H100-centric
deployments the module targets. Mod-008 ex-03 covers NCCL tuning
in more depth.

### vLLM-only orchestration

The exercises use vLLM as the inference runtime. TensorRT-LLM has
better single-GPU throughput; SGLang has better complex routing.
The choice of vLLM is pragmatic: the API is the most stable and
the multi-GPU primitives are well-documented. Mod-007 covers the
selection criteria.

### Replica scaling caps at 64

The HPA configurations cap at 64 replicas. Past that, you need
**fleet management** (sharding by tenant, model, or region) which
is its own engineering problem. Mod-007 ex-04 covers multi-tier
routing for that scale.

## Common mistakes graders see

1. **TP across hosts with InfiniBand**: every all-reduce is now
   network-bound. Either TP within a node + replicas across, or
   PP across.
2. **Scaling on GPU utilization**: late signal; scale-up arrives
   after the queue has already backed up.
3. **Ignoring KV cache memory in capacity math**: at high QPS with
   long context, KV cache can be 10x the model size. Reserve
   accordingly.
4. **Round-robin routing with prefix caching enabled**: you've
   paid the memory cost without getting the hit rate.
5. **Cold-start gated on liveness probe, not readiness**: new
   pods take traffic before they're warm and time out the user.
6. **Not measuring p99 latency under load**: averages hide the
   pipeline-bubble and queue-buildup behavior that matters most.

## When to go beyond this implementation

- Add **disaggregated prefill** (separate replicas for prefill vs
  decode). Prefill is compute-bound; decode is memory-bound; they
  benefit from different hardware allocation.
- Implement **request-level dynamic batching** that chunks long
  prefills across multiple decode steps.
- Move to a **PD-separated** architecture (Mooncake, DistServe) for
  workloads where the prefill/decode mismatch is severe.

## Related curriculum touchpoints

- `performance/mod-004-transformer-optimization` — the
  single-GPU optimizations these distributed strategies build on.
- `performance/mod-007-production-deployment` — production
  rollout, canary, and spot resilience for these architectures.
- `engineer/mod-110-llm-infrastructure/exercise-01-production-llm-serving`
  — the API gateway that uses these strategies.
- `architect/projects/project-303-llm-rag-platform` — the
  enterprise architecture that wraps all this.
