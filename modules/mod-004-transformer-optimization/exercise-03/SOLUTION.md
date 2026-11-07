# Exercise 03 — vLLM + Automatic Prefix Caching

## 1. Solution overview

The exercise asks learners to launch vLLM with automatic prefix
caching enabled, send two contrasting workloads through it — one
with a shared system prompt and one with unique prompts — and
compare throughput, time-to-first-token (TTFT), and the reported
cache hit rate. The deliverable is the launch command plus a client
harness and a short interpretation of the numbers.

Key teaching points:

- vLLM's paged-attention KV cache is addressed by physical blocks,
  so KV blocks produced by one request can be reused by any later
  request whose prompt prefix hashes identically up to a block
  boundary.
- Prefix caching is a **workload-dependent** optimization. If
  prompts do not share a prefix, enabling it cannot help and may
  cost scheduling overhead for no benefit.
- The win shows up most strongly on **TTFT** and **prefill
  throughput**, because prefix caching saves prefill work; decode
  throughput per token is not changed.

## 2. Worked answer or implementation

### 2.1 Server launch

```bash
# modules/mod-004-transformer-optimization/exercise-03/launch.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-mistralai/Mistral-7B-Instruct-v0.2}"

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --enable-prefix-caching \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --port 8000
```

Notes:

- `--enable-prefix-caching` turns on the automatic prefix-cache
  path (prompt hashing + block reuse). No code change on the client
  side is needed; identical prefixes across requests are detected
  by the server.
- `--gpu-memory-utilization` controls how much free GPU memory vLLM
  claims for the KV cache. A larger KV cache means more cached
  prefixes can co-exist with active requests; this is the main
  resource that prefix caching competes for.
- For an apples-to-apples comparison, run a second server with the
  flag removed to produce the "no prefix caching" baseline. Do not
  compare across different `--max-model-len` or
  `--gpu-memory-utilization` values.

### 2.2 Client harness

```python
# modules/mod-004-transformer-optimization/exercise-03/bench.py
import asyncio
import time
from statistics import mean, median

import httpx

URL = "http://127.0.0.1:8000/v1/completions"
MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
CONCURRENCY = 16
N_REQUESTS = 200
MAX_TOKENS = 64


SYSTEM_PROMPT = (
    # A deliberately long shared prefix so every request hits the
    # same KV blocks on the server.
    "You are a careful assistant. "
    + ("Always cite sources. " * 400)
)


async def one(client: httpx.AsyncClient, user_prompt: str) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        URL,
        json={
            "model": MODEL,
            "prompt": SYSTEM_PROMPT + user_prompt,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=120.0,
    )
    r.raise_for_status()
    body = r.json()
    return {
        "latency_s": time.perf_counter() - t0,
        "out_tokens": body["usage"]["completion_tokens"],
    }


async def run(prompts: list[str]) -> None:
    sem = asyncio.Semaphore(CONCURRENCY)

    async with httpx.AsyncClient() as client:
        async def guarded(p):
            async with sem:
                return await one(client, p)

        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded(p) for p in prompts))
        wall = time.perf_counter() - t0

    out_tokens = sum(r["out_tokens"] for r in results)
    lats = [r["latency_s"] for r in results]
    print(f"requests:         {len(results)}")
    print(f"wall clock (s):   {wall:7.2f}")
    print(f"throughput tok/s: {out_tokens / wall:7.1f}")
    print(f"latency p50 (s):  {median(lats):7.2f}")
    print(f"latency mean (s): {mean(lats):7.2f}")


if __name__ == "__main__":
    shared = [f" Question {i}: summarise paragraph {i}." for i in range(N_REQUESTS)]
    unique = [f" Question {i}: {'xyz ' * (i % 50)} explain." for i in range(N_REQUESTS)]
    print("--- shared-prefix workload ---")
    asyncio.run(run(shared))
    print("--- unique-prompt workload ---")
    asyncio.run(run(unique))
```

Decision rationale:

- The shared workload reuses `SYSTEM_PROMPT` verbatim across every
  request so the prefix hash hits on all but the first call. This
  makes the hit-rate ceiling explicit.
- The unique workload keeps the same short system preamble but
  varies the body enough that most requests miss the prefix cache.
  Because of the vLLM block-level hashing, differing content past
  the first block kills reuse — this behaviour is what the exercise
  wants learners to observe.
- Concurrency is set high enough (16) that prefill queueing is
  visible but low enough to fit comfortably in KV cache on a single
  7B model.
- `temperature=0.0` removes sampling noise from the latency
  comparison.

### 2.3 Expected shape of results

With prefix caching enabled and a long shared system prompt,
expect:

- **Shared workload**: large reduction in TTFT after the first
  request; wall-clock throughput several times higher than the
  unique-workload run on the same server.
- **Unique workload**: throughput and latency essentially matching
  a server launched without `--enable-prefix-caching`.
- **vLLM metrics endpoint**
  (`http://127.0.0.1:8000/metrics`) should report
  `vllm:gpu_prefix_cache_hit_rate` near the shared-workload
  ceiling and near zero for unique prompts.

Exact speedup numbers depend on model, GPU, prefix length, and
concurrency; the module SOLUTION notes the qualitative shape and
expects learners to record *their own* measured numbers.

## 3. Validation steps

1. Health-check the server:
   `curl -s http://127.0.0.1:8000/v1/models | jq .`
2. Confirm the flag is active via the metrics endpoint:
   `curl -s http://127.0.0.1:8000/metrics | grep prefix_cache`.
   The metric name on recent vLLM releases is
   `vllm:gpu_prefix_cache_hit_rate` (and a counter variant). It
   must be present; if it is absent, prefix caching was not
   enabled.
3. Run `bench.py` against the prefix-caching-enabled server for
   both workloads. Record throughput, p50 latency, and hit rate.
4. Restart the server **without** `--enable-prefix-caching` and
   repeat for the shared workload. The delta between the two runs
   is the real speedup.
5. Sanity-check correctness: with `temperature=0.0`, a given prompt
   must produce identical completions across runs regardless of
   caching. Diff a few completions to confirm.
6. Record the configuration alongside every number: model, dtype,
   GPU, `--max-model-len`, `--gpu-memory-utilization`,
   concurrency.

## 4. Rubric or review checklist

| Check | Weight | Pass criterion |
|---|---|---|
| Launches vLLM with `--enable-prefix-caching` | 10 | Command recorded verbatim; nothing else swapped between the on/off comparison |
| Produces two workloads (shared and unique) | 15 | Shared workload uses a long, byte-identical prefix; unique workload demonstrably differs early in the prompt |
| Reports throughput and latency for each | 20 | Throughput (tok/s) and p50 latency reported for both workloads, at the same concurrency |
| Reads vLLM metrics to confirm hit rate | 15 | `vllm:gpu_prefix_cache_hit_rate` cited; high for shared, near zero for unique |
| Baseline vs enabled comparison | 15 | Same workload run with the flag off; delta explicit |
| Workload context recorded | 10 | Model, GPU, dtype, `--max-model-len`, `--gpu-memory-utilization`, concurrency, prompt length |
| Correctness check | 10 | Deterministic completions identical between on/off runs |
| Honest interpretation | 5 | Does not claim a workload-independent speedup |

## 5. Common mistakes

1. **Claiming the "vLLM is 3x faster" number as universal.** The
   speedup from prefix caching is an upper bound driven by the hit
   rate. For workloads without shared prefixes, enabling the flag
   produces no benefit.
2. **Comparing runs with different `--gpu-memory-utilization`.**
   Larger KV cache almost always raises steady-state throughput on
   its own. Hold every other flag fixed when isolating the effect
   of prefix caching.
3. **Mutating the prefix per request.** Even a single changed
   character at the start of the prompt pushes the prefix hash into
   a new bucket. Prefixes must be byte-identical across requests to
   hit. Timestamps and request IDs in the prefix defeat the
   optimization.
4. **Measuring only aggregate throughput.** Prefix caching reduces
   prefill work, so the clearest signal is TTFT (time-to-first
   token) at a given concurrency. Report that alongside tok/s.
5. **Ignoring the KV-cache budget.** Blocks reserved for prefix
   caching are blocks unavailable for active request KV. On a
   memory-tight deployment, the optimization can regress tail
   latency even when average throughput improves.
6. **Benchmarking a cold server.** The first request on any prefix
   is a miss. Warm the cache with a dummy request before the timed
   run or expect the first few samples to look like a cache miss.

## 6. References

- vLLM docs — Automatic Prefix Caching:
  <https://docs.vllm.ai/en/latest/features/automatic_prefix_caching.html>
- vLLM docs — Serving with the OpenAI-compatible API:
  <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- vLLM docs — Engine arguments
  (`--enable-prefix-caching`, `--gpu-memory-utilization`,
  `--max-model-len`):
  <https://docs.vllm.ai/en/latest/serving/engine_args.html>
- vLLM docs — Production metrics (Prometheus endpoint):
  <https://docs.vllm.ai/en/latest/serving/metrics.html>
- vLLM paper — "Efficient Memory Management for Large Language
  Model Serving with PagedAttention" (Kwon et al., 2023):
  <https://arxiv.org/abs/2309.06180>
- Module-level context:
  `modules/mod-004-transformer-optimization/SOLUTION.md`
  ("Decision 3: vLLM prefix caching as the production lever")
  and the baseline server launch in
  `modules/mod-004-transformer-optimization/exercise-03/README.md`.
