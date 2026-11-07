# Exercise 04 — Speculative Decoding

## 1. Solution overview

The exercise asks learners to stand up a target model together with
a smaller draft model, enable speculative decoding, measure decoding
latency and throughput against the no-speculation baseline, and
record the **acceptance rate**. The deliverable is the configuration
plus a client-side analysis that connects acceptance rate to the
observed speedup.

Key teaching points:

- Speculative decoding is exact. The large model verifies every
  draft token, so the output distribution matches greedy or
  temperature sampling from the target alone. It is not an
  approximation.
- The win is latency per generated token at low batch, where the
  target model is memory-bound reading its own weights. The draft
  model turns one target forward pass into up to *k* accepted
  tokens.
- The speedup is bounded by the acceptance rate and by the latency
  ratio between draft and target. Without measuring both, a
  deployment can regress silently on a domain the draft model has
  not seen.

## 2. Worked answer or implementation

### 2.1 Server launch (vLLM)

vLLM exposes speculative decoding through engine arguments. The
exact option names changed across releases; the reference below
uses the recent form (`--speculative-model` plus
`--num-speculative-tokens`). Pin the vLLM version in the module
`VERSIONS.md` and prefer the engine-arg form available in that
version.

```bash
# modules/mod-004-transformer-optimization/exercise-04/launch.sh
#!/usr/bin/env bash
set -euo pipefail

TARGET="${TARGET:-meta-llama/Llama-3.1-8B-Instruct}"
DRAFT="${DRAFT:-meta-llama/Llama-3.2-1B-Instruct}"
K="${K:-5}"                     # proposed tokens per step

python -m vllm.entrypoints.openai.api_server \
  --model "$TARGET" \
  --speculative-model "$DRAFT" \
  --num-speculative-tokens "$K" \
  --gpu-memory-utilization 0.90 \
  --max-model-len 8192 \
  --port 8000
```

Why these specific choices:

- **Draft from the same family as target**: a draft model that
  shares the target's tokenizer and training distribution produces
  a much higher acceptance rate than an arbitrary small model. A
  mismatched tokenizer disables the optimization entirely in vLLM.
- **`--num-speculative-tokens` (k) between 3 and 7**: smaller `k`
  under-exploits the draft; larger `k` wastes draft tokens that
  the target is likely to reject late in the proposal. The optimum
  depends on acceptance rate — start at `k=5` and sweep.
- **Greedy or near-greedy decoding during benchmarking**: sampling
  temperature affects acceptance rate; keep it fixed across baseline
  and speculative runs so the comparison isolates the algorithm.

### 2.2 Client harness and metric extraction

```python
# modules/mod-004-transformer-optimization/exercise-04/bench.py
import asyncio
import statistics
import time

import httpx

URL = "http://127.0.0.1:8000/v1/completions"
METRICS = "http://127.0.0.1:8000/metrics"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONCURRENCY = 4
N_REQUESTS = 50
MAX_TOKENS = 256
PROMPT = (
    "You are a helpful assistant. "
    "Explain, step by step, how to compute the dot product of two "
    "vectors in Python using only standard library features."
)


async def one(client: httpx.AsyncClient) -> dict:
    t0 = time.perf_counter()
    r = await client.post(
        URL,
        json={
            "model": MODEL,
            "prompt": PROMPT,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=300.0,
    )
    r.raise_for_status()
    body = r.json()
    return {
        "latency_s": time.perf_counter() - t0,
        "out_tokens": body["usage"]["completion_tokens"],
    }


async def run() -> None:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as c:
        async def guarded():
            async with sem:
                return await one(c)
        t0 = time.perf_counter()
        results = await asyncio.gather(*(guarded() for _ in range(N_REQUESTS)))
        wall = time.perf_counter() - t0

        metrics = (await c.get(METRICS, timeout=30.0)).text

    out = sum(r["out_tokens"] for r in results)
    lats = [r["latency_s"] for r in results]
    per_tok = [r["latency_s"] / r["out_tokens"] for r in results if r["out_tokens"]]

    print(f"wall clock (s):        {wall:7.2f}")
    print(f"throughput tok/s:      {out / wall:7.1f}")
    print(f"latency p50 (s):       {statistics.median(lats):7.2f}")
    print(f"mean s/token:          {statistics.mean(per_tok):7.4f}")

    # Pull speculative-decoding metrics. Metric names on recent vLLM
    # releases include acceptance rate and accepted-token counters;
    # names can shift across versions, so grep broadly and log what
    # is actually emitted rather than hard-coding.
    for line in metrics.splitlines():
        if "spec" in line.lower() or "draft" in line.lower() or "accept" in line.lower():
            print(line)


if __name__ == "__main__":
    asyncio.run(run())
```

Decision rationale:

- Benchmarks are **per-token** latency, not per-request. Speculative
  decoding affects per-token work; a per-request number is confounded
  by varying `completion_tokens`.
- The client dumps every Prometheus line matching `spec`, `draft`,
  or `accept` instead of hard-coding a metric name, because vLLM
  renames spec-decode metrics across releases and the exercise should
  not silently miss a renamed counter. Record whichever name the
  running server emits.
- A low-but-nonzero concurrency (4) keeps the server in the latency
  regime where speculative decoding helps most, without the single-
  request pathology that masks queueing effects.

### 2.3 Expected shape of results

- Baseline (same server launched without the draft) and speculative
  runs should produce byte-identical completions at `temperature=0.0`.
  Diff them.
- Speculative run should show lower mean seconds-per-token on
  domains where the draft tracks the target's distribution well. On
  adversarial domains (random strings, code in a language the draft
  does not know), it can be a slowdown.
- The reported acceptance rate — accepted draft tokens divided by
  proposed draft tokens — is the single number that explains the
  speedup. A theoretical upper bound on the token-latency speedup
  is `1 + α·k · (1 − T_d / T_t)`-shaped (per the Leviathan et al.
  analysis), where α is acceptance rate, `k` is proposed tokens per
  step, and `T_d / T_t` is the draft-to-target latency ratio.
  Report α, `k`, and both forward-pass latencies alongside the
  measured speedup.

## 3. Validation steps

1. Launch both configurations (target-only and target + draft)
   using identical `--max-model-len` and
   `--gpu-memory-utilization`. Anything else leaks into the delta.
2. Verify correctness: with `temperature=0.0`, run the same prompt
   against both servers and diff completions. They must match. If
   they do not, the draft/target pairing is misconfigured (likely a
   tokenizer mismatch).
3. Run `bench.py` against both servers with the same `PROMPT`,
   `CONCURRENCY`, and `MAX_TOKENS`. Capture:
   - mean seconds per output token,
   - throughput (tok/s),
   - acceptance rate (from metrics).
4. Sweep `--num-speculative-tokens` across `{3, 5, 7}` for the
   speculative server. Report the best `k` and the acceptance rate
   at that `k`.
5. Repeat the measurement on a second, deliberately different
   prompt domain (e.g., technical English vs. a short code task).
   Record whether the acceptance rate — and therefore the speedup —
   generalizes.
6. Record configuration: target model, draft model, `k`, GPU,
   dtype, concurrency, prompt token counts.

## 4. Rubric or review checklist

| Check | Weight | Pass criterion |
|---|---|---|
| Correct draft/target pairing | 15 | Shared tokenizer; draft model is substantially smaller than target; justification recorded |
| Launch command is complete and pinned | 10 | vLLM version pinned; draft, target, and `k` all shown |
| Baseline comparison | 15 | Target-only server benchmarked with identical flags except `--speculative-model`/`-k` |
| Reports acceptance rate | 20 | Extracted from vLLM metrics and tied to the speedup number |
| Reports per-token latency | 10 | Seconds-per-token, not seconds-per-request |
| Domain-sensitivity check | 10 | Two prompt domains run; speedup compared across them |
| `k` sweep | 10 | At least three values of `--num-speculative-tokens` tried; chosen value justified |
| Correctness check | 5 | Greedy outputs diff-identical between baseline and speculative |
| Honest interpretation | 5 | Does not claim a general "N× speedup" independent of workload |

## 5. Common mistakes

1. **No acceptance-rate reporting.** Without α, a reported speedup
   is not reproducible or portable. It must be part of the result
   table.
2. **Mismatched tokenizers.** A draft with a different tokenizer
   from the target cannot be used for exact speculative decoding.
   vLLM will refuse (or silently disable, depending on version);
   check the startup logs.
3. **Treating speculative decoding as an approximation.** Correct
   speculative decoding preserves the target's output distribution.
   If greedy outputs differ between baseline and speculative servers,
   something is misconfigured — do not ship it.
4. **Benchmarking at high concurrency.** Speculative decoding's
   wins are largest at low batch where the target is bandwidth
   bound on weight reads. At very high concurrency, continuous
   batching already fills target forward passes and the additional
   draft work can be a net loss.
5. **Choosing `k` without sweeping.** Larger `k` is not strictly
   better. Past the point where the draft reliably agrees with the
   target, extra proposed tokens are pure overhead.
6. **Using temperature or top-p that differ between runs.** Sampling
   configuration changes acceptance rate. Hold it fixed across the
   baseline and speculative runs.
7. **Attributing speedup to the algorithm without accounting for
   draft-model cost.** Draft forward passes consume GPU time and
   KV-cache capacity. On a memory-tight deployment, hosting the
   draft can regress target throughput even when per-request
   latency improves.

## 6. References

- Leviathan, Kalman, Matias — "Fast Inference from Transformers via
  Speculative Decoding" (2023):
  <https://arxiv.org/abs/2211.17192>
- Chen, Borgeaud, Irving, et al. — "Accelerating Large Language
  Model Decoding with Speculative Sampling" (2023):
  <https://arxiv.org/abs/2302.01318>
- vLLM docs — Speculative Decoding:
  <https://docs.vllm.ai/en/latest/features/spec_decode.html>
- vLLM docs — Engine arguments
  (`--speculative-model`, `--num-speculative-tokens`):
  <https://docs.vllm.ai/en/latest/serving/engine_args.html>
- vLLM docs — Production metrics (Prometheus endpoint):
  <https://docs.vllm.ai/en/latest/serving/metrics.html>
- Module-level context:
  `modules/mod-004-transformer-optimization/SOLUTION.md`
  ("Decision 4: Speculative decoding as the 'you pay tokens to
  save tokens' trick"), and the launch pointer in
  `modules/mod-004-transformer-optimization/exercise-04/README.md`.
