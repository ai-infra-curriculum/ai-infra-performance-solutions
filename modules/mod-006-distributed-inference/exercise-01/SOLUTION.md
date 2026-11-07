# Exercise 01 — Tensor-Parallel vLLM Deployment

## 1. Solution overview

Serve a model that exceeds single-GPU memory (for example a 13B
model in FP16, or a 70B-class model in a quantized format) across
multiple GPUs on **one host** using vLLM's built-in tensor
parallelism.

Tensor parallelism (TP) shards every transformer block's linear
projections across GPUs and performs an all-reduce after each
block. On an NVLink-connected host the all-reduce is amortized
over the layer compute; across hosts over commodity networking
it is not. The intended teaching point is therefore not just "run
the command" — it is to internalize that **TP is a single-host
pattern** and to be able to defend that choice.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md) — the parallelism-strategy table
and Decision 1 ("Tensor parallel first — because it's the
production default").

## 2. Worked answer or implementation

### 2.1 Minimal runnable deployment

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-13b-chat-hf \
  --tensor-parallel-size 2 \
  --port 8000
```

- `--tensor-parallel-size` must divide the model's attention-head
  count evenly. For most Llama-family and Mistral-family models
  the number of heads is a multiple of 8, so 1, 2, 4, 8 are safe
  values.
- The model is sharded once at load time; there is no further
  configuration on the HTTP client side.

### 2.2 Validation client

```bash
curl -s http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "meta-llama/Llama-2-13b-chat-hf",
    "prompt": "Explain tensor parallelism in two sentences.",
    "max_tokens": 64,
    "temperature": 0.0
  }'
```

### 2.3 Host topology that makes TP a reasonable choice

- All GPUs are on the **same physical host**.
- Inter-GPU path is NVLink (or NVSwitch on 8-GPU systems), not
  PCIe-only and not cross-host networking.
- You can verify the topology with `nvidia-smi topo -m`; look for
  `NV#` links between each pair of GPUs the TP job will span.

If `nvidia-smi topo -m` shows `PIX`, `PHB`, or `SYS` between the
chosen GPUs, TP will still *run* but the per-layer all-reduce will
dominate step time and the decision should be revisited.

### 2.4 Why tensor-parallel-size = 2 (worked rationale)

A 13B model in FP16 needs roughly 26 GB for weights plus KV cache
and workspace. On a single 24 GB consumer GPU it will not fit at
useful context lengths; on a single 40/80 GB datacenter GPU it
fits but leaves little headroom for KV cache at high concurrency.
Splitting across two NVLink-connected GPUs gives the model room to
grow the KV cache and preserves low per-token latency because the
all-reduce stays on NVLink.

For larger models the TP degree should grow in powers of two up to
the number of GPUs *on one host* — not beyond it. See
exercise-02 for what to do once the model no longer fits on one
host.

## 3. Validation steps

1. **Process health**
   - Server log prints `Loaded model ...` and `Uvicorn running on
     http://0.0.0.0:8000`.
   - `nvidia-smi` shows one worker process per TP rank, each
     holding roughly `1/N` of the total weight footprint plus
     per-rank KV cache.
2. **Functional correctness**
   - The completions call in §2.2 returns a 200 with non-empty
     `choices[0].text`.
   - Running the same prompt at `temperature=0.0` twice returns
     identical text (determinism sanity check).
3. **Topology sanity**
   - `nvidia-smi topo -m` shows `NV#` links between every pair of
     ranks. If any pair shows `SYS`, record it as a finding
     and lower the TP degree until all pairs are NVLink-connected.
4. **Scaling sanity (throughput, not just latency)**
   - Replay a batch of concurrent requests (e.g. with the vLLM
     `benchmarks/benchmark_serving.py` script or any constant-rate
     load generator). Record tokens/second and p50/p95 latency.
   - Compare against a single-GPU run of the same model at a
     smaller max-batch-size; TP should increase sustained
     tokens/second, not just raw peak latency.
5. **Negative control**
   - Attempt `--tensor-parallel-size` equal to the number of GPUs
     spanning two hosts (e.g. 8 across two 4-GPU hosts). Confirm
     the latency degradation predicted by the module SOLUTION.md
     Decision 1 anti-pattern.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | TP degree divides the attention-head count | Server starts; no divisibility error in the log |
| 2 | All ranks on one host | `nvidia-smi topo -m` shows `NV#` between all TP ranks |
| 3 | Chosen TP degree justified | Written rationale ties degree to weights + KV cache headroom, not arbitrary |
| 4 | Latency measured, not asserted | p50/p95 numbers recorded under a repeatable load |
| 5 | Throughput measured under concurrency | Tokens/second at a stated QPS, not just single-request latency |
| 6 | Negative case acknowledged | Candidate states why TP across hosts would be wrong here (NVLink vs IB bandwidth mismatch) |
| 7 | KV cache budgeted | Capacity math includes KV cache at the declared max context and concurrency, not only weight memory |
| 8 | Cleanup | Server is stopped cleanly; no orphan ranks holding GPU memory on exit |

Scoring guidance:

- 7–8 passes: production-ready baseline.
- 5–6 passes: solid; likely missing either the topology check or
  KV-cache budgeting — both are common.
- ≤ 4 passes: the candidate has run the command but not understood
  the decision. Ask them to re-derive the TP degree from the
  model-size + KV-cache-budget argument before signing off.

## 5. Common mistakes

1. **Choosing a TP degree that does not divide head count.** vLLM
   rejects this at load time; reviewers should still catch it
   because the error message is easy to miss in a long log.
2. **Spanning hosts with TP.** The per-layer all-reduce traverses
   whatever the slowest link is. Mixing NVLink with inter-host
   networking collapses to inter-host speed every layer. See
   the module SOLUTION.md Decision 1 anti-pattern.
3. **Sizing on weight memory only.** The KV cache is a first-class
   consumer of GPU memory. At long context and high concurrency
   it can exceed the weight footprint; a TP degree that "just
   fits" the weights OOMs under real load.
4. **Reporting only single-request latency.** TP's throughput win
   shows up under concurrency. A benchmark that fires one prompt
   at a time and reports latency hides the actual production
   behavior.
5. **Mixing GPU types across ranks.** vLLM expects homogeneous
   ranks; mixed SKUs produce either a hard failure or performance
   pinned to the slowest rank.
6. **Assuming linear speedup.** Going from TP=1 to TP=2 is usually
   sub-linear in latency because the all-reduce is not free.
   Candidates who present "2x faster" without measurement are
   usually wrong.

## 6. References

- vLLM project documentation — distributed inference / tensor
  parallelism (`--tensor-parallel-size`): https://docs.vllm.ai/
- NVIDIA developer documentation — `nvidia-smi topo -m` and
  NVLink topology interpretation: https://docs.nvidia.com/
- Local module context —
  [`modules/mod-006-distributed-inference/SOLUTION.md`](../SOLUTION.md),
  especially the parallelism-strategy table and "Decision 1:
  Tensor parallel first".
- Follow-on exercise —
  [`exercise-02`](../exercise-02/SOLUTION.md) for what to do once
  the model exceeds a single host.
