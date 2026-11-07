# Exercise 02 — Pipeline-Parallel for 70B+

## 1. Solution overview

Serve a model that is too large for a single host (classically a
70B-class LLM in FP16, which exceeds 8× 80 GB of weight memory
once KV cache is accounted for) by combining **tensor parallelism
within each host** with **pipeline parallelism across hosts** in
vLLM.

The goal of the exercise is not only to get the model to boot
across two hosts; it is to make the candidate confront the
latency cost of pipeline parallelism (PP) during inference and
justify, in writing, when the resulting trade-off is the right
one and when replica-level scaling of a smaller or compressed
model is a better answer.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md) Decision 2 ("Pipeline parallel for
completeness, but flagged as 'rarely right for inference'").

## 2. Worked answer or implementation

### 2.1 Minimal runnable deployment (two-host example)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-2-70b-chat-hf \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 2 \
  --port 8000
```

Semantics of the flags:

- `--tensor-parallel-size 4` — within each host, the model's
  linear projections are sharded across 4 GPUs connected by
  NVLink/NVSwitch. The per-layer all-reduce stays on NVLink.
- `--pipeline-parallel-size 2` — the model's layers are split
  into 2 stages; each stage runs on one host. Stage boundaries
  send activations between hosts.
- Total GPUs required: `TP × PP = 4 × 2 = 8` across two
  4-GPU hosts.

vLLM dispatches the ranks via its distributed launcher (typically
Ray or `torchrun`-style multi-node execution). Both hosts must
agree on the model identity, dtype, and max context.

### 2.2 Host and network prerequisites

- Each host has 4 NVLink/NVSwitch-connected GPUs with enough
  per-GPU HBM to hold its share of weights plus KV cache.
- Inter-host fabric is a low-latency, high-bandwidth network —
  in practice InfiniBand or RoCE — not commodity Ethernet. The
  PP stage boundary moves activations once per micro-batch per
  layer-group; a slow link shows up directly in p50 latency.
- Both hosts can resolve each other by hostname and route
  distributed-training traffic to each other on the port range
  the launcher expects.

### 2.3 Why this shape, not TP=8 across hosts

A naive alternative is `--tensor-parallel-size 8` across two
4-GPU hosts. TP performs an all-reduce **every transformer
block**; crossing hosts for that all-reduce collapses throughput
to the inter-host link's bandwidth times the number of layers.
Keeping TP inside the host (where NVLink/NVSwitch is fast) and
PP across hosts (where the cross-host traffic is only at stage
boundaries, not every layer) is the deliberate shape.

Candidates must be able to state that distinction. If they
cannot, they have not learned the exercise.

### 2.4 Why PP hurts inference latency (what to show, not assert)

PP divides the model into sequential stages. For a single
request, stage 2 is idle while stage 1 computes its share, and
vice versa — the "pipeline bubble". Micro-batching hides some of
the bubble under load but does not eliminate it, and single-user
interactive latency still sees it.

The exercise therefore asks for **two measurements**:

1. **Single-request p50 latency** under PP, compared to the same
   model served purely with TP within a single larger host when
   that option exists, or to a smaller model (e.g. a 13B or
   quantized 70B) served TP-only.
2. **Throughput under concurrency**, where PP recovers much of
   its efficiency because micro-batches overlap across stages.

The expected finding is that PP looks worse on single-request
latency and competitive on sustained throughput. That shape is
the whole point of Decision 2 in the module SOLUTION.md: PP is a
training-first tool that is sometimes the only answer for
inference when a smaller or compressed variant is unacceptable.

## 3. Validation steps

1. **Cluster bring-up**
   - The distributed launcher reports all `TP × PP` ranks joined
     before the server begins loading weights.
   - On each host `nvidia-smi` shows exactly `TP` worker
     processes, each holding roughly `1/(TP × PP)` of the weights
     plus its share of KV cache.
2. **Intra-host topology check**
   - `nvidia-smi topo -m` on each host shows `NV#` links between
     every pair of TP ranks. If any intra-host pair is `SYS` or
     `PIX`, abort and fix the host build before benchmarking.
3. **Inter-host fabric check**
   - Confirm the inter-host NIC is the intended one (IB/RoCE) and
     that the distributed backend is using it, not a management
     Ethernet NIC. Vendor tools (`ibstat`, `rdma link`) or the
     NCCL debug output (`NCCL_DEBUG=INFO`) can confirm.
4. **Functional correctness**
   - Same `curl` call as in exercise-01 §2.2 returns a 200 with
     non-empty text; greedy decoding is deterministic across
     runs.
5. **Latency measurement (single-request)**
   - Record p50 / p95 time-to-first-token and time-per-output-
     token for single-stream decoding. Compare against a
     TP-only baseline of a comparable model (smaller, or
     quantized) on the same hardware.
6. **Throughput measurement (concurrent)**
   - Drive the server with a constant-rate load generator;
     record tokens/second at saturation and the queueing
     behavior as concurrency climbs.
7. **Failure-mode spot checks**
   - Kill one rank on the second host and confirm the server
     fails fast rather than silently degrading. PP across hosts
     makes the blast radius of a bad node larger than TP on one
     host; the operational posture must account for it.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | Shape justified: TP within host, PP across hosts | Written rationale references the per-layer all-reduce cost of TP and the stage-boundary-only cost of PP |
| 2 | TP × PP = total GPUs, divides head count | Server starts with all ranks joined |
| 3 | Intra-host topology verified | `nvidia-smi topo -m` output recorded for each host; all intra-host pairs are `NV#` |
| 4 | Inter-host fabric identified | IB/RoCE confirmed via vendor tool or NCCL debug output; management Ethernet is **not** the data path |
| 5 | Latency penalty measured, not asserted | p50/p95 TTFT and per-token latency recorded; comparison to a TP-only alternative is on the page |
| 6 | Throughput measured under concurrency | Tokens/second at a stated QPS included |
| 7 | Alternative seriously considered | Candidate names a plausible alternative (quantize-then-TP on one host, replica scaling of a smaller model) and states why PP is justified here |
| 8 | Failure handling discussed | At least one line on what happens when one rank or host dies; operational blast radius is acknowledged |

Scoring guidance:

- 7–8 passes: can defend a PP deployment in a production review.
- 5–6 passes: solid; usually missing either the fabric check or
  the explicit alternative-considered step.
- ≤ 4 passes: the candidate has executed the command. Send them
  back to measure latency against a TP-only alternative before
  signing off; this is the decision the exercise exists to
  teach.

## 5. Common mistakes

1. **Using TP across hosts instead of TP-within + PP-across.**
   Every layer's all-reduce crosses the slower link; throughput
   collapses to inter-host bandwidth. See the module SOLUTION.md
   Decision 1 anti-pattern.
2. **Reporting only throughput.** PP hides its cost in
   single-request latency. A benchmark that measures only
   tokens/second under max concurrency makes PP look free when
   it is not; p50/p95 TTFT must be reported alongside.
3. **Picking the wrong NIC.** If the distributed backend falls
   back to a management Ethernet interface, stage-boundary
   activations traverse a link that is orders of magnitude
   slower than the IB/RoCE fabric you provisioned. Confirm
   explicitly, do not assume.
4. **Skipping the "smaller or quantized" alternative.** Many
   workloads that reach for PP-across-hosts could instead run as
   a quantized model TP-only on one host (see mod-005 compression
   exercises). Not considering that is the most common failure
   of this exercise.
5. **Treating PP and replica-scaling as the same lever.** PP
   stretches one model across more GPUs; replica scaling runs
   more copies of the same model. They solve different problems:
   PP is for models that do not fit; replicas are for throughput.
   Candidates who conflate them will pick the wrong tool in
   production.
6. **No plan for a dead rank.** TP within one host dies with the
   host and HPA replaces it. PP across hosts can leave a
   partially alive deployment that silently degrades. The design
   must declare how a dead rank is detected and recovered.

## 6. References

- vLLM project documentation — distributed serving,
  `--tensor-parallel-size`, `--pipeline-parallel-size`,
  and multi-node launcher usage: https://docs.vllm.ai/
- NVIDIA NCCL documentation — environment variables
  (`NCCL_DEBUG`, interface selection) used when confirming the
  inter-host data path: https://docs.nvidia.com/deeplearning/nccl/
- NVIDIA developer documentation — `nvidia-smi topo -m` for
  intra-host topology verification: https://docs.nvidia.com/
- Local module context —
  [`modules/mod-006-distributed-inference/SOLUTION.md`](../SOLUTION.md),
  especially Decision 2 (PP flagged as "rarely right for
  inference") and Decision 1's TP-across-hosts anti-pattern.
- Prior exercise —
  [`exercise-01`](../exercise-01/SOLUTION.md) for the TP-only
  baseline this exercise measures against.
- Compression alternatives —
  [`mod-005-model-compression/SOLUTION.md`](../../mod-005-model-compression/SOLUTION.md)
  for the "quantize and stay single-host" branch of the
  decision.
