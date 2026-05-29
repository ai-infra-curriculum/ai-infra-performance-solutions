# SOLUTION — Project 03: High-Performance LLM Inference System

> Read this *after* you have walked the learning project's
> `requirements.md`, `architecture.md`, and `STEP_BY_STEP.md`. This
> document is the reviewer's companion. It explains *why* the project
> is shaped the way it is, what passing work actually looks like, and
> where graders should expect to push back.
>
> The full project spec — hard gates, rubric dimensions, deliverable
> manifest, and the exact target SLOs — lives in the paired learning
> repository under
> [`projects/project-03-distributed-inference`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects/project-03-distributed-inference).
> Numbers in this file are pulled from that spec; do not invent new
> targets here. Sections that name a specific gate ID, rubric
> dimension, or numeric threshold that the reviewer should expect
> from the spec are marked `<!-- spec-pin: ... -->` and should be
> reconciled with the learning repo before the reviewer-facing copy
> is treated as final.

## 1. Solution overview

The project is a **single end-to-end production LLM serving system**
that takes a model that does not fit on one GPU (the canonical
reference is a Llama-class 70B model, FP16 weights ≈ 140 GB; cf.
`mod-006-distributed-inference/SOLUTION.md` § "What this module is
really teaching"), distributes it across a small cluster, fronts it
with a routing layer, and operates it under realistic load with a
written SLO.

There is intentionally **no single "answer"**. The project tests
whether the candidate can compose techniques covered separately in
mod-006 (parallelism), mod-007 (production deployment), and mod-008
(advanced topics — CUDA Graphs, NCCL tuning, MIG) into a serving
stack that an SRE would actually deploy, with the trade-offs measured
rather than asserted.

The four design pillars the reviewer is grading against:

1. **The parallelism choice is defended with measurements**, not
   asserted from a vendor blog. TP within a node, replicas across
   nodes is the production default in 2026 (cf. mod-006 SOLUTION.md
   § "Decision 1"), but the submission has to show the measurement
   that says so on the candidate's hardware, not parrot the claim.
2. **The serving SLO is written down before the system is built**.
   The candidate commits to p50 / p95 / p99 first-token-latency
   (TTFT), inter-token-latency (ITL), and end-to-end-request-latency
   targets, plus a throughput floor at a stated input/output length
   distribution. Every later optimization is measured against that
   contract.
3. **Scheduling is continuous, not request-at-a-time**. The
   reference uses a continuous-batching scheduler (the vLLM /
   TGI / TensorRT-LLM-in-flight pattern) with paged attention for
   the KV cache. Submissions that ship "dynamic batching" by
   buffering N requests and launching one fixed-batch forward pass
   miss the entire point of the project.
4. **The operational tax is paid honestly**. Cold-start latency,
   spot interruption resilience, prefix-cache hit rate, and HPA
   signal selection are first-class deliverables — not appendices.

### What a passing submission looks like

A reviewer reading the deliverables should be able to answer, in
under one minute and *from the artifacts only*:

1. Does the system hit its stated **TTFT / ITL / throughput SLO**
   under the declared load profile? — From `reports/slo_report.md`.
   <!-- spec-pin: the exact PR-N gates on TTFT p99, ITL p99, and
   throughput floor live in the learning repo's requirements.md and
   must be quoted verbatim here. -->
2. Is the **parallelism configuration** (`tensor_parallel_size`,
   replica count, intra-node vs cross-node placement) justified
   against at least one alternative on the candidate's hardware? —
   From `reports/parallelism_sweep.md` plus the ADR in
   `docs/adr/0001-parallelism.md`.
3. Was the bench **statistically defensible** (≥ 50 warmup requests,
   ≥ 500 measured requests at the declared concurrency, P50/P95/P99
   reported separately for TTFT and ITL, clocks locked, no thermal
   throttling)? — From the raw JSONL under `reports/raw/` plus the
   `nvidia-smi` clock-state snapshot in the manifest. The
   measurement contract is inherited from project-01 § 1 and
   project-02 § 1; the workload changes, the discipline does not.
4. Is the **prefix-cache hit rate** measured, and does the routing
   layer actually drive it? — From `reports/cache_hit_rate.md`
   showing the round-robin baseline and the prefix-routed
   measurement on the same trace.
5. Does the system **survive a replica loss** without breaching the
   SLO? — From `reports/failover_drill.md` with the chaos-test
   timeline.
6. Will `make verify` reproduce the SLO summary within the declared
   tolerance on the target SKU? — From the `make verify` log in
   the manifest.

If any of those six answers requires reading source code, the
deliverables have failed the profiling-depth and code-quality rubric
dimensions.
<!-- spec-pin: confirm the D-numbering of profiling-depth and
code-quality dimensions in the project-03 rubric.md. -->

### How the project layers onto the module solutions

The project is a *composition + extension* exercise. Each capability
maps to a technique the candidate has already worked at the module
level:

| Capability delivered                                       | Where the underlying technique was taught                                  |
|------------------------------------------------------------|----------------------------------------------------------------------------|
| Tensor-parallel sharding within a node                     | `mod-006-distributed-inference/exercise-01` (tensor parallel)              |
| Pipeline-parallel sharding (and the reasons not to use it) | `mod-006-distributed-inference/exercise-02` (pipeline parallel)            |
| Queue-depth-based HPA (not CPU-based)                      | `mod-006-distributed-inference/exercise-03` (custom HPA metric)            |
| Prefix-aware request routing                               | `mod-006-distributed-inference/exercise-04` (prefix-aware routing)         |
| Cold-start mitigation (pre-warmed pods, readiness gating)  | `mod-006-distributed-inference/exercise-05` (cold-start mitigation)        |
| Framework selection rationale (vLLM vs TRT-LLM vs SGLang)  | `mod-007-production-deployment/exercise-01` (framework selection)          |
| Canary rollout with regression gates                       | `mod-007-production-deployment/exercise-02` (canary deployment)            |
| Spot-instance resilience                                   | `mod-007-production-deployment/exercise-03` (spot resilience)              |
| Multi-tier routing (model-size / tenant tiering)           | `mod-007-production-deployment/exercise-04` (multi-tier routing)           |
| End-to-end deploy under a real SLO                         | `mod-007-production-deployment/exercise-05` (end-to-end deploy)            |
| CUDA Graphs capture on the decode path                     | `mod-008-advanced-topics/exercise-01` (CUDA Graphs)                        |
| Stream overlap for prefill / decode disaggregation         | `mod-008-advanced-topics/exercise-02` (stream overlap)                     |
| NCCL all-reduce tuning across TP ranks                     | `mod-008-advanced-topics/exercise-03` (NCCL tests)                         |
| MIG partition for small-model tenants                      | `mod-008-advanced-topics/exercise-04` (MIG partition)                      |
| Flash-attention, KV-cache, quantization at the engine layer | `mod-004-transformer-optimization/SOLUTION.md`                            |
| Bench-runner contract, manifest discipline, quarantine     | `projects/project-01-model-optimization/SOLUTION.md`                       |

The candidate is not expected to re-derive any of these techniques.
The point of the project is to **commit** to a serving topology,
**measure** it under a written SLO, and **defend** the choice with
load-test data and chaos drills.

## 2. Worked answer / implementation walkthrough

The phase-by-phase build is laid out in the learning repo's
[`STEP_BY_STEP.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-03-distributed-inference/STEP_BY_STEP.md).
That document is canonical for "do this exactly." This section
calls out the **non-obvious design choices** and the **why** behind
each one.

### 2.1 Write the SLO before you write the serving stack

Every gate downstream depends on this. The reference solution
commits to a written SLO **before** the first deployment:

- **TTFT** (time to first token): p50, p95, p99 caps at a stated
  concurrency level and a stated input-length distribution.
- **ITL** (inter-token latency): p50, p95, p99 caps. This is the
  user-visible "is the model fast?" signal.
- **End-to-end request latency**: TTFT + (output_tokens × ITL) at
  the declared output-length distribution.
- **Throughput floor**: tokens / second / replica at the declared
  load, sustained for ≥ 10 minutes without breaching the latency
  caps.
- **Availability**: requests served / requests admitted, after
  excluding requests rejected by admission control (which is a
  feature, not a failure).

The non-obvious choice: the SLO is **per percentile and per
workload class**, not a single average. A serving system that hits
p50 TTFT but blows p99 is the failure mode every load-test misses
if you only watch the mean. Submissions that report only mean
latency lose D2/D3 (latency / throughput) by definition.
<!-- spec-pin: the exact SLO numbers (TTFT p99 cap, ITL p99 cap,
tokens-per-second floor) live in the project-03 requirements.md and
must be quoted verbatim here. -->

### 2.2 Tensor parallel within the node, replicas across — measured, not asserted

Mod-006 SOLUTION.md § "Decision 1" already establishes the
qualitative argument: NVLink between H100s on the same host runs
fast enough that the per-block all-reduce is amortized; once you
cross hosts (NVL switch → InfiniBand), the all-reduce becomes the
bottleneck and TP catastrophically loses.

The project does **not** ask the candidate to re-derive that
conclusion. It asks the candidate to **measure** it on the
candidate's hardware. The reference parallelism sweep is:

| Config                                  | What it tests                                                    |
|-----------------------------------------|------------------------------------------------------------------|
| `TP=1` (replicate-only, if model fits)  | Lower bound on overhead; reference for "no parallelism"          |
| `TP=2` within node                      | Two-way TP, low all-reduce traffic                               |
| `TP=4` within node                      | Production default for the 70B reference workload                |
| `TP=8` within node (single 8-GPU host)  | Test of the within-node ceiling                                  |
| `TP=8` across two 4-GPU hosts           | The anti-pattern — quantify how much it costs                    |
| `PP=2 × TP=4`                           | Hybrid; mostly for completeness, with bubble-cost commentary     |
| `TP=4` × `N` replicas (HPA target)      | The production deployment                                        |

The deliverable is `reports/parallelism_sweep.md` with one row per
config and a per-row TTFT/ITL/throughput tuple. The candidate's
chosen production config is defended against the sweep, not against
a generic vendor recommendation.

Submissions that ship `TP=8` across two 4-GPU hosts as the
production config without running the within-node alternative are
caught by mod-006 SOLUTION.md "Anti-pattern" 1.
<!-- spec-pin: confirm the exact set of configs the project-03 spec
requires; the canonical sweep above mirrors the mod-006 framing. -->

### 2.3 Continuous batching is not "dynamic batching" — and the difference is observable

Every reviewer should grep the scheduler for the word `pad`. A
scheduler that pads requests to a fixed batch shape so that all
requests in a batch finish together is **request-level batching**;
the longest request in the batch determines when any request
finishes, and short requests pay the full long-request latency.

A **continuous-batching** scheduler (the vLLM / TGI / TRT-LLM
in-flight pattern) admits new requests to the running batch at
every decode step, and evicts finished requests at the same
boundary. The KV cache is paged so that adding/removing a request
does not touch the others.

The reference solution uses the engine's continuous-batching
scheduler unmodified. The candidate is not expected to write a new
scheduler; the candidate **is** expected to:

1. Verify the engine's scheduler is actually doing continuous
   batching by inspecting an Nsight Systems trace and showing
   that requests admit / complete at decode-step granularity.
2. Tune the engine's `max_num_seqs` (concurrent sequences) and
   `max_num_batched_tokens` (per-step token budget) to the
   declared SLO.
3. Document the failure mode if those knobs are wrong (cf. § 5).

Submissions that ship the engine's defaults without tuning, and
report numbers anyway, lose engineering-judgment points.
<!-- spec-pin: confirm the engine choice (vLLM is the mod-006
default; the project may permit TRT-LLM as an alternative with
documented selection criteria). -->

### 2.4 Paged attention is the KV-cache memory budget, not just an optimization

For autoregressive serving, the KV cache is the dominant memory
consumer past the model weights. At realistic load (high
concurrency, long context), the KV cache can exceed the model size
by 10×; mod-006 SOLUTION.md "Common mistakes" 3 calls this out as
the most-missed capacity item.

The reference solution does **not** treat paged attention as an
optimization to bolt on; it treats the KV cache as a **first-class
memory budget** in the capacity plan:

- The candidate computes `max_concurrent_sequences = (GPU_HBM -
  weights - activation_workspace) / (kv_bytes_per_token *
  max_sequence_length)` **before** turning on traffic.
- Admission control rejects (or queues) requests that would push
  the cache past the budget. Silently OOMing the worker and
  letting Kubernetes restart it is a failure mode, not a recovery.
- The reference paged-attention block size is the engine default
  (typically 16 tokens per page); deviating from it requires an
  ADR with a measured speedup, not "I thought it might be faster."

A submission that ignores the KV cache in its capacity math and
reports throughput-only numbers is caught at the failover drill —
the first OOM kills the throughput claim. Flag and re-run.

### 2.5 Prefix-aware routing is the highest-leverage routing change for chat workloads

Exercise 04 in mod-006 already establishes the qualitative win: a
consistent hash over the first K tokens of the prompt drives the
prefix-cache hit rate from ~5% (round-robin) to ~95% on chat
workloads with a shared system prompt, with a 2–5× throughput
improvement at long shared prompts (mod-006 SOLUTION.md § "Decision 4").

The project requires the candidate to:

1. **Measure** the round-robin baseline cache hit rate on a
   replay-able trace (the reference uses a synthesized chat
   workload with a shared system prompt; see § 3).
2. Implement the prefix hash router (the reference uses an
   in-process Envoy filter or an in-cluster sidecar; both are
   acceptable).
3. **Measure** the prefix-routed cache hit rate on the same trace,
   and show the latency CDF before and after.
4. Handle the **replica-loss case**: when the replica a prefix
   group is pinned to dies, the next request in that group must
   re-prefill on a new replica; the candidate's router must surface
   this as a metric (`prefix_cache_miss_due_to_failover_total`),
   not silently mask it.

Submissions that ship prefix routing without the failover metric
are caught by the chaos drill (§ 3 step 6); the cache hit rate
drops without explanation, and the SLO breach is unattributable.

### 2.6 The HPA scales on queue depth, not on GPU utilization

This decision is inherited verbatim from mod-006 SOLUTION.md
"Decision 3" and "Anti-pattern" 2. The reasoning is short and
worth re-stating because most submissions get it wrong on the
first attempt:

- **GPU utilization** is a *trailing* signal. By the time
  `nvidia_smi_utilization_gpu` reads 80%, the request queue at the
  ingress is already backed up, the user is already seeing
  elevated TTFT, and the 30-60 seconds it takes to spin up a new
  replica + warm the KV pages is already a breach.
- **Inference queue depth** is a *leading* signal. The HPA scales
  the moment requests start queueing — before the user notices.

The reference solution exposes `inference_queue_depth` from the
gateway via Prometheus, wires it through the HPA external metrics
adapter, and keeps the target at the engine's measured "comfortable
saturation" point (typically queue depth ≤ 5 with the surge buffer
in § 2.8). The candidate's HPA YAML is reviewed against the
mod-006 ex-03 reference.

A submission scaling on `nvidia_smi_utilization_gpu` should be
flagged with the mod-006 SOLUTION.md "Common mistakes" 2 quote and
re-implemented with the queue-depth metric.

### 2.7 The bench is replay, not synthetic — and it includes the prefill/decode mix

The bench-runner contract is inherited from project-01 § 2.1 and
project-02 § 2.2 (CUDA events for engine-internal kernels; for
end-to-end request latency the contract is wall-clock at the
client with the same warmup / measurement discipline). The
non-obvious choice specific to this project:

- The load profile is a **mix of prefill-heavy and decode-heavy
  requests**, not a single shape. A serving system that wins on
  one shape and loses on the other is hiding the win. The
  reference profile is documented in `bench/profiles/<name>.yaml`
  with input-length and output-length distributions, and the
  report includes per-class metrics.
- The bench replays from a **fixed trace file** so the run is
  reproducible. Synthetic random prompts produce different cache
  hit rates on every run and the prefix-routing measurement
  becomes useless.
- The bench measures **TTFT and ITL separately** because they have
  different bottlenecks: TTFT is dominated by prefill (compute-
  bound) and ITL is dominated by decode (memory-bound). Reporting
  a single "latency" number conflates them.

Submissions that report a single mean latency on a single shape
lose D2/D3 points and have to re-run.

### 2.8 The surge buffer is not optional

Mod-006 SOLUTION.md "Decision 5" calls this out: cold-starting an
LLM replica takes 30-90 seconds to load the weights from S3 plus
30-60 seconds to JIT/CUDA-graph warm. During that window, traffic
routed to the new replica gets terrible latency.

The reference HPA configuration keeps **N+1** replicas, where the
"+1" is the surge buffer that absorbs traffic while the next
replica warms. The candidate's HPA YAML:

- `minReplicas` = the SLO floor at the declared traffic.
- `maxReplicas` = the SLO ceiling plus the surge buffer.
- A `scaleDown.stabilizationWindowSeconds` long enough that
  short-lived spikes don't churn replicas (the reference is 300s;
  shorter values trigger thrash on bursty workloads).

Submissions that omit the surge buffer ("HPA will scale up before
the queue builds") are caught by the cold-start drill (§ 3 step
7); the queue does build, and the breach is on the report.

### 2.9 Cold start is gated on readiness, not liveness

This is a Kubernetes-shaped trap that is easy to miss. The
distinction:

- **Liveness probe**: "is the container alive at all?" Kubernetes
  restarts the pod on failure. Cheap; should pass within seconds
  of container start.
- **Readiness probe**: "is the container ready to receive
  traffic?" Kubernetes routes traffic only after success. The
  pod is "running" but is held out of the Service endpoints until
  it passes.

The reference solution gates readiness on a **real inference
request** that exercises the model end-to-end: the probe sends a
short prompt, asserts a non-empty completion within a timeout, and
returns 200 only after success. This catches partial loading
(weights downloaded but compilation failed) that a TCP-port check
would miss.

A submission that gates readiness on a simple HTTP 200 from `/health`
ships a pod that takes traffic before it's warm; mod-006 SOLUTION.md
"Common mistakes" 5 is the canonical statement of this anti-pattern.

### 2.10 Spot resilience is a graceful drain, not a checkpoint restore

Mod-007 ex-03 covers spot-instance resilience. For inference there
is no training-loop checkpoint to restore from — the candidate's
job is to **drain in-flight requests gracefully** before the spot
preemption window closes:

1. Subscribe to the cloud's spot-preemption notice (the reference
   uses the AWS `metadata/spot/instance-action` endpoint or the
   equivalent on the candidate's cloud).
2. On notice, the pod's `preStop` hook sets the readiness probe
   to fail (so Kubernetes stops routing new traffic), then waits
   for in-flight requests to finish (with a hard cap at ~90% of
   the preemption window).
3. The HPA's surge buffer absorbs the lost capacity until the
   replacement replica warms.

The reference does **not** attempt to migrate in-flight requests
to another replica; the engineering effort to checkpoint the KV
cache and restore it elsewhere is not justified by the spot-cost
saving. Submissions that propose a "KV-cache migration" as the
spot story should be pushed back on with this reasoning unless the
candidate has measured numbers showing the migration pays.

### 2.11 Canary deployment gates on production-relevant metrics, not on test-suite green

Mod-007 ex-02 covers canary rollout. The reference canary gate is
**not** "the CI tests pass" (a precondition, not a gate) but the
following observed on the canary replica's first N minutes of
production traffic:

- TTFT p99 within X% of the baseline replicas.
- ITL p99 within X% of the baseline replicas.
- Error rate within Y bp of the baseline replicas.
- KV-cache OOM count = 0.
- Generation length distribution within Z% of baseline (catches
  silent silver-bullet bugs where the canary returns empty / short
  completions and looks "fast").

A submission that promotes a canary after the test suite passes,
without watching the production-traffic deltas, is fragile to the
class of bugs that only manifest under load. Flag it in review.
<!-- spec-pin: confirm whether the project-03 spec requires the
exact X/Y/Z thresholds above; the qualitative argument stands
regardless. -->

### 2.12 CUDA Graphs capture on the decode path is the highest-ROI advanced optimization

Mod-008 ex-01 covers CUDA Graphs. For LLM decode (per-step,
bs=1-ish, latency-critical), the per-step launch overhead is a
real fraction of the step time on H100. A CUDA Graph that captures
the decode step once and replays it for subsequent tokens removes
the per-step launch overhead and tightens ITL p99 noticeably.

The reference solution enables CUDA Graphs on the decode path
(the engine flag varies; vLLM has `enforce_eager=False`, TRT-LLM
exposes it through the build config). The deliverable is a
**cold/warm latency comparison**: the first decode step in a new
graph is slow (graph capture); subsequent steps are fast. The
report shows both numbers and explains the ramp-up cost.

Submissions claiming a CUDA Graphs win without the cold/warm
comparison are caught by the project-01 anti-pattern ("claiming a
stretch result without a measurement").

### 2.13 Quarantine, don't silently drop — inherited from project-01 and project-02

When a configuration misses an SLO gate (e.g., a tested parallelism
sweep configuration fails the TTFT cap), the reference pipeline
writes the partial artifacts to `runs/quarantine/<config>/` and
records `quarantined: true` in the manifest. Exit code is `2`
(distinct from `1` = setup error). The build does **not** mark
itself green. The fall-through is that the report includes the
quarantined config with its failure mode, so the reviewer can see
what was tried and why it failed.

Submissions that silently drop a failed sweep config — or worse,
ship a config whose SLO test was disabled "temporarily" — are
caught by the same anti-pattern the project-01 and project-02
rubrics call out. Flag and fail.

## 3. Validation steps

The reviewer runs, in order:

```bash
# 0. Inside the candidate's Dockerfile, on the target hardware:
docker build -t llm-serve:review .
docker run --gpus all --rm -v "$PWD":/work llm-serve:review nvidia-smi
docker run --gpus all --rm llm-serve:review nvcc --version
```

Confirm: driver and CUDA versions match the manifest's `hardware:`
and `cuda:` fields, GPU compute capability matches the engine's
required arch (e.g., SM 9.0 for H100), NVLink topology matches the
`tensor_parallel_size` claim (`nvidia-smi topo -m` should show NV4+
between TP-grouped GPUs).

```bash
# 1. Bring up the cluster (single-node dev mode).
make cluster-up
```

Expected: the helm chart / `kubectl apply` lays down the engine
StatefulSet, the routing layer, the Prometheus / queue-depth
metrics adapter, and the HPA. `kubectl get pods` shows all pods
`Ready` within the documented cold-start budget.

```bash
# 2. Smoke-test a single request end-to-end.
make smoke
```

Expected: a single request returns a non-empty completion within
the TTFT cap. This is the fastest "is anything wired up?" check;
if it fails, every later test is uninterpretable.

```bash
# 3. Run the parallelism sweep.
make bench-parallelism-sweep
```

Expected: `reports/parallelism_sweep.md` updated with one row per
config from § 2.2. The chosen production config matches the
candidate's ADR in `docs/adr/0001-parallelism.md`. Every row is
sourced from a JSONL file under `reports/raw/`.

```bash
# 4. Run the SLO load test at the declared concurrency.
make bench-slo
```

Expected: `reports/slo_report.md` shows TTFT p50/p95/p99 and ITL
p50/p95/p99 per workload class, sustained over the declared
duration, with the throughput floor met. Numbers within 5% of the
manifest values on the target SKU.

```bash
# 5. Measure prefix-cache hit rate, round-robin vs prefix-routed.
make bench-prefix-routing
```

Expected: `reports/cache_hit_rate.md` shows the round-robin
baseline (≪ prefix-routed) and the prefix-routed measurement on
the same replay trace, plus the latency CDF before / after. The
delta matches the candidate's "prefix routing wins X%" claim
within 5%.

```bash
# 6. Chaos drill — kill a replica mid-load.
make chaos-replica-loss
```

Expected: `reports/failover_drill.md` shows the timeline (kill at
T, queue depth peaks at T+N, replacement ready at T+M, SLO
restored at T+K) and that no SLO percentile breaches its cap for
longer than the documented `failover_recovery_seconds` budget.

```bash
# 7. Cold-start drill — start a new replica from zero, measure
# until first-served-request.
make chaos-cold-start
```

Expected: `reports/cold_start_drill.md` shows the per-phase
breakdown (image pull → weights load → engine warm → readiness
gate → first served request) summing to ≤ the cold-start budget
from the spec. The readiness probe gates traffic correctly (no
requests served before the warm-up health check passes).

```bash
# 8. Spot-preemption drill — simulate a preemption notice.
make chaos-spot-preempt
```

Expected: `reports/spot_drill.md` shows the drain timeline: notice
received at T, readiness fails at T+small, in-flight requests
complete by T+drain, hard-kill avoided. No in-flight request
returns a 5xx during the drain window.

```bash
# 9. Spot-check the manifest signature.
python -m cli.report --verify-manifest
```

Expected: every artifact's recorded sha256 matches the on-disk
file sha256. Exit non-zero on any mismatch. (Inherited from
project-01 and project-02.)

```bash
# 10. Reproduce the SLO under `make verify`.
make verify
```

Expected: every SLO number within the declared tolerance (5% on
the target SKU; 15% cross-SKU per the project-01 convention) of
the value recorded in the manifest.

```bash
# 11. Spot-check the engine is doing continuous batching, not
# fixed-batch padding (§ 2.3).
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
    --duration=15 -o profiles/decode_trace.nsys-rep \
    python -m bench.profile_one --profile decode --duration 15
nsys stats --report nvtx_sum profiles/decode_trace.nsys-rep
```

Expected: NVTX ranges show requests admitting and completing at
decode-step granularity, not at fixed batch boundaries. A trace
that shows large `pad` ranges or fixed-shape batches indicates
the engine is in dynamic-batching mode, not continuous-batching;
flag and re-run with the correct config.
<!-- spec-pin: confirm the project-03 spec's exact engine + flag
recommendations for continuous batching. -->

## 4. Rubric / review checklist

The full rubric is the learning repo's
[`rubric.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-03-distributed-inference/rubric.md).
This section is the **reviewer's quick read** mapped to artifacts.

### 4.1 Hard gates

<!-- spec-pin: the project-03 hard-gate IDs (PR-1..PR-N) and their
numeric thresholds (TTFT p99 caps, ITL p99 caps, throughput
floors, cold-start budgets, failover recovery budgets) are defined
in the learning repo's requirements.md and must be quoted verbatim
here. The qualitative gates below stand on their own as a reviewer
checklist but should be reconciled with the spec before this
section ships. -->

The reviewer-facing gates that apply regardless of the spec's exact
numeric thresholds:

| Gate                          | Source artifact                            | Pass criterion                                                       |
|-------------------------------|--------------------------------------------|----------------------------------------------------------------------|
| Written SLO present           | `docs/slo.md`                              | TTFT, ITL, throughput, availability targets named with percentiles    |
| SLO met under load            | `reports/slo_report.md`                    | All percentile / throughput targets in `docs/slo.md` met              |
| Parallelism sweep ran         | `reports/parallelism_sweep.md`             | At least the configs in § 2.2 ran; chosen config defended in ADR     |
| Continuous batching verified  | `profiles/decode_trace.nsys-rep` + commentary | NVTX trace shows step-granular admit / complete, not fixed shape   |
| Prefix-cache routing measured | `reports/cache_hit_rate.md`                | Both round-robin baseline and prefix-routed measurement on same trace |
| Failover drill passed         | `reports/failover_drill.md`                | Replica kill recovers within `failover_recovery_seconds` budget       |
| Cold-start drill passed       | `reports/cold_start_drill.md`              | New replica reaches `Ready` within cold-start budget                  |
| Spot drill passed             | `reports/spot_drill.md`                    | Graceful drain; no 5xx during drain window                            |
| Bench discipline              | `reports/raw/*.jsonl`                      | std-dev / P50 ≤ 5% on every measured concurrency; clocks locked       |
| Manifest integrity            | `compression_manifest.yaml` (or equivalent) | Every artifact sha256 verifies                                       |
| Reproduces under `make verify`| `make verify` log                          | SLO numbers within 5% of manifest values                              |

### 4.2 Rubric dimensions

<!-- spec-pin: the D-numbering (D1..D8) and the level-3 / level-5
bars for project-03 must come from rubric.md. The qualitative read
below tells a reviewer what to look at for each likely dimension
and what reference-quality looks like; the column for the level
threshold is left empty until the spec is consulted. -->

For each likely dimension below: artifact to read, what the
reviewer is grading on, and what reference-quality looks like.

| Dimension                  | Read this                                                       | Pass-tier read                                                                    | Reference-tier read                                                                                       |
|----------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| Correctness (output)       | `tests/test_correctness.py`, golden completions                 | Engine-vs-reference completions match within tolerance on a held-out trace        | Stress test with edge cases (very long context, empty prompt, multi-lingual) and an explicit known-issues log |
| Latency (TTFT)             | `reports/slo_report.md` TTFT columns                            | p50 / p95 / p99 caps met at declared concurrency                                  | Cold/warm split, CUDA-Graphs win quantified, breakdown by prefill vs queueing                              |
| Latency (ITL)              | `reports/slo_report.md` ITL columns                             | p50 / p95 / p99 caps met                                                          | Decode-path Nsight trace with kernel-level attribution; CUDA Graphs cold/warm reported                     |
| Throughput                 | `reports/slo_report.md` throughput row                          | Floor met, sustained ≥ 10 min without breaching latency caps                      | Throughput-vs-concurrency knee curve shown; admission-control point chosen on the curve, not vibes        |
| Scalability                | `reports/parallelism_sweep.md`, HPA logs                        | HPA scales up under load; queue-depth metric drives it                            | HPA replays a real traffic shape; surge buffer sized from the cold-start budget; min/max defended in ADR  |
| Resilience                 | `reports/failover_drill.md`, `reports/spot_drill.md`            | Replica loss and spot preemption both recover within budget                       | Region-failover or zonal-failure drill included; degraded-mode SLO documented                              |
| Engineering judgment       | `docs/adr/*.md`                                                 | At least 1 ADR each for parallelism, engine, routing, HPA signal                  | ADRs include alternatives considered and the measurement that ruled them out                               |
| Code quality               | `src/`, `mypy --strict` + `ruff` + `black` logs, Helm charts    | Lints clean; YAML follows the project style guide; helm chart renders cleanly     | Reviewer can stand up a second tenant in < 1 hr by copying the chart values                                |

### 4.3 Anti-patterns (auto-deduct)

These come directly from the learning repo's `rubric.md` and from
the project's "common mistakes" catalog. A reviewer should grep for
them explicitly.

- **TP across hosts with InfiniBand as the production config**
  without a measured comparison against TP within a host: caught
  by mod-006 SOLUTION.md "Common mistakes" 1. Major D5 / D7
  deduction.
- **HPA scaling on `nvidia_smi_utilization_gpu`** (grep
  `kustomize/` or `helm/` for `nvidia_smi`): late-signal
  anti-pattern from mod-006 SOLUTION.md "Common mistakes" 2.
- **`torch.cat` (or equivalent) in the KV-cache update path**:
  O(N²) decoding cost; inherited from project-02 § 5.6.
- **`time.perf_counter` for engine-internal kernel timing**
  (grep `src/bench/` for `perf_counter` where a CUDA event is
  warranted): inherited from project-01 § 4.3 and project-02 § 4.3.
- **Warmup < 50 requests in the SLO test**: inherited from
  project-01 / project-02.
- **Mean-only latency reporting** (no per-percentile breakdown):
  hides the p99 failure mode the project is built to catch.
- **Synthetic random prompts** in the prefix-routing measurement:
  the cache hit rate is not reproducible; the bench can't
  attribute the win.
- **No `prefix_cache_miss_due_to_failover_total` metric** after a
  chaos drill: cache hit rate drops without explanation.
- **Readiness gated on `/health` 200 only** (grep for
  `livenessProbe`/`readinessProbe` and check both): pods take
  traffic before warm-up; mod-006 SOLUTION.md "Common mistakes" 5.
- **No surge buffer in the HPA config** (`maxReplicas == minReplicas
  + scale-up-step`): cold-start budget eats the SLO during spikes.
- **KV-cache memory ignored in capacity math** (no
  `max_concurrent_sequences` calc in the ADR): mod-006 SOLUTION.md
  "Common mistakes" 3.
- **"Dynamic batching" by fixed-shape padding** (grep scheduler for
  `pad_to_max_length`): the project is for continuous batching;
  cf. § 2.3.
- **`--use_fast_math` style flags shipped without a numerics
  audit**: inherited from project-02 § 4.3.
- **No quarantine flow; failed configs silently dropped**:
  inherited from project-01 / project-02.
- **Claiming a stretch result without a measurement**: inherited
  from project-01 / project-02.

### 4.4 Stretch / distinction bonuses

These are scored but don't count toward Pass:

- **Disaggregated prefill / decode** (separate replica pools, the
  Mooncake / DistServe / Splitwise pattern referenced in mod-006
  SOLUTION.md "When to go beyond"): cite the measured throughput
  gain and the additional operational cost.
- **FP8 KV cache and / or FP8 attention on H100** (cf.
  mod-008 ex-05): tensor-core FP8 with stated tolerance and
  measured ITL gain.
- **Speculative decoding** with a small draft model: target-vs-draft
  acceptance rate measured, ITL gain quantified, the failure mode
  ("speculation budget overrun") handled by admission control.
- **CUDA Graphs over the decode path** (cf. mod-008 ex-01) with
  the cold/warm latency comparison reported (§ 2.12).
- **Multi-tenant routing with MIG-partitioned small-model
  replicas** (cf. mod-008 ex-04 + mod-007 ex-04): isolation
  guarantees and per-tenant SLO defended.
- **Region-failover drill**: a chaos test that takes out a whole
  AZ and shows the system shifting traffic to a warm secondary.
- **NCCL all-reduce tuned** (cf. mod-008 ex-03) with the
  `nccl-tests` numbers showing the all-reduce moved from
  bottleneck to non-bottleneck on the TP path.

<!-- spec-pin: confirm the exact B-numbering of the stretch
bonuses in the project-03 rubric.md. -->

## 5. Common mistakes

These are the failure modes the reference graders see repeatedly.
They're grouped by phase so a reviewer can locate the cause
quickly.

### 5.1 Bring-up

- **`tensor_parallel_size > number-of-NVLink-connected GPUs on the
  node`.** The all-reduce silently falls back to PCIe (or worse,
  cross-host) and TP catastrophically loses. Verify with
  `nvidia-smi topo -m` before claiming a TP-N number.
- **NCCL not picking up the high-bandwidth interconnect.** Symptom:
  TP scales sub-linearly even within a node. Cause: NCCL chose a
  suboptimal protocol / algorithm; force-set `NCCL_ALGO`,
  `NCCL_PROTO`, `NCCL_P2P_LEVEL` per the mod-008 ex-03 tuning
  notes and re-measure.
- **GPU clocks unlocked.** Inherited from project-01 § 5.1;
  same fix (`nvidia-smi --lock-gpu-clocks`) and same anti-pattern
  if missing.

### 5.2 Engine configuration

- **`max_num_batched_tokens` too low.** The engine starves;
  throughput floor missed even though latency looks fine. The
  knob caps the per-step token budget across all concurrent
  sequences; size it from the GPU's compute roof.
- **`max_num_seqs` too high.** The KV cache OOMs at runtime; the
  worker restarts; the SLO breach is on the report. Size from
  the KV-budget math (§ 2.4), not from "let's see what fits".
- **`enforce_eager=True` (or equivalent CUDA-Graphs-off flag)
  shipped to production.** CUDA Graphs were disabled to make
  development easier and never re-enabled. ITL p99 is 10-20%
  worse than it needs to be. Catch by grepping the engine config.
- **Quantization claimed but the engine actually runs FP16.**
  Symptom: model size on disk shrinks but latency doesn't. Verify
  with `nvidia-smi dmon -s u` showing the int8/fp8 tensor-core
  utilization, not just the FP16 utilization.

### 5.3 Routing

- **Round-robin routing shipped with prefix caching enabled.**
  Inherited from mod-006 SOLUTION.md "Common mistakes" 4: you
  pay the memory cost without getting the hit rate. Catch with
  the cache-hit-rate report.
- **Prefix hash key too long.** Hashing the entire prompt makes
  every request unique; cache hit rate stays near 0. The
  reference key is the first K tokens (e.g., K=64) which captures
  the shared system prompt without hashing the user-specific tail.
- **No fallback when the pinned replica is overloaded.** The
  router stubbornly routes to a queue-saturated replica because
  the prefix hashes there. The reference router has a
  load-overflow fallback: if the pinned replica's queue depth
  exceeds a cap, the next-best replica gets the request, and
  the cache-miss is metered.
- **Prefix routing in front of replicas with non-uniform KV-cache
  capacity** (MIG-mixed). The hash assumes equal-sized caches; the
  small replicas evict more often and the hit rate sags.

### 5.4 Scheduling

- **Admission control absent.** The system accepts requests
  unboundedly and the queue grows until it OOMs. The SLO has no
  meaning past the saturation point. Reference: reject (or 429)
  past a documented cap.
- **`max_model_len` unset or too high.** A single bad request
  consumes the KV cache and starves everything else. Cap
  `max_model_len` at the workload's measured p99 prompt length
  plus headroom, not "infinity because the model supports it."
- **Fairness-blind scheduling.** A few long requests monopolize
  the per-step token budget; short requests pay for the long
  ones. The reference engine's continuous batcher handles this
  by accounting per-request tokens; submissions that bypass it
  (e.g., by buffering and re-launching) lose the fairness.

### 5.5 HPA / autoscaling

- **Scaling on `nvidia_smi_utilization_gpu`.** Late signal;
  inherited from mod-006 SOLUTION.md "Common mistakes" 2.
- **HPA stabilization window too short.** Replica churn under
  bursty traffic; cold-start cost eats the gains. The reference
  stabilization-window is 300s on scale-down.
- **No `minReplicas`** at the floor of the daily traffic curve.
  At 3 a.m. the HPA scales to 1 (or 0); the morning ramp blows
  TTFT p99 for 30+ minutes while the system warms.
- **HPA target value picked from a single load test.** A target
  set against a steady-state workload underprovisions at burst
  workloads. Set against the worst-case envelope, not the
  best-case observation.

### 5.6 Cold start / spot

- **Cold-start gated on liveness, not readiness.** Inherited from
  mod-006 SOLUTION.md "Common mistakes" 5 and § 2.9.
- **Model weights downloaded from S3 on every cold start.** The
  pull dominates the cold-start budget. The reference bakes
  weights into a read-only PVC (mod-006 ex-05) or pins them in a
  side-car init container.
- **CUDA Graphs not re-captured after a pod restart.** The first
  N decode steps after restart are slow because the graph has to
  re-record. The reference includes a warm-up phase in the
  readiness gate (§ 2.9) that captures the graph before the
  pod becomes Ready.
- **Spot drain hard-kills in-flight requests.** The `preStop`
  hook is missing or its timeout is too short. Users see 5xx;
  the SLO drops. Reference: drain to ≤ 90% of the preemption
  window, then fail readiness, then hard-kill only the residue.

### 5.7 Observability / SLO

- **Mean latency reported, not percentiles.** Hides p99 the
  project is built to catch; § 2.1.
- **No `inference_queue_depth` metric exposed.** The HPA can't
  scale on it; the dashboard can't show it; oncall flies blind.
- **No `prefix_cache_hit_rate` metric.** The routing optimization
  is unmeasurable in production; you can't tune the hash key K
  or the load-overflow fallback (§ 5.3).
- **`tokens_per_second` reported as a single number** averaged
  across requests. A bursty workload averages to a flattering
  number; the spike that caused the breach is invisible.
  Report per-percentile or per-second bucket.

### 5.8 Bench / report

- **No clock-state record in the report.** Inherited from
  project-01 § 5.8 and project-02 § 5.8.
- **Synthetic random prompts in the cache-hit measurement.** §
  2.5 and § 5.3.
- **Speedup-vs-baseline-without-the-baseline.** The candidate
  reports "system is fast" without the round-robin / no-CUDA-Graphs
  / no-prefix-routing baseline; the attribution is unverifiable.
- **`make verify` skipped on the wrong SKU and the run claims
  green.** The verify window is wrong; cross-SKU runs need 15%
  tolerance per the project-01 convention, not 5%.

### 5.9 Cross-cutting

- **Claiming a stretch result without a measurement.** "Could use
  speculative decoding" is not a bonus; speculative decoding
  shipped with measured acceptance rate and ITL delta is.
  Inherited from project-01 and project-02.
- **Hardware-specificity unstated.** A topology tuned for the
  H100 SXM 8x baseboard that the manifest claims runs on every
  cluster is a documentation lie. The candidate's
  `docs/hardware.md` must enumerate tested SKUs and their NVLink
  / InfiniBand topology. Inherited from project-02 § 5.9.
- **System claimed to be "production ready" without an oncall
  runbook.** The reference `docs/runbook.md` lists the three
  most likely incident types (queue runaway, KV-cache OOM, a TP
  rank crash), their dashboards, and their first-response steps.

## 6. References

### Project artifacts (paired learning repo)

- [`projects/project-03-distributed-inference/README.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects/project-03-distributed-inference) — high-level overview, learning outcomes, success criteria.
- `requirements.md`, `architecture.md`, `STEP_BY_STEP.md`, `rubric.md`, `deliverables/README.md` under the same directory — the canonical contract.
  <!-- spec-pin: when these files are confirmed in the learning
  repo, replace this collapsed reference with the individual links
  used in project-01 § 6. -->

### Related module solutions (this repo)

- [`modules/mod-001-gpu-fundamentals/SOLUTION.md`](../../modules/mod-001-gpu-fundamentals/SOLUTION.md) — roofline and the vocabulary every SLO claim depends on.
- [`modules/mod-003-performance-profiling/SOLUTION.md`](../../modules/mod-003-performance-profiling/SOLUTION.md) — Nsight Systems / Nsight Compute / PyTorch Profiler discipline; the bench-runner contract for engine-internal kernels is inherited.
- [`modules/mod-004-transformer-optimization/SOLUTION.md`](../../modules/mod-004-transformer-optimization/SOLUTION.md) — FlashAttention, KV-cache, quantization; the per-replica engine internals this project's serving stack composes.
- [`modules/mod-005-model-compression/SOLUTION.md`](../../modules/mod-005-model-compression/SOLUTION.md) — quantization / pruning / distillation rationale that determines how much model fits per replica.
- [`modules/mod-006-distributed-inference/SOLUTION.md`](../../modules/mod-006-distributed-inference/SOLUTION.md) — TP / PP / EP parallelism choices, queue-depth HPA, prefix routing, cold-start mitigation. **The single most important upstream module for this project.**
- [`modules/mod-007-production-deployment/SOLUTION.md`](../../modules/mod-007-production-deployment/SOLUTION.md) — framework selection, canary, spot resilience, multi-tier routing.
- [`modules/mod-008-advanced-topics/SOLUTION.md`](../../modules/mod-008-advanced-topics/SOLUTION.md) — CUDA Graphs, stream overlap, NCCL tuning, MIG, FP8 — sources for the stretch bonuses.
- [`projects/project-01-model-optimization/SOLUTION.md`](../project-01-model-optimization/SOLUTION.md) — the bench-runner contract, manifest discipline, and quarantine flow are inherited verbatim; do not re-derive.
- [`projects/project-02-gpu-optimization/SOLUTION.md`](../project-02-gpu-optimization/SOLUTION.md) — the custom-kernel operators that the per-replica engine internals build from; the "when is hand-written CUDA worth it?" framing from project-02 § 2.5 carries through here.
- [`SOLUTION_OVERVIEW.md`](../../SOLUTION_OVERVIEW.md) — track-wide design philosophy ("measure before you optimize", "verify model quality after every change", "hardware specificity is a feature").

### Official standards and primary sources

These are the authoritative documents the reference solution
defers to. Citations are by document family — pin the specific
version that matches the repo's `requirements.md` dependency table
when reviewing a particular submission.

- **NVIDIA CUDA C++ Programming Guide** — authoritative for the
  CUDA memory hierarchy, stream / event semantics, CUDA Graphs
  capture and instantiation (used in § 2.12), and warp
  specialization on Hopper.
- **NVIDIA NCCL Documentation** — collective semantics,
  environment variables (`NCCL_ALGO`, `NCCL_PROTO`,
  `NCCL_P2P_LEVEL`, `NCCL_IB_HCA`), `nccl-tests` — the source for
  the NCCL-tuning anti-patterns in § 5.1 and stretch bonus.
- **NVIDIA Nsight Systems and Nsight Compute User Guides** — the
  source of truth for the NVTX-range-based continuous-batching
  verification in § 3 step 11 and the kernel-level decode-path
  attribution at the reference-tier of D3.
- **NVIDIA A100 / H100 / Blackwell product briefs and datasheets**
  — for the peak fp16 / bf16 / fp8 TFLOPs, HBM bandwidth, and
  NVLink topology used in the parallelism-sweep ADR.
- **NVIDIA Multi-Instance GPU (MIG) User Guide** — the
  partition / profile semantics used in the multi-tenant stretch
  bonus (mod-008 ex-04).
- **NVIDIA TensorRT-LLM** documentation — engine build flags,
  in-flight batching configuration, the supported parallelism
  axes; the alternative to vLLM the mod-007 ex-01 framework-
  selection ADR weighs.
- **vLLM documentation** — `tensor_parallel_size`,
  `max_num_seqs`, `max_num_batched_tokens`, `enforce_eager`,
  PagedAttention block size, the prefix-cache hit-rate metric.
  The mod-006 reference orchestrator.
- **Kubernetes documentation** — HorizontalPodAutoscaler
  (external metrics), `readinessProbe` vs `livenessProbe`,
  `preStop` hooks, `Pod Disruption Budgets`. The authoritative
  source for the operational gates in § 2.6 – 2.10.
- **Prometheus and the Kubernetes external-metrics adapter**
  documentation — the API contract the HPA reads from.
- **PyTorch documentation**:
  - `torch.cuda.Event` — engine-internal timing primitive used
    in the bench-runner.
  - `torch.distributed` and `torch.distributed.nn.functional` —
    the all-reduce primitives the TP path depends on.

### Foundational papers cited in the project spec

<!-- spec-pin: confirm the project-03 README's reference list;
the entries below are the standard distributed-LLM-inference
reading that the reference solution defers to. -->

- Kwon et al., "Efficient Memory Management for Large Language
  Model Serving with PagedAttention", SOSP 2023 — the paged
  attention pattern used in § 2.4 and the canonical reference
  behind vLLM.
- Yu et al., "Orca: A Distributed Serving System for
  Transformer-Based Generative Models", OSDI 2022 — the
  iteration-level (continuous) batching pattern used in § 2.3.
- Patel et al., "Splitwise: Efficient Generative LLM Inference
  Using Phase Splitting" — disaggregated prefill / decode,
  stretch bonus.
- Qin et al., "Mooncake: A KVCache-centric Disaggregated
  Architecture for LLM Serving" — KV-cache disaggregation,
  stretch bonus.
- Leviathan et al., "Fast Inference from Transformers via
  Speculative Decoding" — speculative-decoding stretch bonus.
- Dao et al., "FlashAttention" / FlashAttention-2 /
  FlashAttention-3 — per-replica attention path; inherited
  upstream from project-02.

### Cross-track pointers

- `engineer-solutions/mod-110-llm-infrastructure/exercise-01-production-llm-serving`
  — the upstream API-gateway exercise that the project's routing
  layer extends.
- `engineer-solutions/mod-110-llm-infrastructure/exercise-06-inference-optimization-llm`
  — the single-replica LLM inference optimization chain whose
  kernels (project-02) and compression (project-01) feed each
  replica in this project's cluster.
- `senior-engineer-solutions/projects/project-201-distributed-training/SOLUTION.md`
  — the multi-GPU training counterpart; this project deliberately
  stays inference-side. The all-reduce / TP / PP primitives are
  the same; the SLO contract is different (training has no
  per-request latency).
- `architect-solutions/projects/project-301-enterprise-mlops/SOLUTION.md`
  — architecture-level cost / capacity reasoning that consumes the
  per-replica throughput and cold-start budgets produced here.
- `architect-solutions/projects/project-303-llm-rag-platform`
  (referenced in mod-006 SOLUTION.md "Related curriculum
  touchpoints") — the enterprise RAG architecture that wraps a
  serving cluster like this one.
