# SOLUTION — Production Deployment

> Read this *after* you have deployed and run a real inference
> service in production for at least a week. This document
> explains *why* the production deployment patterns are shaped the
> way they are and which corners are safe to cut.

## What this module is really teaching

Putting a model on a Kubernetes pod is the easy part. Keeping it
serving traffic with the right SLOs across hardware failures,
traffic spikes, model rollbacks, and cost pressure is where most
inference deployments actually live. This module exists to bridge
the gap between "I made it work in a notebook" and "we serve this
to paying customers with a 99.9% availability target."

The non-obvious truths the module teaches:

1. **Framework selection is a 3-year commitment.** Pick wrong and
   you'll spend more time on plumbing than on the model.
2. **Canary deployments require model-quality metrics, not just
   HTTP 200s.** A canary that returns garbage at 200 OK is worse
   than one that 500s.
3. **Spot instances cut cost in half but require designing for
   eviction.** No design = no savings (or worse).
4. **Multi-tier routing is the cheapest way to control cost.** 80%
   of traffic doesn't need the 70B model; route it to the 8B.
5. **End-to-end deploy is mostly waiting** — for image builds, for
   model downloads, for warm-up. Engineering happens at the edges.

## Architectural decisions and *why*

### Decision 1: Framework selection driven by workload, not popularity

Exercise 01 (framework selection) is deliberately not a "use vLLM"
exercise. It walks through the selection matrix:

| Framework | Best at | Worst at |
|---|---|---|
| **vLLM** | Multi-tenant LLM serving, OSS models | Custom architectures, very-low-latency single-stream |
| **TensorRT-LLM** | Maximum throughput on NVIDIA hardware | Iteration speed, OSS ecosystem |
| **SGLang** | Complex routing, structured output | Documentation, ecosystem maturity |
| **TGI** | HuggingFace-aligned, gentle ramp | Multi-tenant scale |
| **Triton + custom backend** | Mixed model types, classical ML alongside LLMs | Time-to-first-prediction |

The reference solution requires the student to **measure** at
least two frameworks on their target workload before choosing. A
framework that's 30% faster on average might be 2x worse on the
specific input distribution the team actually serves.

**Anti-pattern to avoid**: picking vLLM because "vLLM is the
fastest" without measuring. The benchmarks online are usually run
on workloads (long prefill, short decode) that don't match yours.

### Decision 2: Canary deployment with model-quality gates

Exercise 02 (canary deployment) extends standard Kubernetes canary
(Argo Rollouts or Flagger) with a **model-quality gate**: a
sampled fraction of canary traffic is logged with the model's
output and compared against the baseline model's output for the
same input. If the canary's response degrades on a curated eval
prompt set, the rollout halts.

The reference solution uses:
1. Argo Rollouts for the deployment mechanics.
2. A "shadow traffic" path that mirrors 1% of requests to both
   canary and baseline.
3. A diff-evaluator that scores response similarity (BLEU /
   ROUGE / LLM-as-judge) and rolls back if the canary diverges.

This is the **right pattern in 2026** but most teams still ship
with HTTP-only canaries. The exercise calls out the gap explicitly.

**Anti-pattern to avoid**: canary at 200 OK alone. A quantization
bug can produce syntactically valid but semantically broken
outputs at 100% HTTP 200.

### Decision 3: Spot resilience as a first-class deployment concern

Exercise 03 (spot resilience) treats spot/preemptible instance
eviction as a normal operating condition, not an exception. The
key design moves:

1. **PodDisruptionBudget** keeps a minimum number of replicas
   available during voluntary disruption.
2. **Preemption notification webhook** triggers a graceful drain:
   pod stops accepting new requests, finishes in-flight ones,
   then exits before the 30-90 second eviction grace window
   closes.
3. **Replica diversification** — never run all replicas in the
   same spot pool; spread across instance types and AZs so one
   pool's bid spike doesn't take down the service.
4. **Cold-start fallback to on-demand** — if spot capacity is
   unavailable, the HPA spins up on-demand nodes for capacity,
   accepting the cost premium during a spike.

The cost saving is real: 50-70% off on-demand prices, with
correctness preserved. Without the design, spot eviction during a
traffic spike causes cascading failures.

### Decision 4: Multi-tier routing — model selection as a deployment concern

Exercise 04 (multi-tier routing) implements an inference gateway
that routes requests across model tiers based on **request
classification**:

- Trivial requests (factual lookup, short answers) → 7B model.
- Complex requests (long reasoning, code generation) → 70B model.
- Domain-specific requests (legal, medical) → fine-tuned 13B.

The reference solution uses a lightweight classifier (a 350M
DeBERTa or rule-based router) to pick the tier. The cost savings
are dramatic: routing 80% of traffic to the 7B model can cut total
inference cost by 5-8x with negligible quality impact on the
routed-down traffic.

The teaching: **the cheapest GPU is the one you don't use**. The
right model for a query is rarely "the biggest model we have."

### Decision 5: End-to-end deploy as the integration exercise

Exercise 05 (end-to-end deploy) is the synthesis: model registry
→ image build → canary rollout → traffic shift → monitoring →
rollback. The reference solution wires together:

- MLflow Model Registry for the source of truth.
- A GitOps CD pipeline (Argo CD) that watches the registry.
- The custom HPA from mod-006 ex-03 for autoscaling.
- The prefix-aware routing from mod-006 ex-04.
- The canary quality gate from this module's ex-02.

The exercise is mostly about **plumbing**: the components are all
known, but wiring them so they observe each other's state without
race conditions is a non-trivial engineering exercise. The
reference walks through the failure modes (registry update without
image build, canary stuck at 50%, rollback during a deploy) so the
"happy path" demo isn't mistaken for the full story.

## Trade-offs we deliberately accepted

### Kubernetes assumed

The exercises target Kubernetes (EKS/GKE/AKS or self-managed). The
patterns (canary, HPA, PDB) transfer to other orchestrators, but
the YAML doesn't. Teams on ECS or pure VM deployments will need to
port the concepts; that's a one-time engineering exercise.

### Argo CD as the GitOps tool

Flux or Spinnaker would also work. Argo CD is chosen for its
maturity in ML platform deployments — it's the de facto standard
for Kubeflow, vLLM helm chart deployments, and the
mlops/architect tracks.

### No serverless GPU here

Modal, Replicate, and AWS Inferentia serverless are interesting
alternatives but have very different operational profiles (cold
starts, vendor lock-in, billing models). The exercises focus on
self-managed Kubernetes because that's the foundational skill;
serverless GPU is a "next chapter" topic.

## Common mistakes graders see

1. **Canary that only checks HTTP status**: misses semantic
   regression in model outputs.
2. **PDB with `minAvailable: 0`**: PDB does nothing; voluntary
   evictions can kill all replicas during a node drain.
3. **HPA scale-down too aggressive**: replicas churn during
   normal traffic dips, paying repeated cold-start costs. Set
   `scaleDown.stabilizationWindowSeconds` to ≥ 5 minutes.
4. **Spot eviction handled by retry alone**: works for stateless
   requests, not for streaming responses where the user has
   already seen partial tokens.
5. **Multi-tier routing without monitoring per tier**: you can't
   tell if the router is misclassifying or if the underlying tier
   is degraded.
6. **End-to-end deploy without a `git revert` rollback path**: the
   "rollback" button in MLflow is one path; a Git revert that
   triggers the same CD pipeline is the other. Have both.

## When to go beyond this implementation

- Add **request-level priority queues** so high-tier traffic
  preempts low-tier when GPU saturates.
- Implement **automatic shadow → canary → production promotion**
  driven by eval-set scores, not human gates.
- Move to **multi-region** with **DNS-based failover** for
  cross-region disaster recovery.

## Related curriculum touchpoints

- `performance/mod-006-distributed-inference` — the autoscaling +
  routing primitives this module deploys.
- `performance/mod-008-advanced-topics` — the bleeding-edge
  techniques you'll layer on once production is stable.
- `mlops/projects/project-2-model-serving` — the canary + registry
  workflow from the MLOps perspective.
- `architect/projects/project-301-enterprise-mlops` — the
  organizational layer wrapped around all of this.
