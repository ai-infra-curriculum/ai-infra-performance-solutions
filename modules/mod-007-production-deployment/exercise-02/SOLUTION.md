# Exercise 02 — Canary via Argo Rollouts

## 1. Solution overview

Deploy a new model version alongside the current baseline, shift a
controllable fraction of production traffic to the new version, and
promote or roll back **based on a model-quality signal — not just
HTTP status**. The reference implementation uses
[Argo Rollouts](https://argo-rollouts.readthedocs.io/) for the
traffic-shift mechanics and an `AnalysisTemplate` to gate each
promotion step on measurable health.

This exercise is deliberately split into two concerns that the
learner must keep separate:

1. **Traffic mechanics** — percentage splits, step weights, pause
   windows, and the automated promotion decision. This is a
   Kubernetes/service-mesh exercise and is well-documented upstream.
2. **The health signal the gate reads** — HTTP success rate is
   necessary but not sufficient for a model deployment. A quantized
   or fine-tuned replacement can degrade semantic output quality
   while returning 100% HTTP 200. The reference solution requires
   at least one **model-quality metric** in the analysis template
   alongside the operational metrics.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md), Decision 2 ("Canary deployment with
model-quality gates").

## 2. Worked answer or implementation

### 2.1 Rollout manifest

The Rollout below shifts traffic in five steps (10% → 25% → 50% →
75% → 100%), pausing at each step for an `AnalysisRun` to score the
canary against the baseline. If any step's analysis fails, Argo
Rollouts aborts and scales the canary ReplicaSet back to zero; the
baseline ReplicaSet never lost traffic.

```yaml
# modules/mod-007-production-deployment/exercise-02/rollout.yaml
apiVersion: argoproj.io/v1alpha1
kind: Rollout
metadata:
  name: inference-api
  namespace: inference
spec:
  replicas: 6
  selector:
    matchLabels: { app: inference-api }
  template:
    metadata:
      labels: { app: inference-api }
    spec:
      containers:
        - name: server
          image: registry.internal/inference-api:REPLACED_BY_CD
          ports: [{ containerPort: 8000 }]
          readinessProbe:
            httpGet: { path: /healthz, port: 8000 }
            initialDelaySeconds: 30
            periodSeconds: 5
          resources:
            limits:  { nvidia.com/gpu: 1 }
            requests: { nvidia.com/gpu: 1 }
  strategy:
    canary:
      canaryService: inference-api-canary
      stableService: inference-api-stable
      trafficRouting:
        # Replace with your mesh of choice (Istio, SMI, ALB, NGINX).
        istio:
          virtualService:
            name: inference-api
            routes: [primary]
      steps:
        - setWeight: 10
        - pause: { duration: 10m }
        - analysis:
            templates:
              - templateName: canary-quality-gate
            args:
              - name: canary-service
                value: inference-api-canary
              - name: stable-service
                value: inference-api-stable
        - setWeight: 25
        - pause: { duration: 15m }
        - analysis:
            templates: [{ templateName: canary-quality-gate }]
            args:
              - { name: canary-service, value: inference-api-canary }
              - { name: stable-service, value: inference-api-stable }
        - setWeight: 50
        - pause: { duration: 20m }
        - analysis:
            templates: [{ templateName: canary-quality-gate }]
            args:
              - { name: canary-service, value: inference-api-canary }
              - { name: stable-service, value: inference-api-stable }
        - setWeight: 75
        - pause: { duration: 20m }
        - setWeight: 100
```

### 2.2 Analysis template — operational + quality signals

The `AnalysisTemplate` runs for the duration of each pause. It
evaluates **four** metrics; any one failing aborts the step.

```yaml
# modules/mod-007-production-deployment/exercise-02/analysis-template.yaml
apiVersion: argoproj.io/v1alpha1
kind: AnalysisTemplate
metadata:
  name: canary-quality-gate
  namespace: inference
spec:
  args:
    - name: canary-service
    - name: stable-service
  metrics:
    # 1. HTTP success rate — necessary floor, not sufficient on its own.
    - name: success-rate
      interval: 1m
      count: 10
      successCondition: result[0] >= 0.995
      failureLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            sum(rate(
              http_requests_total{service="{{args.canary-service}}",code!~"5.."}[2m]
            ))
            /
            sum(rate(
              http_requests_total{service="{{args.canary-service}}"}[2m]
            ))

    # 2. p95 latency must not regress more than 20% vs stable.
    - name: latency-p95-ratio
      interval: 1m
      count: 10
      successCondition: result[0] <= 1.20
      failureLimit: 2
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            histogram_quantile(0.95,
              sum by (le) (rate(
                http_request_duration_seconds_bucket{service="{{args.canary-service}}"}[2m]
              ))
            )
            /
            histogram_quantile(0.95,
              sum by (le) (rate(
                http_request_duration_seconds_bucket{service="{{args.stable-service}}"}[2m]
              ))
            )

    # 3. Model quality — eval-set pass rate from an offline job running
    #    the curated prompt set against the canary endpoint.
    - name: eval-pass-rate
      interval: 5m
      count: 4
      successCondition: result[0] >= 0.98
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            model_eval_pass_rate{service="{{args.canary-service}}"}

    # 4. Output-distribution divergence vs stable (shadow-traffic diff).
    #    Scored by the shadow job described in §2.3. A sustained jump
    #    here catches semantic regressions that pass the eval set.
    - name: shadow-divergence
      interval: 5m
      count: 4
      successCondition: result[0] <= 0.15
      failureLimit: 1
      provider:
        prometheus:
          address: http://prometheus.monitoring.svc:9090
          query: |
            model_shadow_divergence{service="{{args.canary-service}}"}
```

Only metrics 1 and 2 are generic SRE signals. Metrics 3 and 4 are
what make this a **model** canary rather than a generic web-service
canary; without them the gate cannot catch a canary that returns
degraded content at 200 OK.

### 2.3 Shadow-traffic diff job (sketch)

The `shadow_divergence` metric comes from a sidecar/batch job that:

1. Samples a small fraction (1–5%) of production requests.
2. Sends each sampled request to **both** the stable and canary
   services and records both responses.
3. Scores each pair with a response-similarity metric — BLEU/ROUGE
   for structured text, exact-match for classification endpoints,
   or an LLM-as-judge rubric for open-ended generation.
4. Exports the moving-average divergence to Prometheus as
   `model_shadow_divergence{service="..."}`.

```python
# modules/mod-007-production-deployment/exercise-02/shadow_diff.py (sketch)
import httpx
from prometheus_client import Gauge, start_http_server

DIVERGENCE = Gauge(
    "model_shadow_divergence",
    "Mean per-request divergence between canary and stable outputs",
    ["service"],
)

async def sample_and_compare(request, stable_url, canary_url, scorer):
    async with httpx.AsyncClient() as http:
        stable  = (await http.post(stable_url,  json=request)).json()
        canary  = (await http.post(canary_url,  json=request)).json()
    return scorer(stable, canary)   # 0.0 identical, 1.0 unrelated
```

The scorer is task-specific — the exercise expects the candidate to
state which scorer they chose and why, not to invent a universal
one.

### 2.4 Why these step weights and pause durations

- **Step weights (10/25/50/75/100)** keep the first exposure small
  so a catastrophic regression is bounded to ~10% of traffic, and
  widen as confidence grows.
- **Pause duration ≥ eval-set runtime**: if the eval set takes 8
  minutes end-to-end, a 5-minute pause lets the rollout advance
  before the slowest signal has data. The durations above assume a
  sub-5-minute eval; tune to your eval-set latency.
- **Metric `count` ≥ 4** on the slower signals: avoids a single
  noisy sample failing the gate.

### 2.5 Promotion and rollback

- **Automatic promotion**: all metrics meet their `successCondition`
  for every step → Argo Rollouts shifts the final 25% and scales
  the stable ReplicaSet down. The new version becomes stable.
- **Automatic rollback**: any metric fails its condition more times
  than `failureLimit` → Argo aborts, re-routes 100% of traffic to
  stable, and scales the canary to zero. The old version never
  left service.
- **Manual override**: `kubectl argo rollouts promote <name>` or
  `abort <name>` for operator intervention.

## 3. Validation steps

1. **Install/confirm Argo Rollouts**
   - `kubectl argo rollouts version` resolves; controller pod is
     running in `argo-rollouts` namespace.
2. **Dry-run the manifest**
   - `kubectl apply --dry-run=server -f rollout.yaml` passes.
   - `kubectl apply --dry-run=server -f analysis-template.yaml`
     passes.
3. **Apply and observe baseline**
   - `kubectl apply -f rollout.yaml`
   - `kubectl argo rollouts get rollout inference-api -w` shows
     `Status: Healthy`, 6/6 stable, 0 canary.
4. **Trigger a canary**
   - Bump the image tag:
     `kubectl argo rollouts set image inference-api server=registry.internal/inference-api:v2`
   - Watch the rollout progress through each step. The analysis run
     at 10% should print metric results in the controller log.
5. **Inject a regression; confirm rollback**
   - Deploy a deliberately-broken image (e.g. a quantized model
     that fails the eval-set).
   - Verify the `eval-pass-rate` metric falls below threshold,
     `AnalysisRun` enters `Failed`, and Argo Rollouts status
     shows `Degraded`; traffic returns fully to stable within one
     step cycle.
6. **Verify no stable-service disruption during rollback**
   - `http_requests_total{service="inference-api-stable",code=~"5.."}`
     did not spike during the abort — the stable ReplicaSet was
     never touched.
7. **Verify both quality metrics actually fire**
   - A rollout that passes operational metrics but fails
     `shadow-divergence` or `eval-pass-rate` is also rolled back.
     Run at least one test case where only the quality signal is
     bad; this is the step that makes the exercise a model canary
     rather than a generic web canary.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | Progressive traffic shift present | Rollout defines ≥ 3 `setWeight` steps with pauses between them |
| 2 | Operational metrics gated | `AnalysisTemplate` includes success-rate and latency, with thresholds, not just presence |
| 3 | At least one model-quality metric gated | Eval-pass-rate **or** shadow-divergence (ideally both) with a numeric threshold |
| 4 | Shadow traffic described concretely | Candidate names the sampling rate, the scorer, and how the metric is exported |
| 5 | Rollback path end-to-end | Candidate can show a failing analysis aborting the rollout and returning traffic to stable |
| 6 | Pause ≥ metric latency | Each pause is at least as long as the slowest metric's `interval × count` |
| 7 | Noise handling | Metrics set a `failureLimit` > 1 so a single noisy sample does not abort |
| 8 | Scorer choice justified | Candidate explains why the chosen similarity metric fits the task (e.g. exact-match for classification, LLM-as-judge for open-ended generation) |

Scoring guidance:

- 7–8 passes: ready to own a production canary. Move on to end-to-end
  deploy (exercise-05).
- 5–6 passes: the canary mechanics are right but the quality signal
  is weak. Have the candidate add or sharpen metric 3 or 4 before
  signing off.
- ≤ 4 passes: the candidate has learned Argo Rollouts, not how to
  deploy a model. Send them back to design the quality gate; this
  is the decision the exercise exists to teach.

## 5. Common mistakes

1. **HTTP-200-only canary.** The module SOLUTION.md flags this
   explicitly: a quantization bug can produce syntactically valid
   but semantically broken outputs at 100% HTTP 200. If the only
   gate is success rate, the canary promotes.
2. **Pause shorter than the metric evaluation window.** If the
   eval-pass-rate metric runs every 5 minutes with `count: 4`, a
   10-minute pause can advance the rollout before the metric has
   enough samples to fail. Pauses must cover `interval × count`.
3. **Missing `failureLimit`.** `failureLimit: 0` makes a single
   noisy data point abort the rollout; `failureLimit: 1` on a
   `count: 4` metric is the usual compromise.
4. **Shadow traffic sent to production path, not a shadow path.**
   Doubling the request volume on the stable service to collect
   diffs can itself trip latency alerts. The shadow comparator must
   call the stable and canary services, not duplicate the live
   user's request against the same endpoint twice.
5. **Metric defined but never populated.** The `AnalysisTemplate`
   can query a Prometheus series that has never received a sample,
   which Argo Rollouts reports as "Inconclusive" — not "Failed".
   Candidates read this as a pass. The scorer job must be verified
   running before the first canary is attempted.
6. **Rollback without an eval-set snapshot.** When a rollback
   fires, the reason — "which eval case regressed" — is often the
   most important signal. If the shadow-diff job only exports a
   scalar, you cannot debug the regression. Export per-case
   results too, not only the aggregate.
7. **Quality metric scored by the canary itself.** The scorer must
   be an independent process; a canary that self-reports its own
   eval-pass-rate can be optimized to lie.

## 6. References

- Argo Rollouts documentation — `Rollout` spec, canary strategy,
  and `AnalysisTemplate`:
  https://argo-rollouts.readthedocs.io/
- Argo Rollouts analysis & progressive delivery overview:
  https://argoproj.github.io/argo-rollouts/features/analysis/
- Prometheus documentation — `histogram_quantile` and `rate` used in
  the analysis queries: https://prometheus.io/docs/
- NIST AI Risk Management Framework — "Measure" function
  (continuous monitoring of AI system performance post-deployment)
  informs why a model-quality gate belongs in the rollout:
  https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Machine Learning Security Top 10 — ML output monitoring
  as a control against model-serving integrity failures:
  https://owasp.org/www-project-machine-learning-security-top-10/
- MITRE ATLAS — model-serving tactics and techniques; useful when
  deciding what the canary should detect beyond simple regressions:
  https://atlas.mitre.org/
- Local module context —
  [`modules/mod-007-production-deployment/SOLUTION.md`](../SOLUTION.md),
  Decision 2 ("Canary deployment with model-quality gates") and the
  anti-pattern "Canary that only checks HTTP status".
- Local stub reference —
  [`modules/mod-007-production-deployment/exercise-02/README.md`](./README.md)
  points at the engineer-track canary rollout spec this solution
  elaborates.
