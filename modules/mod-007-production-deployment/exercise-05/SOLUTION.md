# Exercise 05 — End-to-End Production Deploy

## 1. Solution overview

Wire together every prior piece of the module into a **single
deployable system** that a team could run in production:

- A model registry as the source of truth (MLflow Model Registry
  in the reference; Hugging Face Hub private repos, S3 + manifest
  files, or a vendor registry are drop-in substitutes).
- A **GitOps CD pipeline** (Argo CD) that reconciles the
  cluster against a Git-tracked manifest set, including the image
  tag and model version.
- The **Rollout with quality gate** from exercise-02 for progressive
  promotion.
- The **spot-resilient deployment shape** from exercise-03 for cost.
- The **multi-tier gateway** from exercise-04 in front of the model
  tiers.
- **Autoscaling, SLOs, cost attribution, and a runbook** attached
  to the whole thing.

The exercise is not about inventing new patterns. It is about
discovering that the integration is where the real engineering
is: the components all work individually, but wiring them so each
observes the others' state **without race conditions** is the
non-trivial part. The reference solution walks through the
integration failure modes explicitly.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md), Decision 5 ("End-to-end deploy as
the integration exercise"), including the "deploy without a `git
revert` rollback path" anti-pattern.

## 2. Worked answer or implementation

### 2.1 Architecture at a glance

```
        ┌──────────────────────────┐
        │   MLflow Model Registry  │  (source of truth; "Production" alias)
        └──────────┬───────────────┘
                   │  webhook on alias change
                   ▼
        ┌──────────────────────────┐
        │     Model-build job      │  bakes a container image per model version
        └──────────┬───────────────┘
                   │  pushes image, updates Git manifest
                   ▼
        ┌──────────────────────────┐
        │      GitOps repo         │  (Argo CD watches this; manifests only)
        └──────────┬───────────────┘
                   │  reconcile
                   ▼
        ┌──────────────────────────┐
        │   Kubernetes cluster     │
        │  ┌────────────────────┐  │
        │  │ Argo Rollout       │  │  canary via exercise-02
        │  │  ├─ Tier-S Deploy  │  │  (vLLM 8B on spot — ex-03 shape)
        │  │  ├─ Tier-L Deploy  │  │  (vLLM 70B on spot/on-demand mix)
        │  │  └─ Router Deploy  │  │  (gateway from ex-04)
        │  └─────────┬──────────┘  │
        │            │             │
        │  ┌─────────▼──────────┐  │
        │  │ Service / Ingress  │  │
        │  └────────────────────┘  │
        └──────────────────────────┘
                   │
                   ▼
        ┌──────────────────────────┐
        │  Prometheus + Grafana    │  SLO + cost dashboards + canary metrics
        │  Alertmanager → PagerDuty│  routed per severity
        └──────────────────────────┘
```

The arrows above form the integration contract. Each step has a
single owner; the exercise is to make each transition observable
and reversible.

### 2.2 Repository layout (GitOps)

```
infra/
  argocd/
    inference-app.yaml           # Argo CD Application (root)
  envs/
    prod/
      kustomization.yaml
      rollout.yaml               # from exercise-02 (image tag is parameterized)
      analysis-template.yaml     # from exercise-02
      tier-s-deployment.yaml     # from exercise-04
      tier-l-deployment.yaml     # from exercise-04
      router-deployment.yaml     # from exercise-04
      pdb.yaml                   # from exercise-03
      topology-spread-patch.yaml # from exercise-03
      hpa.yaml                   # per tier
      service.yaml
      ingress.yaml
      prometheus-rules.yaml      # SLO recording + alert rules
      grafana-dashboard.yaml     # ConfigMap, provisioned to Grafana
      servicemonitor.yaml        # Prometheus Operator scrape config
      runbook.md                 # linked from alert annotations
```

The image tag and `MODEL_VERSION` env value are the only fields
mutated by CD; everything else is code-review-gated like any
application change. See §2.5 for how the mutation happens.

### 2.3 Registry → image → manifest pipeline

The simplest working pipeline:

1. A learner promotes a model version in MLflow to the
   `Production` alias.
2. MLflow fires a webhook (or a scheduled poller detects the alias
   change — both work).
3. A CI job builds a container image that bakes the model files at
   `/model` and tags the image with `v<MODEL_VERSION>-<GIT_SHA>`.
4. The CI job opens a pull request to the GitOps repo that updates
   the `rollout.yaml` image tag and `MODEL_VERSION` env.
5. On merge, Argo CD reconciles. The Rollout controller sees a new
   pod spec and begins the canary from exercise-02.

The reason the CI job opens a **pull request** rather than pushing
directly is the `git revert` rollback path flagged in the module
SOLUTION.md: Git history is the authoritative record of what is
deployed, and reverting the commit rolls back exactly the change
that went out.

```yaml
# modules/mod-007-production-deployment/exercise-05/argocd-app.yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: inference-prod
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://git.internal/platform/inference-infra.git
    targetRevision: main
    path: envs/prod
  destination:
    server: https://kubernetes.default.svc
    namespace: inference
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - ApplyOutOfSyncOnly=true
      - CreateNamespace=true
```

`automated.selfHeal: true` is deliberate — a human running
`kubectl edit` on a managed resource gets overwritten on the next
reconcile, so the Git repo stays the single source of truth.

### 2.4 SLOs, alerts, and the runbook

Define SLOs narrowly enough that they correspond to user pain:

```yaml
# modules/mod-007-production-deployment/exercise-05/prometheus-rules.yaml
groups:
  - name: inference-slo
    rules:
      # Recording rules for the SLO numerator/denominator.
      - record: inference:requests:rate5m
        expr: sum(rate(http_requests_total{service=~"inference-.*"}[5m]))
      - record: inference:errors:rate5m
        expr: sum(rate(http_requests_total{service=~"inference-.*",code=~"5.."}[5m]))
      - record: inference:availability:ratio5m
        expr: 1 - (inference:errors:rate5m / inference:requests:rate5m)

      # SLO-based alert: fast burn (1h window, 14.4× budget).
      - alert: InferenceErrorBudgetBurnFast
        expr: |
          (1 - inference:availability:ratio5m) > (14.4 * (1 - 0.995))
        for: 5m
        labels: { severity: page, service: inference }
        annotations:
          summary: "Inference 99.5% SLO burning fast (last 5m)"
          runbook: "https://git.internal/platform/inference-infra/blob/main/envs/prod/runbook.md#error-budget-burn"

      # Latency SLO: p95 under 2s.
      - alert: InferenceLatencyP95
        expr: |
          histogram_quantile(0.95,
            sum by (le)(rate(http_request_duration_seconds_bucket{service=~"inference-.*"}[5m]))
          ) > 2
        for: 10m
        labels: { severity: ticket, service: inference }
        annotations:
          summary: "Inference p95 latency > 2s for 10m"
          runbook: "https://git.internal/platform/inference-infra/blob/main/envs/prod/runbook.md#latency-regression"
```

Two deliberate choices: burn-rate alerts (not raw threshold) so the
on-call is paged on the signal that correlates with actual SLO
violation, and every alert carries a runbook URL. The runbook —
sketched in §2.6 — is the artifact an on-call engineer actually
reads at 3 a.m.

### 2.5 Cost attribution

Attach labels at the pod level that join to the cloud bill:

```yaml
# modules/mod-007-production-deployment/exercise-05/cost-labels-patch.yaml
spec:
  template:
    metadata:
      labels:
        app.kubernetes.io/name: inference
        cost-center: ml-platform
        tier: S          # or L / X per Deployment
        model-family: llama-3.1-8b-instruct
```

Combined with per-tier metrics from the exercise-04 gateway
(`router_cost_dollars_total{tier=...}`) and cluster-level cost
tooling (OpenCost, Kubecost, or the cloud-native billing export),
the labels let finance reconcile "we spent $X on inference last
month, split $Y to tier-S and $Z to tier-L" with the request-level
accounting. Without the labels, cost allocation collapses into a
single "ML cluster" line item and the routing design in
exercise-04 loses its accountability.

### 2.6 Runbook sketch

```markdown
# inference-prod runbook

## Error-budget burn (fast)

**Symptoms**: page from `InferenceErrorBudgetBurnFast`.

1. Grafana → "Inference SLO" dashboard → confirm burn is current
   (not a one-shot spike).
2. `kubectl argo rollouts get rollout inference-api -n inference` —
   is a canary in progress?
   - If **yes** and the Analysis is `Failed`: the rollout will
     auto-abort; no action beyond acknowledging the page.
   - If **yes** and Analysis is `Running`: manual abort is
     available: `kubectl argo rollouts abort inference-api -n inference`.
   - If **no**: proceed to step 3.
3. `kubectl get events -n inference --sort-by=.lastTimestamp | tail -40` —
   look for OOMKilled, image-pull-failed, node-not-ready.
4. Per-tier split: Grafana → "Router" dashboard → which tier's
   error rate is elevated? If only `tier-L`, scale up or route down.
   If `tier-S` only, route up (gateway fallback should do this
   automatically — verify the metric `router_requests_total{
   tier="L",reason=~".*fallback.*"}` is nonzero).
5. Rollback decision:
   - Preferred: `git revert <merge SHA>` in the inference-infra
     repo → Argo CD reconciles → previous image returns.
   - Fallback: `kubectl argo rollouts undo inference-api -n inference`
     promotes the previous stable ReplicaSet. **Follow up with a
     Git revert** so the GitOps state matches the cluster state.

## Latency regression

1. Confirm tier: which tier's `router_request_latency_seconds` p95
   is above target?
2. Autoscaling: `kubectl get hpa -n inference` — is the tier at
   `maxReplicas`? If yes, raise `maxReplicas` (temporary) and open
   a capacity ticket.
3. Spot churn: `kubectl get events -n inference | grep -i spot` —
   a reclaim event correlates with the regression? Fallback to
   on-demand per exercise-03 should be covering this; confirm with
   `kubectl get nodes -L karpenter.sh/capacity-type`.

## Model-quality regression (silent)

Fired by `shadow-divergence` or `eval-pass-rate` from exercise-02.

1. The Rollout controller will auto-abort the canary. Confirm
   stable has the traffic.
2. Pull the eval case that regressed from the shadow diff export.
3. Open an incident for the model owner; do not re-attempt the
   promotion until the regression is understood.
```

## 3. Validation steps

1. **Static integration check**
   - `kubectl kustomize envs/prod/` renders cleanly and
     `kubectl apply --dry-run=server` passes.
   - Argo CD `Application` points at the expected path and
     revision; `argocd app get inference-prod` shows `Synced`.
2. **Registry-to-cluster round trip**
   - Promote a new model version in MLflow to `Production`.
   - Observe: CI job runs → image built → PR opened against
     `inference-infra` → on merge, Argo CD reconciles → Argo
     Rollouts begins canary → Analysis passes → traffic shifts to
     100% new version → stable ReplicaSet scales to zero.
   - No manual `kubectl` is required at any step.
3. **Rollback via Git**
   - Promote a deliberately-broken model (fails the eval set) and
     confirm Rollout auto-aborts.
   - Then force-promote it anyway (`argo rollouts promote`), then
     perform `git revert` of the merge commit. Confirm Argo CD
     brings the previous image back and the stable ReplicaSet is
     the old one.
4. **Spot resilience exercised**
   - Cordon the node(s) carrying one tier's replicas. The PDB and
     spread constraints keep the service serving; the HPA brings
     new replicas on a different node/zone. SLO error budget is
     not burned by the exercise.
5. **Router behavior under load**
   - Replay a captured traffic sample through the Ingress; the
     per-tier metrics show the expected traffic split, cost
     accumulates per tier, and no silent down-tier fallback
     occurs.
6. **Observability smoke tests**
   - Every alert in `prometheus-rules.yaml` can be fired manually
     with a synthetic condition (e.g. `kubectl patch` a
     `ServiceMonitor` to bad values) and reaches the on-call
     channel with the runbook URL rendered.
   - The Grafana dashboard loads; each panel has data for the last
     15 minutes.
7. **Deliberate incident drill**
   - Run at least one tabletop or game-day exercise: break
     something (image-pull failure, bad model, exhausted spot
     pool) and have an engineer who did not design the system
     resolve it using only the runbook. Capture the gaps; fix the
     runbook, not the engineer.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | Single source of truth for model version | MLflow (or equivalent) alias change is the only input that produces a production image |
| 2 | GitOps reconcile loop | Argo CD Application present; no step in the pipeline requires manual `kubectl` to deploy |
| 3 | Canary quality gate wired | The Rollout from exercise-02 is in the live path; a bad model is auto-aborted end-to-end |
| 4 | Spot design present | PDB + spread + preStop from exercise-03 appear in the rendered manifests for every tier |
| 5 | Multi-tier gateway present | Router from exercise-04 is in front of the tiers; metrics show per-tier traffic |
| 6 | SLOs expressed, not just thresholds | Burn-rate alerts referencing a stated availability/latency SLO, each with a runbook URL |
| 7 | Cost attribution to tier | Pod labels + per-tier metrics; candidate can produce a per-tier spend number for one billing period |
| 8 | Git-revert rollback path | At least one documented rollback via `git revert` observed end-to-end; Rollout `undo` is the fallback, not the primary path |
| 9 | Runbook is real | Runbook exists, is linked from alert annotations, and has been exercised by someone other than the author |

Scoring guidance:

- 8–9 passes: this is a system a team can operate. The candidate
  is ready for the module's follow-on work.
- 6–7 passes: pipeline works but observability or the runbook is
  thin. Fix those two first — they are the on-call experience.
- ≤ 5 passes: the pieces are present but the integration is not.
  Walk the candidate through the "promote a bad model, watch it
  auto-rollback" flow end-to-end before signing off; this is the
  exercise.

## 5. Common mistakes

1. **Registry update without image rebuild.** The alias flipped to
   `v5` in MLflow, but no CI job watches that alias; the cluster is
   still serving `v4`. The pipeline's weakest link is almost always
   this webhook/poller.
2. **Image tag mutated in Git by a human.** Direct commits to the
   GitOps repo bypass the model-build job's provenance. Only the
   CI job should write image tags; humans review the PR.
3. **Argo CD `selfHeal` off.** A `kubectl edit` against a managed
   resource persists silently; the next deploy overwrites it with
   a surprise. `selfHeal: true` makes drift noisy immediately.
4. **Canary quality gate not in the production path.** The gate
   from exercise-02 is in a staging Rollout but production uses a
   vanilla Deployment with `kubectl rollout` semantics. The whole
   module's teaching is bypassed.
5. **Rollback via Rollouts `undo` without Git revert.** The
   cluster is restored to the previous image, but the GitOps state
   still points at the broken version; the next Argo CD reconcile
   redeploys the broken image. `undo` is a tool for fast cluster
   action; `git revert` is the authoritative rollback.
6. **Alerts without runbooks.** A page without an actionable
   runbook is a tax on the on-call engineer. Every alert created
   by this exercise must have a `runbook` annotation that points
   at a section covering that alert specifically.
7. **Cost labels missing or inconsistent.** Mixed labels (`tier=S`
   on one Deployment, `size=small` on another) make the billing
   view uninterpretable. The label set should be committed as a
   convention and lint-checked at CI.
8. **Tabletop skipped.** The runbook is read by the author and
   found sufficient. Someone else must try it on the real
   failure modes; the gaps are the exercise's real output.

## 6. References

- Argo CD documentation — GitOps `Application`, sync policies,
  self-heal:
  https://argo-cd.readthedocs.io/
- Argo Rollouts documentation — progressive delivery, analysis
  (used by exercise-02 and by this integration):
  https://argo-rollouts.readthedocs.io/
- MLflow documentation — Model Registry, aliases, and webhooks:
  https://mlflow.org/docs/latest/model-registry.html
- Prometheus documentation — recording rules and alerting rules,
  used for the SLO burn-rate alerts:
  https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/
- Kubernetes documentation — HorizontalPodAutoscaler,
  PodDisruptionBudget, topology spread (invoked from exercise-03
  manifests brought into this integration):
  https://kubernetes.io/docs/
- NIST AI Risk Management Framework — "Govern" and "Manage"
  functions covering model lifecycle, change control, and incident
  response for AI systems — the governance frame around the
  pipeline described here:
  https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Machine Learning Security Top 10 — supply-chain and
  deployment-integrity concerns relevant to the "who writes the
  image tag" discussion:
  https://owasp.org/www-project-machine-learning-security-top-10/
- MITRE ATLAS — tactics/techniques against ML systems in
  production, useful when designing what the canary gate and
  runbook need to detect/respond to:
  https://atlas.mitre.org/
- Local module context —
  [`modules/mod-007-production-deployment/SOLUTION.md`](../SOLUTION.md),
  Decision 5 ("End-to-end deploy as the integration exercise") and
  the "deploy without a `git revert` rollback path" anti-pattern.
- Local prior exercises —
  [`exercise-02`](../exercise-02/SOLUTION.md) (canary quality gate),
  [`exercise-03`](../exercise-03/SOLUTION.md) (spot resilience),
  [`exercise-04`](../exercise-04/SOLUTION.md) (multi-tier gateway)
  — this exercise composes them.
- Local stub reference —
  [`modules/mod-007-production-deployment/exercise-05/README.md`](./README.md)
  lists the engineer-track components (FastAPI factory, Helm chart,
  SLO + error budgets, cost tagging, runbook) this solution
  elaborates.
