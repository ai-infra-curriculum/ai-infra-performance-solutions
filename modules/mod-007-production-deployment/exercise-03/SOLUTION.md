# Exercise 03 — Spot Instances with Resilience

## 1. Solution overview

Run an inference (or training) workload on **preemptible/spot** GPU
capacity for 50–70% off on-demand pricing while preserving SLOs by
treating eviction as an expected, not exceptional, event. The
reference solution combines four controls:

1. **Karpenter (or an equivalent autoscaler)** that provisions spot
   nodes first and falls back to on-demand when spot capacity is
   unavailable.
2. **PodDisruptionBudget (PDB)** so no voluntary drain (including
   the autoscaler consolidating nodes) can take the service below a
   quorum.
3. **Replica diversification** across instance types and
   availability zones, so one spot pool's capacity event does not
   evict every replica at once.
4. **Graceful preemption handling** inside the pod: on the eviction
   signal, stop accepting new requests, drain in-flight work (or
   checkpoint), and exit cleanly before the cloud provider's grace
   window closes.

The teaching objective is to internalize that **spot savings are
contingent on design**. A naive spot deployment is cheaper per
hour and more expensive per nine of availability; the controls
above are what turn the raw price difference into a real win.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md), Decision 3 ("Spot resilience as a
first-class deployment concern").

## 2. Worked answer or implementation

### 2.1 Karpenter NodePool — spot-first with on-demand fallback

The NodePool below instructs Karpenter to prefer spot capacity, but
allows on-demand when spot is unavailable. It also pins acceptable
instance families to GPU-capable types and diversifies across AZs.

```yaml
# modules/mod-007-production-deployment/exercise-03/nodepool.yaml
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
  name: gpu-inference
spec:
  template:
    metadata:
      labels:
        workload: inference
    spec:
      requirements:
        - key: karpenter.k8s.aws/instance-family
          operator: In
          values: ["g5", "g6", "p4d", "p5"]   # GPU families; adjust per cloud
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]       # spot preferred, on-demand allowed
        - key: topology.kubernetes.io/zone
          operator: In
          values: ["us-east-1a", "us-east-1b", "us-east-1c"]
        - key: kubernetes.io/arch
          operator: In
          values: ["amd64"]
      taints:
        - key: nvidia.com/gpu
          effect: NoSchedule
      nodeClassRef:
        group: karpenter.k8s.aws
        kind: EC2NodeClass
        name: gpu-default
  limits:
    cpu: "2000"
    nvidia.com/gpu: "64"
  disruption:
    consolidationPolicy: WhenEmptyOrUnderutilized
    consolidateAfter: 60s
```

Two decisions in the manifest matter:

- **`capacity-type: ["spot", "on-demand"]`** — Karpenter's
  weighted-priority provisioner treats spot as preferred and only
  requests on-demand when spot cannot be satisfied. This is the
  fallback that keeps the service up during a spot squeeze without
  a human in the loop.
- **Three zones listed** — a spot capacity event is typically
  per-pool-per-AZ. Replicas spread across three zones rarely all
  get evicted in the same minute.

### 2.2 PodDisruptionBudget — quorum under voluntary disruption

```yaml
# modules/mod-007-production-deployment/exercise-03/pdb.yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: inference-api
  namespace: inference
spec:
  minAvailable: "66%"
  selector:
    matchLabels: { app: inference-api }
  unhealthyPodEvictionPolicy: AlwaysAllow
```

`minAvailable: "66%"` keeps two-thirds of replicas up during any
voluntary drain (node upgrade, Karpenter consolidation, manual
`kubectl drain`). `unhealthyPodEvictionPolicy: AlwaysAllow` makes
sure a stuck crash-looping pod can still be evicted so capacity can
be recovered. **PDB covers voluntary disruption only — it does not
stop spot eviction itself** (that is an involuntary disruption from
the cluster's perspective); its job is to prevent the autoscaler
from making a bad situation worse by draining additional nodes
while spot capacity is already thin.

### 2.3 Replica diversification via topology spread

Spreading replicas across zones (and thus across independent spot
pools) is the single highest-leverage resilience control.

```yaml
# modules/mod-007-production-deployment/exercise-03/deployment-fragment.yaml
spec:
  template:
    spec:
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway   # prefer spread; do not block scheduling
          labelSelector:
            matchLabels: { app: inference-api }
        - maxSkew: 2
          topologyKey: node.kubernetes.io/instance-type
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels: { app: inference-api }
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
```

Two spread constraints are deliberate: AZ spread handles pool-level
events; instance-type spread handles "the `g5.12xlarge` pool got
reclaimed" events. `whenUnsatisfiable: ScheduleAnyway` is important
— a strict `DoNotSchedule` would turn a capacity shortage into a
scheduling outage, which is the exact failure mode the spot design
is meant to prevent.

### 2.4 Graceful preemption handler

Spot eviction arrives as a pod termination with a provider-specific
grace window (commonly 30–120 seconds). Inside the pod, the
`preStop` hook and the main process must cooperate so that
in-flight work is handled, not abandoned.

```yaml
# modules/mod-007-production-deployment/exercise-03/deployment-fragment.yaml (cont.)
      containers:
        - name: server
          image: registry.internal/inference-api:v2
          lifecycle:
            preStop:
              exec:
                command:
                  - /bin/sh
                  - -c
                  - |
                    # 1. Flip readiness probe false so the service stops
                    #    routing new requests to this pod.
                    touch /tmp/draining
                    # 2. Let in-flight requests finish; honor the provider
                    #    grace window minus a safety margin.
                    sleep 20
          readinessProbe:
            exec:
              command:
                - /bin/sh
                - -c
                - 'test ! -f /tmp/draining && curl -fs localhost:8000/healthz'
            periodSeconds: 2
            failureThreshold: 1
          terminationGracePeriodSeconds: 45
```

The flow is:

1. Kubelet calls `preStop` → the sentinel file appears → the
   readiness probe flips to failing → the Service endpoint
   controller removes the pod from load-balancer rotation.
2. In-flight requests complete during the `sleep` window.
3. SIGTERM is delivered; the server process shuts down.
4. For **training** workloads substitute the `sleep` with a
   checkpoint-write step, and recover from that checkpoint on the
   replacement node. The module SOLUTION.md flags this explicitly
   as "graceful checkpoint pattern".

`terminationGracePeriodSeconds: 45` is tuned to the cloud's spot
notice window (often ~2 minutes on AWS, ~30 seconds on GCP) with
margin. If you set the grace period longer than the provider's
notice window, the kernel kills the container anyway.

### 2.5 The spot-notice listener (optional but recommended)

Cloud providers expose an instance-metadata endpoint that signals
imminent reclamation **before** the pod SIGTERM. A small sidecar
can listen for this signal and flip readiness earlier, buying
additional drain time:

```python
# modules/mod-007-production-deployment/exercise-03/spot_watcher.py (sketch)
import time, pathlib, httpx
URL = "http://169.254.169.254/latest/meta-data/spot/instance-action"   # AWS example
SENTINEL = pathlib.Path("/tmp/draining")

while True:
    try:
        r = httpx.get(URL, timeout=1.0)
        if r.status_code == 200:
            SENTINEL.touch()
            break
    except httpx.HTTPError:
        pass
    time.sleep(2)
```

The sidecar touches the same `/tmp/draining` sentinel the `preStop`
hook uses, so the drain path is one code path regardless of signal
source. The metadata endpoint, polling interval, and expected
notice window differ per cloud; see each provider's documentation.

## 3. Validation steps

1. **Static validation**
   - `kubectl apply --dry-run=server -f nodepool.yaml` passes.
   - `kubectl apply --dry-run=server -f pdb.yaml` passes.
   - Rendered Deployment includes the `topologySpreadConstraints`
     and `preStop` hook.
2. **Scheduling on spot**
   - Scale the Deployment up; confirm with
     `kubectl get nodes -L karpenter.sh/capacity-type -L topology.kubernetes.io/zone`
     that provisioned nodes are `capacity-type=spot` and are
     distributed across at least two zones.
3. **PDB is honored**
   - `kubectl drain <one-node> --ignore-daemonsets` succeeds when
     the remaining replicas satisfy `minAvailable`. A drain that
     would breach the budget blocks with a PDB message; that is
     the expected behavior, not a failure.
4. **Graceful drain under eviction**
   - `kubectl delete pod <one-replica> --grace-period=60`.
   - Observe: readiness goes false immediately, in-flight requests
     complete, the pod exits cleanly within the grace window. No
     5xx spike in `http_requests_total{code=~"5.."}` for the
     service during this window.
5. **Spot fallback actually works**
   - Simulate spot exhaustion (either by constraining the NodePool
     to a sold-out instance family in a test region, or by using
     the provider's fault-injection tooling). Confirm Karpenter
     provisions on-demand replacement nodes within the autoscaling
     latency target rather than leaving pods pending.
6. **Replica diversification under failure**
   - Cordon/drain every node in one zone. Service availability
     holds above the PDB floor. If availability dips below the
     floor, either the PDB is wrong, the spread constraint is not
     being honored, or you have fewer AZs than replicas / the
     spread floor.
7. **Cost sanity (post-run)**
   - After one week of steady-state operation, confirm the share of
     `capacity-type=spot` node-hours vs on-demand node-hours. If
     on-demand share is high, the fallback is working — but the
     expected cost savings will not materialize. Revisit instance
     family selection, zone list, or bid strategy.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | Spot-first with on-demand fallback | NodePool accepts both capacity types; Karpenter provisioner prefers spot |
| 2 | PDB non-trivial | `minAvailable` ≥ 1 (and preferably ≥ 50% of replicas); not `minAvailable: 0` |
| 3 | AZ diversification | `topologySpreadConstraints` on `topology.kubernetes.io/zone`; NodePool lists ≥ 2 zones |
| 4 | Instance-type diversification | Either a second spread constraint on instance-type, or the NodePool lists ≥ 2 families |
| 5 | Graceful drain implemented | `preStop` + readiness flip; `terminationGracePeriodSeconds` tuned to provider's notice window |
| 6 | Training variant addressed (if applicable) | Checkpoint-on-SIGTERM path described; recovery resumes from last checkpoint, not scratch |
| 7 | Evidence of validation | At least one drain test and one fallback test executed with results recorded |
| 8 | Cost argument quantified | Candidate states the achievable savings range (50–70% off on-demand is the module reference) and the risk if the controls above are missing |

Scoring guidance:

- 7–8 passes: resilient spot deployment. Safe to run production
  non-critical workloads.
- 5–6 passes: usually missing the graceful drain, the instance-type
  spread, or the spot-notice listener. The deployment will survive
  most spot events but regress noticeably when a whole pool goes
  away.
- ≤ 4 passes: the YAML compiles but the design is "spot and pray".
  Send the candidate back to add either the PDB or the spread
  constraints; those are the two single-largest resilience levers.

## 5. Common mistakes

1. **`minAvailable: 0` on the PDB.** The PDB exists but does
   nothing. A node drain can evict every replica at once. The
   module SOLUTION.md calls this out explicitly as a top grader
   finding.
2. **No topology spread.** Kubernetes can colocate all replicas on
   one node or in one zone. One spot capacity event takes down
   100% of the service. Spread constraints are the cheapest fix.
3. **Graceful drain via retry at the client.** For stateless
   request/response APIs this is sometimes acceptable; for
   streaming or long-running inference the user has already seen
   partial output and a retry is not transparent.
4. **`terminationGracePeriodSeconds` longer than the provider
   notice window.** The kernel kills the container when the window
   closes regardless of grace period; in-flight work is still
   lost. Match the grace period to the window, not to wishes.
5. **Spot-only, no on-demand fallback.** During regional spot
   shortages the service goes partially down. The fallback does
   not have to be "always available"; it only has to exist as a
   lever during capacity events.
6. **Checkpointing to node-local storage for training.** When the
   spot node evicts, the checkpoint goes with it. Training
   checkpoints must land in durable, cross-node storage (object
   storage or a networked filesystem) to survive eviction.
7. **Measuring only hourly cost.** Spot wins on $/hour, but if the
   fallback shares creep up during capacity stress the realized
   savings are lower. Measure monthly realized blended cost, not
   headline spot price.
8. **Silently losing diversification over time.** As Karpenter
   consolidates, replicas may end up on fewer nodes/types than the
   constraints suggest at scale-up. Alert on
   `count by (zone)(kube_pod_info{...})` falling below 2.

## 6. References

- Karpenter documentation — `NodePool`, capacity-type selection
  (spot + on-demand), and consolidation:
  https://karpenter.sh/docs/
- Kubernetes documentation — PodDisruptionBudget semantics,
  voluntary vs involuntary disruption:
  https://kubernetes.io/docs/concepts/workloads/pods/disruptions/
- Kubernetes documentation — Pod topology spread constraints:
  https://kubernetes.io/docs/concepts/scheduling-eviction/topology-spread-constraints/
- Kubernetes documentation — container lifecycle hooks and
  graceful shutdown:
  https://kubernetes.io/docs/concepts/containers/container-lifecycle-hooks/
- AWS documentation — EC2 Spot Instance interruption notices
  (two-minute warning, instance metadata endpoint):
  https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-instance-termination-notices.html
- NIST AI Risk Management Framework — "Manage" function
  (incident response and operational resilience for AI systems)
  for framing spot design as operational risk, not just a cost
  lever: https://www.nist.gov/itl/ai-risk-management-framework
- Local module context —
  [`modules/mod-007-production-deployment/SOLUTION.md`](../SOLUTION.md),
  Decision 3 ("Spot resilience as a first-class deployment
  concern") and the "PDB with `minAvailable: 0`" anti-pattern.
- Local stub reference —
  [`modules/mod-007-production-deployment/exercise-03/README.md`](./README.md)
  points at the engineer-track Karpenter NodePool and PDB pattern
  this solution elaborates.
