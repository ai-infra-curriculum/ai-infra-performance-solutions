# Exercise 04 — Multi-Tier Cost Routing

## 1. Solution overview

Deploy multiple model tiers (a cheap small model, an expensive
large model, optionally a domain-specialist model) behind a single
inference gateway. A **router** inspects each incoming request and
dispatches it to the cheapest tier that is good enough. The win is
asymmetric: most production LLM traffic does not require the
flagship model, so routing even a majority of requests down one
tier reduces aggregate GPU spend substantially while keeping
quality within an acceptable band for the routed-down slice.

The exercise is not "add a cheaper model and call it done". It is
to implement:

1. A **classifier/router** that picks a tier per request.
2. A **per-tier backend** (vLLM/Triton deployments) with
   independent scaling and health.
3. A **measurement harness** that quantifies cost per request **and
   quality per tier**, so the routing policy is accountable.
4. A **fallback policy** when the chosen tier is saturated,
   unavailable, or refuses the request.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md), Decision 4 ("Multi-tier routing —
model selection as a deployment concern"). The module's reference
target is that routing ~80% of traffic to a 7B model (vs a 70B
default) can cut aggregate inference cost by 5–8× with negligible
quality impact on the routed-down slice; the engineer-track
reference implementation linked from the exercise README reports a
~60% measured savings on a specific workload. Candidates should
treat these as **plausible targets to reproduce on their own
workload**, not as asserted outcomes.

## 2. Worked answer or implementation

### 2.1 Tier design

| Tier | Example model | Role | Typical fraction |
|---|---|---|---|
| Small (S) | Llama-3.1-8B, Mistral-7B | Factual lookup, short answers, format conversion | 60–80% |
| Large (L) | Llama-3.1-70B, Mixtral-8x22B | Long reasoning, multi-step code, nuanced generation | 15–30% |
| Specialist (X) | Fine-tuned 13B per domain (optional) | Legal/medical/code domain-specific calls | 5–15% |

Fractions are **workload-dependent** and the exercise expects the
candidate to **measure** their own distribution, not assume it.

### 2.2 Router — rule-based or classifier-based

The reference supports both; a candidate should pick one and defend
the choice.

**Rule-based** (cheaper, more legible, harder on long-tail inputs):

```python
# modules/mod-007-production-deployment/exercise-04/router_rules.py
import re
from dataclasses import dataclass


@dataclass
class RoutingDecision:
    tier: str                 # "S", "L", or "X"
    reason: str               # for logging/debugging the policy
    allow_fallback: bool = True


KEYWORD_SPECIALIST = {
    "legal":  re.compile(r"\b(contract|clause|statute|jurisdiction)\b", re.I),
    "medical": re.compile(r"\b(diagnosis|patient|dose|ICD-?10)\b", re.I),
}
TRIVIAL_PATTERNS = [
    re.compile(r"^(what is|who is|when did|define)\b", re.I),
    re.compile(r"^(convert|translate|summarize in \d+ words)\b", re.I),
]


def route(request: dict) -> RoutingDecision:
    prompt = request.get("prompt", "") or request.get("messages", [{}])[-1].get("content", "")

    # 1. Domain match → specialist tier (if deployed).
    for domain, pattern in KEYWORD_SPECIALIST.items():
        if pattern.search(prompt):
            return RoutingDecision(tier="X", reason=f"domain:{domain}")

    # 2. Short, trivial prompts → small tier.
    if len(prompt) < 200 and any(p.search(prompt) for p in TRIVIAL_PATTERNS):
        return RoutingDecision(tier="S", reason="trivial-short")

    # 3. Long context or explicit "reason step by step" → large tier.
    if len(prompt) > 4000 or "step by step" in prompt.lower():
        return RoutingDecision(tier="L", reason="long-or-reasoning")

    # 4. Everything else → small tier with fallback enabled.
    return RoutingDecision(tier="S", reason="default-small", allow_fallback=True)
```

**Classifier-based** (handles the long tail, requires training and
monitoring):

```python
# modules/mod-007-production-deployment/exercise-04/router_classifier.py (sketch)
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "microsoft/deberta-v3-small"   # ~140M, CPU-servable
LABELS = ["S", "L", "X"]

tok  = AutoTokenizer.from_pretrained(MODEL)
mdl  = AutoModelForSequenceClassification.from_pretrained(
    MODEL, num_labels=len(LABELS)
)


@torch.inference_mode()
def route(prompt: str, threshold_L: float = 0.4) -> tuple[str, dict]:
    enc = tok(prompt, truncation=True, max_length=512, return_tensors="pt")
    probs = mdl(**enc).logits.softmax(-1)[0].tolist()
    # Escalate to L whenever L's probability is above a conservative threshold —
    # routing errors toward "too big" are quality-safe, toward "too small" are not.
    if probs[1] >= threshold_L:
        return "L", {"probs": dict(zip(LABELS, probs))}
    tier = LABELS[int(torch.tensor(probs).argmax())]
    return tier, {"probs": dict(zip(LABELS, probs))}
```

**Trade-off**: rules are transparent and easy to audit; a classifier
captures signals a rule set cannot (sentence complexity, implicit
reasoning). The classifier needs its own labeled dataset, its own
monitoring, and its own retraining schedule. For a first version,
start with rules; migrate to a classifier once you have labeled
traffic to train on.

### 2.3 Gateway service

The gateway sits in front of all tiers. It calls `route()`, looks
up the backend URL, forwards the request, and on failure optionally
retries at a higher tier.

```python
# modules/mod-007-production-deployment/exercise-04/gateway.py
import os, time
from fastapi import FastAPI, HTTPException
import httpx
from prometheus_client import Counter, Histogram, make_asgi_app

from router_rules import route, RoutingDecision

TIER_URLS = {
    "S": os.environ["TIER_S_URL"],
    "L": os.environ["TIER_L_URL"],
    "X": os.environ.get("TIER_X_URL"),
}
# Cost weights (tokens-per-dollar numbers are workload-specific; populate from
# the measurement harness in §2.5, not from a guess).
TIER_COST_PER_1K_TOK = {
    "S": float(os.environ.get("TIER_S_COST", "0.10")),
    "L": float(os.environ.get("TIER_L_COST", "1.00")),
    "X": float(os.environ.get("TIER_X_COST", "0.30")),
}

app = FastAPI()
app.mount("/metrics", make_asgi_app())

REQ    = Counter("router_requests_total",
                 "Requests per tier/outcome", ["tier", "outcome", "reason"])
LAT    = Histogram("router_request_latency_seconds",
                   "End-to-end latency per tier", ["tier"])
COST   = Counter("router_cost_dollars_total",
                 "Attributed cost per tier", ["tier"])


@app.post("/v1/completions")
async def completions(payload: dict):
    decision: RoutingDecision = route(payload)
    tried: list[str] = []
    order = [decision.tier]
    if decision.allow_fallback and decision.tier == "S":
        order.append("L")   # escalate on failure

    for tier in order:
        url = TIER_URLS.get(tier)
        if not url:
            continue
        tried.append(tier)
        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as http:
                r = await http.post(f"{url}/v1/completions", json=payload)
            r.raise_for_status()
            resp = r.json()
            LAT.labels(tier=tier).observe(time.perf_counter() - start)
            usage = resp.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            COST.labels(tier=tier).inc(
                tokens / 1000.0 * TIER_COST_PER_1K_TOK[tier]
            )
            REQ.labels(tier=tier, outcome="ok",
                       reason=decision.reason).inc()
            resp["routing"] = {"tier": tier, "tried": tried,
                               "reason": decision.reason}
            return resp
        except httpx.HTTPError:
            REQ.labels(tier=tier, outcome="error",
                       reason=decision.reason).inc()
            continue   # try next tier in `order`

    raise HTTPException(status_code=503, detail={"tried": tried})
```

Two important properties of this gateway:

- **It records the routing decision in the response** (`routing`
  field). Downstream logging can then compare which tier served
  which request without parsing logs.
- **Fallback is one-way (S → L), never L → S.** Escalating on
  failure is quality-safe; de-escalating would silently degrade.

### 2.4 Per-tier backend deployment

Each tier is a standard vLLM/Triton deployment with its own HPA. A
minimal sketch:

```yaml
# modules/mod-007-production-deployment/exercise-04/tier-s.yaml (fragment)
apiVersion: apps/v1
kind: Deployment
metadata: { name: tier-s, namespace: inference }
spec:
  replicas: 4
  selector: { matchLabels: { app: tier-s } }
  template:
    metadata:
      labels: { app: tier-s, tier: S }
    spec:
      containers:
        - name: vllm
          image: vllm/vllm-openai:latest
          args:
            - --model=meta-llama/Llama-3.1-8B-Instruct
            - --tensor-parallel-size=1
            - --port=8000
          resources:
            limits: { nvidia.com/gpu: 1 }
```

Deploy an analogous `tier-l.yaml` (a 70B with `TP=4` or
`TP=8`) and an optional `tier-x.yaml` (fine-tuned 13B). Scale each
tier independently; the whole point of this design is that you do
not pay for large-tier capacity you are not using.

### 2.5 Measurement harness — don't trust, measure

The exercise is only complete when the candidate has measured:

1. **Traffic mix per tier** (shares of total requests).
2. **Cost per request per tier** (tokens × tier price).
3. **Quality per tier on a shared eval set**, including the
   routed-down tier evaluated on prompts that the router sent to
   it in production.

The measurement harness replays a captured traffic sample through
each tier and scores quality with the same scorer used in the
canary quality gate (exercise-02). The output is a table of
`(tier, cost/request, quality score)` which is the artifact the
exercise is actually asking for; the router implementation is the
means.

## 3. Validation steps

1. **Unit-level routing**
   - Run the router against a labeled test set of prompts; every
     prompt in the specialist/large/small buckets is routed
     correctly (or the classifier reports confusion-matrix numbers
     that match the committed policy threshold).
2. **End-to-end gateway**
   - Bring up at least two tiers (`tier-s`, `tier-l`). The gateway
     forwards a trivial prompt to `tier-s` and a complex prompt to
     `tier-l`; the `routing` field in the response matches the
     decision.
3. **Fallback**
   - Kill the `tier-s` deployment. A request routed to S by the
     policy must succeed via L fallback; the `tried` field in the
     response lists `["S", "L"]`. Metrics show
     `router_requests_total{tier="S",outcome="error"}` incrementing
     and `router_requests_total{tier="L",outcome="ok"}` catching
     the retry.
4. **No reverse fallback**
   - Kill `tier-l`. A request routed to L must **not** silently
     answer from `tier-s`. It must fail or return a degraded
     response with an explicit degradation flag. Silent down-tier
     fallback is a quality bug, not a resilience feature.
5. **Cost accounting**
   - Replay a captured hour of traffic through the gateway in a
     staging deployment; `router_cost_dollars_total` broken down
     by tier reflects the cost mix. Compare against the "L only"
     counterfactual by replaying the same traffic with the policy
     forced to L — the cost delta is the realized saving for that
     window.
6. **Quality per tier**
   - For each tier, score the scorer against a curated eval set
     drawn from **that tier's actual traffic slice**. The small
     tier's score on the small slice is the quality number that
     matters, not its score on prompts it would never see.
7. **Observability**
   - Grafana (or equivalent) shows per-tier request rate,
     latency, cost, and error rate. One-tier-only dashboards hide
     the regressions that this design is specifically prone to.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | At least two tiers deployed | Independent Deployments; different model sizes; scaled independently |
| 2 | Routing policy committed, not ad-hoc | A single `route()` function (rules or classifier); covered by tests |
| 3 | Routing decision observable | Response carries a `routing` block; metrics labeled by tier and reason |
| 4 | Upward fallback only | S → L escalation on error exists; L → S never happens silently |
| 5 | Cost attributed per tier | `router_cost_dollars_total{tier=...}` or equivalent; per-tier price configurable, not hardcoded |
| 6 | Quality measured per tier on its own traffic | Eval numbers per tier against the prompts that tier actually receives |
| 7 | Saturation handled | If the chosen tier is saturated, the policy either fails fast or escalates; it does not queue unbounded |
| 8 | Reproducible savings number | Candidate produces a before/after cost figure on their own traffic, not a copied headline figure |

Scoring guidance:

- 7–8 passes: the router is a controllable cost lever. Move on to
  end-to-end deploy (exercise-05) which will sit this gateway
  behind canary rollouts.
- 5–6 passes: usually missing per-tier quality measurement or the
  cost figure. Without those two the routing policy is not
  accountable.
- ≤ 4 passes: the gateway forwards requests but cannot justify its
  decisions with data. Send the candidate back to the measurement
  harness; the routing policy is the easy half.

## 5. Common mistakes

1. **Routing without observability.** The router decides silently;
   when quality regresses nobody can tell whether the classifier
   is wrong or the small model is wrong. The `routing` field in
   the response and per-tier labels on every metric are the
   minimum.
2. **Silent down-tier fallback.** On `tier-l` failure, serving a
   small-tier response masquerading as a large-tier response is
   worse than returning an error — users will not detect the
   regression until a quality audit catches it.
3. **Treating the router as "trained once".** Request
   distributions shift: a new product feature can double the share
   of complex requests overnight. The router — whether rule set or
   classifier — needs a scheduled re-evaluation and a drift alert.
4. **Classifier on GPU.** A 7B classifier on a GPU per request
   eliminates the cost advantage of routing. The router model must
   be small and CPU-servable (~100–300M params) or rule-based.
5. **Measuring cost by headline price, not realized spend.**
   Spot/discount/commit pricing shifts realized tier costs. The
   `TIER_COST_PER_1K_TOK` map should be refreshed from billing
   data periodically, not pinned to the provider price sheet.
6. **Quality evaluated on the wrong slice.** Scoring the small
   tier against the hard prompts the router would never send it
   produces a gloomy and irrelevant number. Score each tier on the
   traffic the policy actually assigns it.
7. **No bucketing for prompts the router is uncertain about.** A
   classifier confidence around the threshold is where routing
   errors concentrate. Logging these borderline cases is how the
   next version of the policy gets written.
8. **Fallback that masks capacity planning failures.** If S → L
   fallback fires constantly because `tier-s` is under-scaled, the
   realized cost mix is "large tier for everyone". Alert on
   `rate(router_requests_total{outcome="error"}[5m]) /
   rate(router_requests_total{}[5m])` above a small threshold.

## 6. References

- vLLM project documentation — per-tier serving configuration,
  tensor parallelism, OpenAI-compatible API used by the gateway:
  https://docs.vllm.ai/
- Hugging Face Transformers documentation — small classifier
  models (DeBERTa, DistilBERT) suitable for a CPU-servable router:
  https://huggingface.co/docs/transformers/
- Prometheus client library documentation — Counters/Histograms
  used by the gateway for per-tier cost and latency metrics:
  https://prometheus.io/docs/
- NIST AI Risk Management Framework — "Measure" and "Govern"
  functions framing the need to continuously evaluate routed-down
  tiers against quality thresholds, not just the headline tier:
  https://www.nist.gov/itl/ai-risk-management-framework
- OWASP Machine Learning Security Top 10 — considerations for
  model-selection-as-a-security-surface (e.g. an attacker crafting
  prompts that force expensive tiers for denial-of-wallet
  attacks):
  https://owasp.org/www-project-machine-learning-security-top-10/
- Local module context —
  [`modules/mod-007-production-deployment/SOLUTION.md`](../SOLUTION.md),
  Decision 4 ("Multi-tier routing — model selection as a deployment
  concern") and the "Multi-tier routing without monitoring per
  tier" anti-pattern.
- Local stub reference —
  [`modules/mod-007-production-deployment/exercise-04/README.md`](./README.md)
  points at the engineer-track multi-tier routing reference whose
  measured savings figure (~60% on a specific workload) frames the
  "reproduce on your traffic" ask here.
