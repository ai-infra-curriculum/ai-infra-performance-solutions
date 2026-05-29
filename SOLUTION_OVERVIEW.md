# SOLUTION_OVERVIEW — Performance Engineering Track

> This file explains the design philosophy and reading philosophy
> across the performance-solutions module set. Per-exercise solutions
> live under `modules/<mod-NNN>/`. For the content index see
> [`SOLUTIONS_INDEX.md`](./SOLUTIONS_INDEX.md) (if present) or the
> module READMEs.

## What this track is teaching

Performance engineering is **measurement-driven**, not
**intuition-driven**. The single most important property of every
solution in this track is that the answer to "why is it fast?" or
"why is it slow?" can be *measured*, not guessed.

Concretely, this track teaches you to:

1. **Reason about the hardware** — GPU architecture, memory hierarchy,
   bandwidth vs. compute ridge points.
2. **Profile honestly** — Nsight, PyTorch Profiler, DCGM, and how to
   read them without fooling yourself.
3. **Optimize the right layer** — kernel-level changes when warranted,
   framework-level changes when they suffice.
4. **Quantify trade-offs** — every "optimization" has a cost
   (engineering time, code complexity, model quality). The solutions
   surface that cost.

## How the modules relate

| Module | Role in the track |
|---|---|
| `mod-001-gpu-fundamentals` | The *vocabulary*. Every later module assumes you know it. |
| `mod-002-cuda-programming` | The *low level*. Most platform engineers won't write CUDA daily, but reading it is non-negotiable. |
| `mod-003-performance-profiling` | The *measurement tools*. Without these, the rest is theatre. |
| `mod-004-transformer-optimization` | Domain depth: flash-attention, KV-cache, quantization. |
| `mod-005-model-compression` | Quality/size/speed trade-offs explicitly. |
| `mod-006-distributed-inference` | Tensor/pipeline parallelism for >1-GPU serving. |
| `mod-007-production-deployment` | Where the lab-grade optimizations meet real traffic. |
| `mod-008-advanced-topics` | Frontier patterns (speculative decoding, etc.). |

Read in module order. Skipping `mod-001` and `mod-003` and going to
`mod-004` is the most common mistake — you end up applying
optimizations you can't measure to a model you don't understand.

## Cross-cutting principles every solution observes

### Measure before you optimize

Every solution that claims a speedup includes the *baseline* number
and the *target* number. A solution that says "this is faster"
without a number is incomplete.

### Optimize the dominant kernel, not the obvious one

Profiling almost always shows that 80% of time is in 20% of the
operations. Optimizations that don't target the dominant kernel
produce measurable but tiny wins.

### Verify model quality after every change

Quantization, pruning, and operator substitutions can silently
degrade model quality. Every optimization solution includes a
quality validation step.

### Hardware specificity is a feature, not a bug

A solution that runs on every GPU equally well is usually leaving
performance on the table. Solutions identify which hardware they're
targeting and what they assume about it.

## Where the projects live

The *project specifications* (overview, requirements, architecture,
step-by-step build guide, rubric) live in
[`ai-infra-performance-learning/projects/`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects).
Each project's optimization narrative pulls from multiple modules
here.

Project-layer **solution rationale** (the reviewer's companion to each
spec — design choices, validation steps, rubric mapping, common
mistakes) is published under [`projects/`](./projects/) in this repo
as it is written. See [`projects/README.md`](./projects/README.md) for
the per-project index.

## Production gap checklist (track-wide)

A reader who has worked through every module still needs the
following to operate performance engineering in production:

- [ ] Continuous regression testing for performance, not just
      correctness
- [ ] Per-workload SKU selection process tied to cost data
- [ ] A profiling budget per release (someone owns the time)
- [ ] Hardware roadmap awareness (Hopper → Blackwell → Rubin
      transitions)
- [ ] Cost-per-token / cost-per-prediction observability surfaced
      to product teams
- [ ] On-call rotation that includes performance-regression triage

## Cross-references

| Topic | Where to go |
|---|---|
| GPU fundamentals lecture | `performance-learning/modules/mod-001-gpu-fundamentals/` |
| LLM-specific serving optimizations | `engineer-solutions/mod-110` |
| Distributed training fundamentals | `senior-engineer-solutions/projects/project-201-distributed-training/SOLUTION.md` |
| Architecture-level cost / capacity reasoning | `architect-solutions/projects/project-301-enterprise-mlops/SOLUTION.md` |

## Time budget for the track

- **Surveyor read**: 1 week (read every module README, skim solutions).
- **Practitioner read**: 2–3 months (work every exercise on real
  hardware, measure your numbers, compare to the reference).
- **Expert read**: 6+ months. Performance engineering takes
  reps; the modules are signposts.
