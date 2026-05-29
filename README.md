# AI Infrastructure Performance Engineer — Solutions Repository

> **Status**: ✅ **Published** — 8 modules with 40+ reference solutions live as of 2026-05. Content is AI-assisted and undergoing human review; treat as a learning reference and cross-check with primary sources (NVIDIA datasheets, PyTorch / TensorFlow docs, NCCL specs) before adopting patterns in production.

Reference implementations for the **AI Infrastructure Performance Engineer** specialization track ([ai-infra-performance-learning](https://github.com/ai-infra-curriculum/ai-infra-performance-learning)).

For the design philosophy across this module set, see [`SOLUTION_OVERVIEW.md`](./SOLUTION_OVERVIEW.md). For the authoritative content index, see the per-module READMEs under `modules/`.

## What's new — 2026-05-27

Module-level `SOLUTION.md` design-rationale docs for all 8 modules (`mod-001-gpu-fundamentals` through `mod-008-advanced-topics`). Each doc explains the GPU/perf-engineering "why" behind the reference implementations — the choice of CUDA primitives, memory-hierarchy decisions, inference framework selection — rather than re-walking the code in cross-referenced engineer/mod-107 + mod-110 exercises. Audit score: 51 → 63.

## What's in here

- **`modules/`** — Per-exercise solutions organized by module:
  - `mod-001-gpu-fundamentals` — GPU architecture vocabulary, roofline analysis.
  - `mod-002-cuda-programming` — CUDA kernel patterns.
  - `mod-003-performance-profiling` — Nsight, PyTorch Profiler, DCGM.
  - `mod-004-transformer-optimization` — flash-attention, KV-cache, quantization.
  - `mod-005-model-compression` — quality/size/speed trade-offs.
  - `mod-006-distributed-inference` — tensor/pipeline parallelism.
  - `mod-007-production-deployment` — where lab-grade optimizations meet real traffic.
  - `mod-008-advanced-topics` — frontier patterns (speculative decoding, etc.).
- **`guides/`** — Cross-cutting walkthroughs.
- **`resources/`** — Shared references.
- **[`SOLUTION_OVERVIEW.md`](./SOLUTION_OVERVIEW.md)** — Design philosophy across the track.

The full project specifications (overview, requirements, architecture, step-by-step, rubric) live in the [learning repo's `projects/`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects). Reviewer-facing solution rationale for each project is published under [`projects/`](./projects/) in this repo as it is written.

## How to use this repository

1. **Attempt the exercise yourself first** in the learning repo — solutions only help if you've struggled with the problem.
2. **Baseline before optimizing.** Every solution in this track includes a baseline number; if you can't reproduce the baseline, you can't measure the optimization.
3. **Read the profiling output**, not just the code. The optimization makes sense only against the profile.
4. **Verify model quality after every change.** Quantization, pruning, and operator substitutions can silently degrade quality.

## Prerequisites

- The [Engineer track](https://github.com/ai-infra-curriculum/ai-infra-engineer-learning) and ideally the [Senior Engineer track](https://github.com/ai-infra-curriculum/ai-infra-senior-engineer-learning).
- Comfort reading CUDA-adjacent code (you don't need to write it daily, but reading it is non-negotiable).
- A GPU-accessible environment for the hands-on exercises (cloud or local).

**Experience level**: Advanced (4–6 years engineering experience, with some prior exposure to GPU workloads).
**Time commitment**: 200–250 hours across the track.

## Learning objectives

The Performance track prepares you to:

- Reduce inference latency by 50%+ through measured optimizations.
- Improve GPU utilization from typical baseline (~40%) to production-acceptable levels (~85%+).
- Reduce infrastructure costs by 30–50% through efficiency gains, with the trade-offs quantified.
- Build performance regression testing so optimizations don't silently regress.
- Choose appropriate hardware (procurement) based on actual workload characteristics, not heuristics.

## Related repositories

- [ai-infra-performance-learning](https://github.com/ai-infra-curriculum/ai-infra-performance-learning) — companion learning materials with project-layer build-outs.
- [ai-infra-engineer-solutions](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions) — broader engineering depth.
- [ai-infra-senior-engineer-solutions](https://github.com/ai-infra-curriculum/ai-infra-senior-engineer-solutions) — distributed training reference (project-201).
- [ai-infra-architect-solutions](https://github.com/ai-infra-curriculum/ai-infra-architect-solutions) — architecture-level cost / capacity reasoning.

## Known limitations

- **Content is AI-assisted and partly under human review.** Verify against NVIDIA datasheets and current PyTorch / TensorFlow docs before quoting specific numbers (especially for newer hardware like Blackwell).
- **Hardware specificity matters.** A solution that wins on H100 may not win on A100 or Blackwell; each solution identifies its target hardware and assumptions.
- **Profiling output is the source of truth.** Numbers in the lecture notes are illustrative; reproduce on your hardware before relying on them.

## Contributing

Issues, corrections, and pull requests are welcome. See [`CONTRIBUTING.md`](./CONTRIBUTING.md). The most useful contributions:

- Updating baseline numbers as hardware ships (Blackwell B200, future generations).
- Adding regression-test infrastructure for the optimization solutions.
- Refining the profiling walkthroughs as PyTorch Profiler / Nsight evolve.

## License

See [`LICENSE`](./LICENSE).

---

**Last updated**: 2026-05-26
**Maintainer**: AI Infrastructure Curriculum Project
