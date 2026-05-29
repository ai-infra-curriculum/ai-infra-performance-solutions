# SOLUTION — Project 01: Production Model Optimization Pipeline

> Read this *after* you have walked the learning project's
> `requirements.md`, `architecture.md`, and `STEP_BY_STEP.md`. This
> document is the reviewer's companion. It explains *why* the project
> is shaped the way it is, what passing work actually looks like, and
> where graders should expect to push back.
>
> The full project spec, including hard gates PR-1..PR-10 and rubric
> dimensions D1..D8, lives in the paired learning repository under
> [`projects/project-01-model-optimization`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects/project-01-model-optimization).
> Numbers in this file are taken from that spec; do not invent new
> targets here.

## 1. Solution overview

The project is a **single end-to-end optimization pipeline** for two
canonical models — ResNet-50 (vision) and BERT-base (NLP) — that takes
each model from an unmodified FP32 PyTorch checkpoint to a final
TensorRT 10 INT8 engine via FP16, PTQ, QAT, structured pruning, and
distillation paths. Every variant is benchmarked on the same hardware
with a single reproducible CLI (`make all`), every artifact is
sha256-signed in `compression_manifest.yaml`, and every accuracy /
latency gate fails the build rather than landing silently.

There is intentionally **no single "answer"**. The project tests
whether the candidate can compose techniques covered separately in
mod-003, mod-004, mod-005, and mod-008 into a pipeline that an SRE
would actually deploy, with the trade-offs measured rather than
asserted.

### What a passing submission looks like

A reviewer reading the deliverables should be able to answer, in under
one minute and *from the artifacts only*:

1. Does the final TRT engine hit **PR-1 (>= 3.0x bs=1 latency
   speedup)**, **PR-3 (>= 3.5x bs=32 throughput speedup)**, and
   **PR-6 (<= 0.25x model size on disk)**? — From `reports/benchmark_summary.md`.
2. Did **PR-4 / PR-5 (<= 2.0 pp accuracy drop on IN-1k top-1 and
   MNLI-m)** hold? — From `reports/accuracy_summary.md`.
3. Was the bench statistically defensible (**PR-8: std-dev / P50
   <= 5%**, GPU clocks locked, no thermal throttling)? — From the
   raw JSONL under `reports/raw/` plus the `nvidia-smi` clock-state
   snapshot in the manifest.
4. Is the win attributable to specific layers in the optimization
   stack (precision, fusion, layout) and not a measurement artifact? —
   From `reports/roofline_*.png` plus `profiles/*.ncu-rep`.
5. Will `make verify` reproduce the numbers within 5%? — From the
   `make verify` log in the manifest.

If any of those five answers requires reading source code, the
deliverables have failed the **D7 (profiling depth)** and **D5 (code
quality)** rubric dimensions.

### How the project layers onto the module solutions

The project is intentionally a *composition* exercise, not a *new
technique* exercise. Each pipeline stage maps to an existing module
solution that the candidate has already worked:

| Stage in the pipeline                     | Where the technique was taught                                       |
|-------------------------------------------|----------------------------------------------------------------------|
| Baseline benchmark (CUDA events, warmup)  | `mod-003-performance-profiling/exercise-03-pytorch-profiler`         |
| Nsight Systems trace                      | `mod-003-performance-profiling/exercise-01-nsys-trace`               |
| Nsight Compute hottest kernel             | `mod-003-performance-profiling/exercise-02-ncu-deep-dive`            |
| Roofline analysis                         | `mod-001-gpu-fundamentals/exercise-04-roofline-analysis`             |
| FP16 (`model.half()` + autocast)          | `mod-004-transformer-optimization/exercise-02-torch-compile`         |
| INT8 PTQ + calibrator selection           | `mod-005-model-compression/exercise-02-int8-static-quantization`     |
| INT8 QAT (FX graph mode)                  | `mod-005-model-compression/exercise-02-int8-static-quantization`     |
| Structured channel pruning                | `mod-005-model-compression/exercise-03-2-4-sparsity` (related)       |
| Knowledge distillation                    | `mod-005-model-compression/exercise-05-distillation`                 |
| TensorRT 10 engine build                  | `mod-007-production-deployment/exercise-01-pick-a-framework` (TRT pick) |
| End-to-end bench-and-report               | `mod-003-performance-profiling/exercise-05-end-to-end-optimization`  |

The candidate is not expected to re-derive any of these techniques.
The point of the project is to wire them together with **gates between
stages** and a **single artifact manifest** at the end.

## 2. Worked answer / implementation walkthrough

The implementation is laid out phase by phase in the learning repo's
[`STEP_BY_STEP.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/STEP_BY_STEP.md).
That document is canonical for "do this exactly." This section calls
out the **non-obvious design choices** and the **why** behind each one.

### 2.1 The bench runner is the most important class in the project

Every gate in the rubric depends on this. The reference implementation
is the `BenchRunner` shown in `STEP_BY_STEP.md` § 1.2: CUDA events
(not `time.perf_counter`), ≥ 50 warmup iterations, ≥ 500 measured
iterations, and a hard refusal to report numbers when
`std_ms / p50_ms >= 5%`. If a submission's runner does not assert this
invariant, every downstream number is suspect and the entire submission
fails **D2/D3 (throughput/latency)** by definition.

The non-obvious choice: the runner samples
`nvidia-smi --query-gpu=clocks_throttle_reasons.active` *between*
measured iterations, not just before. Thermal throttling is a
mid-run event on long sweeps. A pre-run check that passes and a post-run
check that passes can still hide a 10% derate in the middle of the run.

### 2.2 The PTQ → QAT → TRT handoff is JSON, not Python objects

The reference solution serializes per-tensor calibration ranges to
`calibration_<variant>.json` (architecture.md § 3.2). This is the
clean cut between the "PyTorch world" (where observers see activations)
and the "TRT world" (where `IInt8EntropyCalibrator2` consumes them).

Why JSON and not, e.g., pickle or a shared in-memory object?

1. **Reproducibility**: the calibration ranges become a hashable input
   to `build_inputs_sha256` (PR-10 determinism gate).
2. **Debuggability**: graders can `grep` the file. A range with `min`
   ≈ `max` is a dead channel; a range with `max` >> percentile-99 is
   an outlier the candidate's calibrator missed.
3. **Cross-stage reuse**: the same JSON feeds both the PT INT8 path
   (PTQ sanity check) and the TRT calibrator. Two paths, one source
   of truth.

Submissions that bypass this and call into `torch.ao.quantization`
APIs directly from the TRT builder lose **D5 (code quality)** points
and are fragile to PyTorch quantization API churn.

### 2.3 Per-channel weights, per-tensor activations

This is the TensorRT INT8 convention (architecture.md § 6.2-6.3) and
the reference solution follows it without deviation:

- Weights: per-channel symmetric (zero-point = 0). Modern hardware
  pays no cost for the per-channel scale.
- Activations: per-tensor asymmetric (zero-point ≠ 0). Post-ReLU
  distributions have a hard floor at zero; symmetric quantization
  wastes half the dynamic range.

A submission that uses per-tensor for weights "to save metadata" is
chasing an optimization that does not exist on A100/H100 INT8 tensor
cores and will tank accuracy on the layers with wide channel-wise
range. Flag this in review.

### 2.4 The dependency-aware pruner is non-negotiable

`STEP_BY_STEP.md` § 3.3 gives the FX-trace-based dependency graph.
The reference solution builds **coupled groups** — sets of layers
whose channel dimensions must shrink together (a conv's output, the
downstream conv's input, the BN affine params between them). Pruning
one without the others produces silent shape mismatches at inference,
which is rubric anti-pattern -2 on **D1** and **D5**.

The subtlety the candidate often misses: pruning the **classifier head**.
Don't. ResNet-50's `fc` layer is sized to 1000 classes; reducing its
input dim breaks the model. Add it to a no-prune list.

### 2.5 BN folding has a unit test, not a comment

`STEP_BY_STEP.md` § 3.2 lists the BN-folding sanity test. The
reference solution treats this as a hard CI gate. The reason is
historical: missed BN folds are the single most common cause of QAT
accuracy collapse, and the bug manifests *after* training — the model
trains cleanly, then loses 5-10 pp at eval time because the folded
weights stop matching the unfolded forward.

Submissions without an explicit
`test_bn_folding_preserves_output()`-style test lose D1 reviewer
confidence. The reference test asserts `torch.allclose(y_before,
y_after, atol=1e-5)` on a held-out batch.

### 2.6 Distillation uses both KL and CE, with an alpha sweep

The reference solution implements the loss shown in `STEP_BY_STEP.md`
§ 4.1: `α * KL(student/T || teacher/T) * T² + (1-α) * CE(student, labels)`
with `T=4.0`. The alpha sweep across `{0.3, 0.5, 0.7, 0.9}` is in
`reports/distillation_alpha_sweep.md`.

The intermediate-layer **hint loss** (FitNets-style, `STEP_BY_STEP.md`
§ 4.2) is included even though it adds a learnable projection. Without
it, distillation on BERT collapses for tasks that depend on
mid-network representations rather than just final logits.

A "distillation done by matching final logits only" works on
ImageNet-style classification but loses on MNLI. The reference
distilled BERT student is therefore evaluated on **MNLI-m, not
perplexity** (cf. mod-005 SOLUTION.md, common mistake 6).

### 2.7 The TRT builder enables `OBEY_PRECISION_CONSTRAINTS` in CI

`STEP_BY_STEP.md` § 5.4 ("Gotchas") explains the failure mode: a TRT
builder with `kFP16` set but no `OBEY_PRECISION_CONSTRAINTS` will
silently fall back to FP32 on any layer the builder can't satisfy at
the requested precision. The latency number then looks fine on a small
test, then *regresses* in production once a different shape hits the
fallback path.

The reference CI build uses `OBEY_PRECISION_CONSTRAINTS=True` so the
builder errors loudly. Production builds may relax this if and only if
the fallback layers are listed in `configs/trt/<variant>_overrides.yaml`
with measured deltas.

### 2.8 The timing cache is committed but versioned

TRT timing caches are tactic-timing memos. They are hardware-specific
(an A100 cache will not help on an L4) but they are also large and
binary, so they are easy to mis-commit. The reference solution:

- Writes the cache to `engines/timing_cache.bin` on every build.
- Only commits it to source control behind a `--commit-timing-cache`
  flag (architecture.md § 8) so noisy timing diffs don't pollute PRs.
- Records the cache's sha256 in `compression_manifest.yaml` so a
  reviewer can confirm whether the second build was a tactic-cache
  hit or a cold rebuild.

This is what unlocks the "second build under 60s" check in
`STEP_BY_STEP.md` Gate 5.

### 2.9 The manifest is signed, not just listed

`compression_manifest.yaml` (deliverables README § 2) includes a
sha256 per artifact, plus `build_inputs_sha256` = hash of the
`(state_dict, calibration data, config)` triple. Two runs with the
same inputs MUST produce the same `build_inputs_sha256`. This is the
**reproducibility audit trail** that turns "I got 3.4x speedup" from a
claim into a verifiable measurement.

Submissions that ship a manifest without `build_inputs_sha256` lose
D6 (reproducibility) above level 3.

### 2.10 Quarantine, don't silently drop

When a variant misses an accuracy gate (architecture.md § 7), the
reference pipeline writes the partial artifacts to
`engines/quarantine/<variant>/` and records `quarantined: true` in the
manifest. Exit code is `2` (distinct from `1` = setup error). The
build does **not** mark itself green.

Submissions that silently drop failing variants are caught by D1
anti-pattern ("no quarantine flow; failed variants silently dropped",
-1 on D1 and D5).

## 3. Validation steps

The reviewer runs, in order:

```bash
# 0. Inside the candidate's Dockerfile, on the target hardware:
docker build -t opt-pipe:review .
docker run --gpus all --rm -v "$PWD":/work opt-pipe:review nvidia-smi
```

Confirm: driver >= 545, CUDA >= 12.4, Nsight >= 2024.2, and the GPU
matches the manifest's `hardware:` field.

```bash
# 1. Reproduce the full sweep.
docker run --gpus all --rm -v "$PWD":/work opt-pipe:review make all
```

Expected: exit 0, wall clock <= 90 minutes on A100 (PR-7),
`compression_manifest.yaml` present, every artifact in deliverables
README § 1 emitted.

```bash
# 2. Reproduce the numbers.
docker run --gpus all --rm -v "$PWD":/work opt-pipe:review make verify
```

Expected: every benchmark number within 5% of the value recorded in
the manifest. If the reviewer is on different hardware, the candidate
should support `make verify HARDWARE=<sku>` with a 15% comparison
window (deliverables README § 6).

```bash
# 3. Spot-check the gates.
python -m cli.report --check-gates
```

Reads the manifest, asserts each PR-1..PR-7 row in
`reports/benchmark_summary.md` is marked `pass`. Exit non-zero on any
fail.

```bash
# 4. Spot-check that the FP16 path is actually FP16.
nsys stats --report gpukernsum profiles/nsys_resnet50_fp16.nsys-rep \
    | grep -iE "sgemm|fp32"
```

Expected: empty output, or only kernels in a documented no-fold list.
A non-empty grep is a silent FP32 fallback (rubric anti-pattern,
-1 D1 / -1 D7).

```bash
# 5. Re-check accuracy on the canonical eval set.
python -m cli.optimize model=resnet50 variant=trt_int8 \
    hardware=$(nvidia-smi -L | head -1 | awk '{print $5}') \
    eval=imagenet_val_full
python -m cli.optimize model=bert_base variant=trt_int8 \
    hardware=$(nvidia-smi -L | head -1 | awk '{print $5}') \
    eval=mnli_dev
```

Expected: deltas within 2.0 pp of FP32 baseline (PR-4, PR-5).

```bash
# 6. Confirm clock-state honesty.
grep -E "throttle|clock" reports/raw/*.jsonl | head
```

Expected: every measured iteration sampled `clocks_throttle_reasons`
and recorded the active mask. Any non-`0x0` mask should appear in the
manifest's `clock_state_anomalies` field.

```bash
# 7. Inspect the manifest signature.
python -m cli.report --verify-manifest
```

Expected: every artifact's recorded sha256 matches the on-disk file
sha256. Exit non-zero on any mismatch (a tampered or stale manifest).

## 4. Rubric / review checklist

The full rubric is the learning repo's
[`rubric.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/rubric.md).
This section is the **reviewer's quick read** mapped to artifacts.

### 4.1 Hard gates (fail the project if any miss)

| Gate  | Source artifact                          | Pass criterion                                    |
|-------|------------------------------------------|---------------------------------------------------|
| PR-1  | `reports/benchmark_summary.md` (bs=1 ResNet-50 row)   | speedup >= 3.0x                       |
| PR-2  | `reports/benchmark_summary.md` (bs=1 BERT-base row)   | speedup >= 3.0x                       |
| PR-3  | `reports/benchmark_summary.md` (bs=32 ResNet-50 row)  | throughput speedup >= 3.5x            |
| PR-4  | `reports/accuracy_summary.md`            | ResNet-50 top-1 drop <= 2.0 pp                    |
| PR-5  | `reports/accuracy_summary.md`            | BERT-base MNLI-m drop <= 2.0 pp                   |
| PR-6  | `reports/benchmark_summary.md` (size column) | final engine size <= 0.25x FP32 baseline      |
| PR-7  | `make all` log                           | wall clock <= 90 minutes on A100                  |
| PR-8  | `reports/raw/*.jsonl`                    | std-dev / P50 <= 5% on every measured iteration   |
| -     | `make verify` log                        | reproduces every number within 5%                 |
| -     | `compression_manifest.yaml`              | every artifact sha256 verifies, `build_inputs_sha256` present |

### 4.2 Rubric dimensions (D1–D8)

For each dimension below: artifact to read, the level-3 (Pass) bar,
and the level-5 (Reference-quality) bar.

| D | Dimension              | Read this                                                   | L3 (Pass)                                          | L5 (Reference)                                                       |
|---|------------------------|-------------------------------------------------------------|----------------------------------------------------|----------------------------------------------------------------------|
| 1 | Correctness            | `reports/accuracy_summary.md`, `engines/quarantine/`        | All accuracy gates met; quarantine flow works     | Calibrator comparison (entropy/percentile/MSE) per model with ablation numbers |
| 2 | Throughput (bs=32)     | `reports/benchmark_summary.md` bs=32 ResNet-50 row          | 2.5–3.5x (meets PR-3)                              | > 4.5x with breakdown of fusion / precision / layout contributions   |
| 3 | Latency (bs=1)         | `reports/benchmark_summary.md` bs=1 worst-of-two row        | 3.0x (meets PR-1/PR-2)                             | > 4x with CUDA Graphs + tactic-cache reuse + cold/warm comparison    |
| 4 | Memory                 | `reports/memory_breakdown.md`                               | Size <= 0.25x; peak mem documented (PR-6)         | Nsight Compute Memory Workload Analysis working-set chart + activation recompute trade-off |
| 5 | Code quality           | `src/` tree, `mypy --strict` + `ruff` + `black` logs        | Module layout matches `architecture.md`; lints pass | Reviewer can extend with a new stage in < 30 minutes via the `Stage` protocol |
| 6 | Reproducibility        | `make all` log, `compression_manifest.yaml`                 | `make all` inside Docker reproduces (PR-12)       | `build_inputs_sha256` plus CI reproducibility test                  |
| 7 | Profiling depth        | `reports/roofline_*.png`, `profiles/*.ncu-rep`              | One Nsight Compute report per variant + roofline classified as memory/compute bound | Roofline shows the kernel walking up the ridge after each compression stage, with the *why* explained |
| 8 | Engineering judgment   | `docs/adr/*.md`, `reports/benchmark_summary.md` narrative   | PTQ-vs-QAT chosen per model with stated reason    | Stretch path (FP8 / AWQ / CUDA Graphs) shipped with measured numbers and "when to use" guidance |

Pass = all hard gates met AND >= 3/5 on every dimension. Distinction
= >= 4/5 on at least 6 of 8.

### 4.3 Anti-patterns (auto-deduct)

These come directly from the learning repo's `rubric.md` § 4 and the
project's "common mistakes" catalog. A reviewer should grep for them
explicitly.

- **Hidden FP32 fallback in "FP16" path** (`nsys stats | grep sgemm`
  non-empty): -1 D1 and -1 D7.
- **Python `time.perf_counter` for GPU timing** (grep `src/bench/`
  for `perf_counter`): -1 D2 and -1 D3.
- **Warmup < 50 iterations** (grep `src/bench/` for `warmup=`):
  -1 D2 and -1 D3.
- **Calibration data leaks from train into eval** (read
  `pipeline.calibrator` and confirm `train` / `val` splits are
  disjoint): -2 D1 (correctness violation).
- **Pruning without dependency awareness** leading to a shape
  mismatch at inference: -2 D1 and -2 D5.
- **`make all` requires manual intervention**: -2 D6.
- **No quarantine flow; failed variants silently dropped**: -1 D1
  and -1 D5.

### 4.4 Stretch / distinction bonuses

These are scored but don't count toward Pass:

- **B1**: NVIDIA TransformerEngine FP8 path on H100 with measured
  speedup (see mod-008 ex-05 for the underlying technique).
- **B2**: AWQ 4-bit ablation on BERT-base with accuracy and speedup
  numbers (see mod-005 ex-01).
- **B3**: CUDA Graphs capture on the bs=1 path with cold/warm
  latency comparison (see mod-008 ex-01).
- **B4**: 2:4 structured sparsity ablation on Ampere/Hopper (see
  mod-005 ex-03).
- **B5**: Streamlit (or equivalent) dashboard reading
  `benchmark_*.json` so a PM can see the wins without reading code.

## 5. Common mistakes

These are the failure modes the reference graders see repeatedly.
They're grouped by phase so a reviewer can locate the cause quickly.

### 5.1 Baseline (Week 1)

- **Using `time.perf_counter` instead of CUDA events**. The candidate
  measures launch latency, not execution time. The downstream speedup
  numbers are then meaningless. Mandatory CUDA events; see
  `STEP_BY_STEP.md` § 1.4 "Gotchas".
- **Skipping warmup**. The first 30-50 iterations include cuDNN
  algorithm selection. Reporting those as "real" inflates baseline
  latency, which *flatters* every later speedup.
- **GPU clocks unlocked**. Boost behavior is non-deterministic; the
  same workload can vary ± 15% run to run. Lock with `nvidia-smi
  --lock-gpu-clocks` (or document the SKU deviation in the manifest).

### 5.2 FP16 (Week 2)

- **Silent FP32 fallback in attention**. Common BERT failure mode:
  the candidate calls `.half()` but a `.float()` cast leaks into
  attention probabilities for "numerical stability." Now the hottest
  kernel is `sgemm`, FP16 speedup is 1.1x not 2.0x, and nobody noticed.
  Catch with `nsys stats --report gpukernsum | grep sgemm`.
- **Mixing pure half and autocast in the same run**. They have
  different fallback rules; the candidate ends up with neither
  cleanly. Pick one per variant.

### 5.3 INT8 PTQ (Week 2)

- **Class-imbalanced calibration set**. A 1024-sample subset that
  misses rare classes will produce skewed activation ranges and silent
  accuracy regressions on those classes. Stratify the calibration
  loader. (`STEP_BY_STEP.md` § 2.4.)
- **Histogram observer bin count too low**. Default 2048 bins is
  fine; < 512 underestimates the tail and clips. Don't override
  unless you've measured.
- **Reporting "PTQ worked!" without the sensitivity sweep**. A 1.4 pp
  drop might be one bad layer carrying 1.0 pp of it. The
  sensitivity CSV (FR-4 + STEP_BY_STEP § 2.4) is what lets the
  candidate keep that one layer in FP16 and ship.

### 5.4 QAT (Week 3)

- **Missed BN folding**. Single biggest QAT bug. Symptom: model
  trains cleanly, then loses 5-10 pp at eval. Fix with the BN-folding
  unit test (STEP_BY_STEP § 3.2). If the test isn't there, the rubric
  is correct to fail D1.
- **QAT learning rate too high**. The default LR is for FP32
  training; for QAT fine-tune use 10% of that (cosine schedule). A
  candidate using full LR will see oscillating fake-quant scales and
  unstable training.
- **NaNs under FP16 mixed precision during QAT**. Train in BF16 (or
  pure FP32) instead; inference still goes through INT8.

### 5.5 Pruning (Week 3)

- **Channel mismatch at inference**. Dependency graph not built or
  not walked. Causes a `RuntimeError: size mismatch` after the first
  pruning round. The reference dep-graph (architecture.md § 3.6 +
  STEP_BY_STEP § 3.3) groups conv-BN-conv triples so the slice is
  atomic.
- **Pruning the classifier head**. ResNet-50's `fc` is sized for 1000
  classes. Reducing input dim breaks the model. Add to a no-prune list.
- **Pruning schedule too aggressive**. 30% in one shot loses 3-5 pp.
  Iterative schedule (10% → fine-tune → 20% → fine-tune → 30%) is
  what recovers it. Document the per-round accuracy.

### 5.6 Distillation (Week 4)

- **Mode collapse**. Student matches teacher on average but fails on
  some classes entirely. Detect with per-class recall against the
  teacher; fail if any class drops below 50% relative recall.
- **Distilling on the wrong eval**. Perplexity is a *calibration*
  target, not an *evaluation* target. A distilled BERT student that
  matches teacher perplexity can still drop 4 pp on MNLI. Evaluate
  on the *production-relevant* eval, not the proxy.
- **Skipping the intermediate hint loss on BERT**. Final-logit-only
  distillation transfers shallow knowledge; mid-layer hints transfer
  representation structure. Without hints, the BERT student
  underperforms on inference-heavy tasks (mod-005 SOLUTION.md, common
  mistake 6 is the same observation at the module level).

### 5.7 TensorRT (Week 5)

- **No `OBEY_PRECISION_CONSTRAINTS` in CI**. Silent FP32 fallback at
  build time. Latency looks great until production shape hits the
  fallback path. (STEP_BY_STEP § 5.4 "Gotchas".)
- **Workspace too small**. Under 256 MB you lose tactic options; the
  builder picks slower kernels. Use >= 1 GB on A100/H100 for these
  models.
- **Stale calibration cache**. TRT silently rebuilds when the cache
  hash doesn't match the network signature. The candidate sees the
  warning, ignores it, and ships a build that didn't actually use
  the cached INT8 scales. Delete on any network-graph change.
- **Reusing one engine across batch sizes**. Per-batch-size engines
  (or a single engine with optimization profiles covering 1, 8, 32)
  is the working setup; using a bs=1 engine at bs=32 is a 2-3x
  throughput hit.

### 5.8 Benchmark / report (Week 6)

- **No clock-state record in the report**. Reviewers can't tell
  whether the bench was honest. Always record
  `clocks_throttle_reasons.active` per iteration and surface the
  rollup in the manifest.
- **Roofline plotted as a generic illustration**. The point is to
  show each kernel's actual `(arithmetic_intensity, achieved_GFLOPs)`
  point against the hardware ridge, and to explain *why* each
  compression stage moved the kernel. A blank roofline scatter with
  no commentary is L2, not L3, on D7.
- **Manifest missing `build_inputs_sha256`**. Without it, the
  determinism claim in D6 is unverifiable; the rubric caps at L3.

### 5.9 Cross-cutting

- **Claiming a stretch result without a measurement**. "Could use
  AWQ" is not a B2; AWQ shipped with `delta_pp` and `speedup_vs_fp32`
  in the manifest is. Negative results are *fine* and rewarded; hand-
  wavy results are not.
- **CPU vs GPU mismatch**. A candidate reports CPU-side data-loader
  time as part of inference latency. Strip the data-loader cost from
  the bench window — it's not what PR-1/PR-2 measure.

## 6. References

### Project artifacts (paired learning repo)

- [`projects/project-01-model-optimization/README.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/README.md) — high-level overview, learning outcomes, success criteria.
- [`projects/project-01-model-optimization/requirements.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/requirements.md) — functional (FR-1..FR-12), performance (PR-1..PR-10), and non-functional (NFR-1..NFR-6) requirements.
- [`projects/project-01-model-optimization/architecture.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/architecture.md) — component diagram, data flow, trade-off discussion.
- [`projects/project-01-model-optimization/STEP_BY_STEP.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/STEP_BY_STEP.md) — phase-by-phase build guide with code snippets.
- [`projects/project-01-model-optimization/rubric.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/rubric.md) — full scoring rubric (D1..D8 + bonuses + anti-patterns).
- [`projects/project-01-model-optimization/deliverables/README.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/deliverables/README.md) — submission inventory and manifest schema.

### Related module solutions (this repo)

- [`modules/mod-001-gpu-fundamentals/SOLUTION.md`](../../modules/mod-001-gpu-fundamentals/SOLUTION.md) — roofline and the vocabulary every speedup claim depends on.
- [`modules/mod-003-performance-profiling/SOLUTION.md`](../../modules/mod-003-performance-profiling/SOLUTION.md) — Nsight Systems / Nsight Compute / PyTorch Profiler discipline.
- [`modules/mod-004-transformer-optimization/SOLUTION.md`](../../modules/mod-004-transformer-optimization/SOLUTION.md) — FlashAttention, torch.compile, KV-cache decisions that feed the BERT path.
- [`modules/mod-005-model-compression/SOLUTION.md`](../../modules/mod-005-model-compression/SOLUTION.md) — quantization / pruning / distillation rationale; the compression toolkit this project composes.
- [`modules/mod-007-production-deployment/SOLUTION.md`](../../modules/mod-007-production-deployment/SOLUTION.md) — framework selection (TRT vs ORT vs vLLM) discussion.
- [`modules/mod-008-advanced-topics/SOLUTION.md`](../../modules/mod-008-advanced-topics/SOLUTION.md) — CUDA Graphs, FP8, NCCL — sources for the distinction bonuses B1, B3.
- [`SOLUTION_OVERVIEW.md`](../../SOLUTION_OVERVIEW.md) — track-wide design philosophy ("measure before you optimize", "verify model quality after every change", "hardware specificity is a feature").

### Official standards and primary sources

These are the authoritative documents the reference solution defers
to. Citations are by document family — pin the specific version that
matches the repo's `requirements.md` dependency table when reviewing a
particular submission.

- **NVIDIA TensorRT Developer Guide** (v10.x) — the source of truth
  for `BuilderConfig`, precision flags, `IInt8EntropyCalibrator2`,
  timing-cache semantics, and `OBEY_PRECISION_CONSTRAINTS` behavior.
- **NVIDIA Nsight Systems and Nsight Compute User Guides** —
  authoritative for `--trace`, `--launch-skip`, `--launch-count`, and
  the kernel-metric names (`dram__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__warps_active.avg.pct_of_peak_sustained_active`).
- **NVIDIA A100 / H100 product briefs and datasheets** — for peak
  TFLOPs (FP16, INT8) and HBM bandwidth used in the roofline. (E.g.,
  A100 80GB SXM: 312 TOPS INT8 tensor-core peak, 2.039 TB/s HBM2e —
  used in `STEP_BY_STEP.md` § 6.4 as the roofline reference.)
- **PyTorch documentation**: `torch.ao.quantization` (PTQ, QAT,
  `prepare_qat_fx`, `QConfigMapping`), `torch.cuda.Event` (timing),
  `torch.onnx.export` (opset, dynamic axes, constant folding),
  `torch.fx` (symbolic trace used by the dependency-aware pruner).
- **ONNX**: opset 17 reference and `onnx.shape_inference` / `onnx.checker`
  contracts.
- **ONNX Runtime GPU** documentation — used as the cross-check
  inference path mentioned in `requirements.md` Section 7.

### Foundational papers cited in the project spec

These are the references in the learning repo's
[`README.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-01-model-optimization/README.md)
Section 14. Reviewers should expect the candidate to cite them in
`docs/adr/` when defending design choices.

- "A White Paper on Neural Network Quantization", Qualcomm AI
  Research, 2021 — calibration choices, symmetric vs asymmetric,
  per-tensor vs per-channel.
- Han et al., "Deep Compression", ICLR 2016 — original pruning +
  quantization + Huffman coding pipeline; the conceptual ancestor of
  the project.
- Hinton, Vinyals, Dean, "Distilling the Knowledge in a Neural
  Network", 2015 — temperature-scaled KL distillation loss used in
  Phase 4.
- Romero et al., "FitNets: Hints for Thin Deep Nets", ICLR 2015 —
  intermediate-layer hint loss used in Phase 4.
- Lin et al., "AWQ: Activation-aware Weight Quantization for LLM
  Compression and Acceleration", 2023 — distinction bonus B2.
- Xiao et al., "SmoothQuant: Accurate and Efficient Post-Training
  Quantization for Large Language Models", ICML 2023 — referenced in
  the learning-outcomes list (technique choice between PTQ, QAT,
  AWQ, GPTQ, SmoothQuant).

### Cross-track pointers

- `engineer-solutions/mod-110-llm-infrastructure/exercise-06-inference-optimization-llm`
  — the end-to-end LLM inference optimization chain that downstream
  project-03 builds on.
- `senior-engineer-solutions/projects/project-201-distributed-training/SOLUTION.md`
  — the multi-GPU training counterpart; this project deliberately
  stays single-GPU (requirements § 1.2 out-of-scope).
- `architect-solutions/projects/project-301-enterprise-mlops/SOLUTION.md`
  — architecture-level cost / capacity reasoning that consumes the
  compression manifest produced here.
