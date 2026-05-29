# SOLUTION — Project 02: Custom CUDA Kernels for Transformer Optimization

> Read this *after* you have walked the learning project's
> `requirements.md`, `architecture.md`, and `STEP_BY_STEP.md`. This
> document is the reviewer's companion. It explains *why* the project
> is shaped the way it is, what passing work actually looks like, and
> where graders should expect to push back.
>
> The full project spec — hard gates, rubric dimensions, deliverable
> manifest, and the exact target metrics — lives in the paired learning
> repository under
> [`projects/project-02-gpu-optimization`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects/project-02-gpu-optimization).
> Numbers in this file are pulled from that spec; do not invent new
> targets here. Sections that name a specific gate ID, rubric
> dimension, or numeric threshold that the reviewer should expect from
> the spec are marked `<!-- spec-pin: ... -->` and should be reconciled
> with the learning repo before the reviewer-facing copy is treated as
> final.

## 1. Solution overview

The project is a **hand-written-kernel deep-dive** for the transformer
inference stack. The candidate takes the operators that dominate
modern LLM serving — fused attention, fused
LayerNorm/RMSNorm + residual, fused GEMM-epilogue (GEMM + bias + GELU
/ SwiGLU), and a fused KV-cache update — and replaces the stock
PyTorch / cuBLAS / cuDNN path with **custom CUDA and Triton kernels**
that:

1. **Are correct** — bitwise-comparable to the reference within a
   stated tolerance, with a numerical regression test that fails CI
   on drift.
2. **Are measurably faster** — every speedup claim is backed by an
   Nsight Compute roofline showing the kernel walked from
   memory-bound toward (or onto) the hardware ridge.
3. **Compose with PyTorch** — every kernel ships as a `torch.utils.cpp_extension`
   or `triton.jit` module with autograd integration where the op
   appears in training graphs, and stream-safe forward-only paths
   where it does not.
4. **Are honest about when they should not be used** — every kernel's
   README documents the shape window where the custom kernel beats
   the library, and the shape window where it loses. There is no
   "always faster" claim.

There is intentionally **no single "answer"**. The project tests
whether the candidate can move from *reading* CUDA (the bar set by
mod-002) to *writing and shipping* CUDA that survives contact with a
production transformer, with the measurement discipline established
in mod-003 and the operator catalog from mod-004.

### What a passing submission looks like

A reviewer reading the deliverables should be able to answer, in
under one minute and *from the artifacts only*:

1. Does each custom kernel **match the reference within the stated
   tolerance** (default fp16/bf16: `atol=1e-2`, `rtol=1e-2` against
   an fp32 reference; cf. PyTorch numerical guarantees for
   `torch.allclose`)? — From `tests/test_numerics.py` log and the
   `numerics_report.md` table.
2. Does each kernel **hit its speedup target** vs. the library
   baseline on the declared shape window? — From
   `reports/benchmark_summary.md`.
   <!-- spec-pin: the per-op speedup gates (PR-N) live in the
   learning repo's requirements.md / rubric.md and must be quoted
   verbatim here. -->
3. Was the bench **statistically defensible** (CUDA events, ≥ 50
   warmup iterations, ≥ 500 measured iterations,
   `std_ms / p50_ms <= 5%`, GPU clocks locked, no thermal throttling)?
   — From the raw JSONL under `reports/raw/` plus the `nvidia-smi`
   clock-state snapshot in the manifest. (This bench discipline is
   the same one set by project-01 § 1; the kernels change, the
   measurement contract does not.)
4. Is the win **attributable to a specific hardware lever** (shared
   memory, async copy via `cp.async`, tensor cores via WMMA / MMA,
   warp specialization, vectorized loads) and not a measurement
   artifact? — From `reports/roofline_*.png` plus
   `profiles/*.ncu-rep` annotated with the metric that moved.
5. Will `make verify` reproduce the numbers within 5% on the target
   SKU? — From the `make verify` log in the manifest.

If any of those five answers requires reading source code, the
deliverables have failed the profiling-depth and code-quality rubric
dimensions.
<!-- spec-pin: confirm the D-numbering of profiling-depth and
code-quality dimensions in the project-02 rubric.md. -->

### How the project layers onto the module solutions

The project is a *composition + extension* exercise. Each kernel
maps to a technique the candidate has already worked at the module
level:

| Operator delivered                                  | Where the underlying technique was taught                              |
|-----------------------------------------------------|------------------------------------------------------------------------|
| Vectorized memory access (`float4`, `__ldg`)        | `mod-002-cuda-programming/exercise-01` (vector-add)                    |
| Two-level tiled GEMM (block tile → warp tile)       | `mod-002-cuda-programming/exercise-02` (tiled matmul)                  |
| Shared-mem + warp-shuffle reductions (for softmax)  | `mod-002-cuda-programming/exercise-03` (reduction)                     |
| Occupancy / block-size sweeps                       | `mod-002-cuda-programming/exercise-04` (occupancy tuning)              |
| PyTorch C++ extension packaging + autograd          | `mod-002-cuda-programming/exercise-05` (pytorch-extension)             |
| FlashAttention tiling and online softmax            | `mod-004-transformer-optimization/exercise-01` (FlashAttention)        |
| Triton kernel authoring (`@triton.jit`, masks)      | `mod-004-transformer-optimization/exercise-05` (Triton LayerNorm)      |
| Nsight Compute kernel-level profiling               | `mod-003-performance-profiling` (entire module)                        |
| Roofline classification (compute- vs memory-bound)  | `mod-001-gpu-fundamentals/exercise-04-roofline-analysis`               |

The candidate is not expected to re-derive any of these techniques.
The point of the project is to **commit** to a kernel implementation,
**measure** it against the production library it replaces, and
**defend** the choice with profiling data.

## 2. Worked answer / implementation walkthrough

The phase-by-phase build is laid out in the learning repo's
[`STEP_BY_STEP.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-02-gpu-optimization/STEP_BY_STEP.md).
That document is canonical for "do this exactly." This section calls
out the **non-obvious design choices** and the **why** behind each
one.

### 2.1 The numerics test runs *before* the bench, not after

Every kernel exercise has the same three-stage gate: **build →
numerics → bench**. The numerics gate runs first because a kernel
that produces wrong outputs but happens to be fast is the worst
possible state for the downstream pipeline — it passes the bench
gate and then fails silently inside a longer integration test.

The reference numerics test (mirroring the discipline established
in `mod-002` ex-05) does three things:

1. Runs the candidate kernel and the reference op
   (`torch.nn.functional.scaled_dot_product_attention`,
   `torch.nn.LayerNorm`, `torch.addmm`, etc.) on the same input.
2. Asserts `torch.allclose(out_custom, out_ref, atol=1e-2,
   rtol=1e-2)` for fp16/bf16, `atol=1e-5` for fp32, against an fp32
   reference run.
3. Also asserts that the **per-element absolute error percentile**
   (99.9th percentile, not max) stays within the tolerance. A naive
   `allclose` lets a single outlier kill the test; the percentile
   variant lets the reviewer distinguish "one element with
   denormal-handling drift" (acceptable) from "the kernel is just
   wrong" (not).

Submissions that swap step (3) for "I checked `allclose` passes" lose
correctness points and should be flagged for a re-run with the
expanded test.
<!-- spec-pin: confirm the tolerance numbers above match the
project-02 spec; PyTorch's documented default `torch.allclose`
defaults are `atol=1e-8, rtol=1e-5` and are too tight for fp16. -->

### 2.2 The bench uses CUDA events, not `time.perf_counter` — same as project-01

The bench-runner contract is inherited verbatim from
project-01 § 2.1: `torch.cuda.Event(enable_timing=True)` start/stop,
`torch.cuda.synchronize()` between iterations, ≥ 50 warmup, ≥ 500
measured, refuse to report numbers when `std_ms / p50_ms >= 5%`,
sample `nvidia-smi --query-gpu=clocks_throttle_reasons.active`
*between* measured iterations. The kernels under test change; the
measurement contract does not.

The non-obvious choice specific to this project: the bench is run
**twice per kernel** — once standalone (the kernel called in a tight
loop) and once **embedded** (the kernel called from a 1-layer
transformer block). The standalone number tells you whether the
kernel is good; the embedded number tells you whether the
PyTorch/dispatcher overhead is eating the win. A custom kernel that
is 2.5x faster standalone but only 1.1x faster embedded is a
**packaging** problem, not a kernel problem, and the reviewer should
push back on the C++ extension boundary, not the `.cu` source.

### 2.3 The attention kernel is the FlashAttention tiling pattern, not a re-derivation

`mod-004` ex-01 already established that `torch.nn.functional.scaled_dot_product_attention`
dispatches to the FlashAttention path on supported hardware (Ampere
SM 8.0+ for v2, Hopper SM 9.0+ for v3 features). The project does
not ask the candidate to beat the reference FA implementation — that
is a multi-engineer-year effort. It asks the candidate to **implement
the tiling pattern** for a smaller, tractable case (e.g., a single
attention head, fixed block size, no causal mask variants), measure
it against the reference, and *explain the gap*.

The reference solution uses the FlashAttention online-softmax
recurrence from Dao et al. (2022) — a fp32 running max `m`, a
fp32 running sum `l`, and a per-block correction `exp(m_prev -
m_new)` applied to the output accumulator. The deliverable is **not**
a competitive FA kernel; it is a kernel that demonstrates the
candidate understands *why* materializing the `N×N` attention matrix
is the bottleneck and *how* the streaming softmax breaks the
dependency.

A submission whose attention kernel materializes the full
softmax-normalized attention matrix into shared memory has missed
the entire point of the exercise. Flag as a 0 on the attention-kernel
gate even if the numerics match.
<!-- spec-pin: confirm the spec's stated tile shapes (Br/Bc)
and whether the candidate is expected to implement backward as well
as forward. -->

### 2.4 LayerNorm / RMSNorm: Triton first, CUDA second

The Triton LayerNorm in
`modules/mod-004-transformer-optimization/exercise-05/layernorm_triton.py`
is the reference for the forward path. The reasoning chain that
makes it correct is short and worth re-stating, because most
candidate submissions get one of these steps wrong:

1. **Load row in fp16/bf16, accumulate in fp32.** The cast to
   `tl.float32` inside the kernel is mandatory. A pure-fp16
   variance accumulation overflows for moderately wide hidden
   dimensions and produces silently wrong outputs.
2. **`mean = tl.sum(x) / N`, then `x - mean` masked to zero outside
   the valid range, then `var = tl.sum((x-mean)^2) / N`.** The
   masked subtraction is what lets you fuse the second pass with the
   first; without the mask, the out-of-range elements contribute
   non-zero values to the variance.
3. **`rstd = 1.0 / tl.sqrt(var + eps)`, then `y = (x - mean) * rstd
   * w + b`** with `w` and `b` loaded once and reused. The
   one-pass fused write of `y` is what saves the bandwidth.

Going from Triton to CUDA after the Triton version works is
optional in the reference solution but is the path the rubric
rewards with the higher distinction tier. The CUDA version uses
shared memory + warp-shuffle reductions (the pattern from `mod-002`
ex-03) and `__ldg` for the read-only `w`/`b` loads. Past that, the
optimizations are diminishing returns; the Triton compiler is good
enough that hand-rolled CUDA rarely beats it by more than 10–20%
on this op.
<!-- spec-pin: confirm whether the project requires *both*
Triton and CUDA implementations or just one; the rubric tier
treatment depends on this. -->

### 2.5 Fused GEMM-epilogue: ride cuBLAS, don't fight it

The temptation, given a "write a custom kernel" project, is to
hand-roll a GEMM and chase tensor-core peak. **Don't.** The reference
solution does what production code does: calls cuBLAS / cuBLASLt for
the GEMM and **fuses only the epilogue** (bias add, GELU/SwiGLU, dropout).
Two approaches both score full marks on this exercise:

- **`torch.nn.functional.linear` + custom fused epilogue.** The
  PyTorch path materializes the GEMM output to HBM, then a custom
  kernel reads it back, applies bias + activation, and writes the
  final result. This is bandwidth-bound but trivially correct.
- **cuBLASLt with a built-in epilogue.** `cublasLtMatmul` accepts
  an `epilogue` enum (e.g., `CUBLASLT_EPILOGUE_GELU_BIAS`,
  `CUBLASLT_EPILOGUE_BIAS_RELU_AUX`) that fuses the bias and
  activation **inside** the GEMM kernel, eliminating the HBM
  round-trip. This is the production-grade answer; it is faster but
  requires more glue code.

Either is acceptable. Both options must be benchmarked against
unfused PyTorch (`F.linear` → `+ bias` → `F.gelu`) so the win is
quantified. A submission that re-implements GEMM by hand because it
"feels more like CUDA" loses code-quality points: in production you
ride the library, you don't re-derive it.
<!-- spec-pin: confirm cuBLASLt epilogue requirements and the
exact activation variant (GELU vs SwiGLU) the project asks for. -->

### 2.6 The KV-cache update is in-place, not allocate-and-copy

For autoregressive decoding the KV cache grows by one token per
step. The naive PyTorch idiom is `kv_cache = torch.cat([kv_cache,
new_kv], dim=seq_dim)`. This is **catastrophically wrong** at scale:
`torch.cat` allocates a fresh tensor every step, copies the entire
existing cache into it, and walks the allocator. For a long-context
request the per-step latency grows linearly with sequence length.

The reference custom kernel pre-allocates the cache to its maximum
length at session start and **scatters the new KV in-place** at the
current write offset. This is conceptually trivial — a single
`tl.store` per element — but it is the largest measurable win in the
project, often 10×+ at long context lengths.

The non-obvious correctness check: the in-place update must respect
the `attention_mask` so that the kernel reading the cache cannot
read past the current write offset. The reference solution stores
the write offset in a small device-side tensor and the attention
kernel uses it as the dynamic upper bound on the inner loop.

Submissions that re-use `torch.cat` "because it's clearer" should be
flagged for re-implementation; the project's serving-side gate is
not satisfiable without an in-place update.
<!-- spec-pin: confirm the exact serving-throughput PR-N gate
that the in-place update is required to hit; the qualitative
argument above stands either way. -->

### 2.7 Every kernel ships with a shape-window README

Every custom kernel in `src/kernels/<op>/` has a `README.md`
documenting **the shape window where the custom kernel beats the
library and the shape window where it loses**. This is the single
most important documentation artifact in the project. Reviewers
should grep for it explicitly; its absence is the strongest signal
the candidate has not actually swept the input space.

The reference template:

```markdown
# Fused LayerNorm + Residual (custom Triton kernel)

## When to use
- Hidden dim 512–8192, batch * seq ≥ 4096 rows.
- bf16 / fp16 inputs. fp32 falls back to the PyTorch path (we measured
  cuBLAS / cuDNN already saturate bandwidth there).

## When NOT to use
- Hidden dim < 512: PyTorch dispatcher overhead dominates the win.
- Hidden dim > 16384: Triton tile size hits register spill threshold
  on SM 9.0; PyTorch's path is within 5%.
- fp32: see above.

## Measured speedup
See `reports/benchmark_layernorm.md`. Headline: 1.4–2.2× on
the supported window, 1.0× outside.

## Hardware tested
A100 SXM 80 GB (CUDA 12.4), H100 SXM 80 GB (CUDA 12.4).
B200 not tested.
```

A kernel without this README is, by rubric default, **not shippable**
even if its numerics and benchmark gates pass.

### 2.8 The packaging boundary: `setup.py` vs `cpp_extension.load()`

`mod-002` ex-05 covered the two PyTorch C++/CUDA extension paths.
The project requires both:

- **Development**: `torch.utils.cpp_extension.load()` for fast
  iteration. The candidate edits `.cu`, the loader recompiles, the
  Python REPL picks it up. Build cache lives in
  `~/.cache/torch_extensions/`.
- **Production**: `setup.py` with `CUDAExtension` + a pinned
  `nvcc` flag set (`-O3 --use_fast_math` is a question, not an
  answer — see § 5.4). The artifact is a `.whl` that the deployment
  pipeline can `pip install` without a compiler on the target host.

The non-obvious correctness item: the production wheel must be
built against the **same CUDA major version** as the deployment
target. A wheel built against CUDA 12.4 will load on a 12.x host but
not on 11.x; the reference `setup.py` enforces this with a
`CUDA_HOME` check at build time.

### 2.9 Autograd integration: the `torch.autograd.Function` boundary

For ops that appear in training graphs (LayerNorm, attention,
linear-epilogue), the custom kernel must integrate with autograd or
the deliverable is only half-shipped. The reference pattern is the
PyTorch-documented one: a `torch.autograd.Function` subclass with
`forward()` and `backward()` static methods, with the saved tensors
stored via `ctx.save_for_backward()` and re-materialized in the
backward pass.

Submissions that ship a forward-only kernel without explicitly
labeling it as inference-only in the operator's README lose
code-quality points: an op silently broken under autograd is the
worst kind of bug to inherit.
<!-- spec-pin: confirm whether project-02 requires a backward
implementation for the attention kernel or only the LayerNorm /
linear-epilogue kernels. -->

### 2.10 Quarantine, don't silently drop — inherited from project-01

When a kernel misses its numerics gate, the reference pipeline writes
the partial artifacts to `kernels/quarantine/<op>/` and records
`quarantined: true` in the manifest. Exit code is `2` (distinct from
`1` = setup error). The build does **not** mark itself green. The
fall-through path for downstream consumers is to use the reference
PyTorch op for the quarantined operator; this lets the rest of the
pipeline benchmark while the broken kernel is being fixed.

Submissions that silently drop a failing kernel — or worse, ship a
kernel whose numerics test was disabled "temporarily" — are caught
by the same anti-pattern the project-01 rubric calls out. Flag and
fail.

## 3. Validation steps

The reviewer runs, in order:

```bash
# 0. Inside the candidate's Dockerfile, on the target hardware:
docker build -t gpu-opt:review .
docker run --gpus all --rm -v "$PWD":/work gpu-opt:review nvidia-smi
docker run --gpus all --rm gpu-opt:review nvcc --version
```

Confirm: driver and CUDA versions match the manifest's `hardware:`
and `cuda:` fields, GPU compute capability matches the kernels'
target arch (e.g., `-gencode arch=compute_80,code=sm_80` for A100,
`compute_90` for H100).

```bash
# 1. Build the extensions.
docker run --gpus all --rm -v "$PWD":/work gpu-opt:review make build
```

Expected: every kernel in `src/kernels/` compiles cleanly, every
PyTorch C++ extension `.so` lands in `build/`, no `nvcc` warnings
on the production build (development builds may have them; the
production `setup.py` should treat them as errors).

```bash
# 2. Run the numerics gate.
docker run --gpus all --rm -v "$PWD":/work gpu-opt:review make test
```

Expected: every kernel's `tests/test_numerics.py` passes — both the
`torch.allclose` check and the percentile-error check (cf. § 2.1).
Any quarantined kernel is reported with its failure mode.

```bash
# 3. Run the full benchmark sweep.
docker run --gpus all --rm -v "$PWD":/work gpu-opt:review make bench
```

Expected: `reports/benchmark_summary.md` updated with one row per
(kernel, shape, batch) point, every speedup claim sourced from a
JSONL file under `reports/raw/`.

```bash
# 4. Reproduce the numbers.
docker run --gpus all --rm -v "$PWD":/work gpu-opt:review make verify
```

Expected: every benchmark number within 5% of the value recorded in
the manifest on the target SKU. (Cross-SKU reproduction uses the
`make verify HARDWARE=<sku>` form with a wider 15% window, matching
the project-01 convention.)

```bash
# 5. Spot-check attribution: was the win actually shared memory /
# tensor cores / async copy, or was it a measurement artifact?
ncu --set full --target-processes all --launch-skip 5 --launch-count 1 \
    --section MemoryWorkloadAnalysis,SchedulerStats,Occupancy,SpeedOfLight \
    -o profiles/layernorm_custom \
    python -m bench.profile_one --op layernorm --variant custom
```

Expected: the Nsight Compute report for each custom kernel shows
the metric the candidate **claimed** moved (e.g., DRAM throughput
toward peak for a memory-bound op, SM throughput toward peak for a
compute-bound op). The roofline PNGs in `reports/roofline_*.png`
visualize the same data; the NCU report is what the reviewer
double-checks against.

```bash
# 6. Spot-check the manifest signature.
python -m cli.report --verify-manifest
```

Expected: every artifact's recorded sha256 matches the on-disk file
sha256. Exit non-zero on any mismatch. (Inherited from project-01.)

```bash
# 7. Spot-check the shape-window claim.
python -m bench.shape_sweep --op layernorm --variant custom \
    --hidden 256,512,1024,2048,4096,8192,16384 \
    --bs-seq 1024,4096,16384,65536
```

Expected: the speedup-vs-shape curve in
`reports/shape_window_<op>.md` matches the kernel README's claimed
window. A kernel claiming "best at hidden dim 1024–4096" whose
measured curve peaks at 8192 is a documentation bug; flag it.

## 4. Rubric / review checklist

The full rubric is the learning repo's
[`rubric.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/blob/main/projects/project-02-gpu-optimization/rubric.md).
This section is the **reviewer's quick read** mapped to artifacts.

### 4.1 Hard gates

<!-- spec-pin: the project-02 hard-gate IDs (PR-1..PR-N) and
their numeric thresholds (per-op speedup ratios, latency caps,
wall-clock caps) are defined in the learning repo's requirements.md
and must be quoted verbatim here. The qualitative gates below stand
on their own as a reviewer checklist but should be reconciled with
the spec before this section ships. -->

The reviewer-facing gates that apply regardless of the spec's exact
numeric thresholds:

| Gate                       | Source artifact                                      | Pass criterion                                              |
|----------------------------|------------------------------------------------------|-------------------------------------------------------------|
| Numerics — `allclose`      | `reports/numerics_report.md`                         | Every shipped kernel within tolerance vs fp32 reference     |
| Numerics — 99.9 pct error  | `reports/numerics_report.md`                         | Every shipped kernel within tolerance at 99.9th percentile  |
| Bench discipline           | `reports/raw/*.jsonl`                                | std-dev / P50 <= 5%; clocks locked; throttle mask logged    |
| Speedup vs library         | `reports/benchmark_summary.md`                       | Custom kernel beats the library on its declared shape window |
| Shape-window honesty       | `src/kernels/<op>/README.md` + shape sweep           | Claimed window matches measured curve                        |
| Manifest integrity         | `compression_manifest.yaml` (or equivalent)          | Every artifact sha256 verifies                              |
| Reproduces under `make verify` | `make verify` log                                | Numbers within 5% of manifest values                        |

### 4.2 Rubric dimensions

<!-- spec-pin: the D-numbering (D1..D8) and the level-3 / level-5
bars for project-02 must come from rubric.md. The qualitative read
below tells a reviewer what to look at for each likely dimension and
what reference-quality looks like; the column for the level threshold
is left empty until the spec is consulted. -->

For each likely dimension below: artifact to read, what the reviewer
is grading on, and what reference-quality looks like.

| Dimension                  | Read this                                                        | Pass-tier read                                                                 | Reference-tier read                                                                                 |
|----------------------------|------------------------------------------------------------------|--------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------|
| Correctness                | `tests/test_numerics.py`, `reports/numerics_report.md`           | `allclose` passes for all shipped kernels                                      | 99.9 pct error reported; backward pass tested where applicable; numerical-stability commentary       |
| Per-op speedup             | `reports/benchmark_summary.md` (standalone column)               | Custom > library on declared window                                            | > 1.5× on attention, > 1.3× on LayerNorm / RMSNorm, with a documented decomposition                  |
| Embedded speedup           | `reports/benchmark_summary.md` (embedded column)                 | Custom kernel does not regress the 1-layer transformer block                    | Embedded speedup tracks standalone within 10%, proving the C++ ext boundary is clean                  |
| Profiling depth            | `profiles/*.ncu-rep`, `reports/roofline_*.png`                   | One NCU report per kernel, classified memory- or compute-bound                  | Roofline shows each kernel walking up the ridge after each optimization, with the *why* explained    |
| Packaging                  | `setup.py`, `pyproject.toml`, generated `.whl`                   | `pip install` works on a clean container                                       | Wheel works across CUDA minor versions; cross-arch builds tested                                     |
| Reproducibility            | `make verify` log, manifest                                      | `make verify` reproduces every number within 5% on the same SKU                | `build_inputs_sha256` present; CI verifies on every PR                                               |
| Engineering judgment       | `docs/adr/*.md`, kernel READMEs                                  | Each kernel has a "when to use / when not to use" section grounded in the sweep | Stretch path (warp-specialized variant, async-copy variant, FP8 path on H100) shipped with measured deltas |
| Code quality               | `src/`, `mypy --strict` + `ruff` + `black` logs + nvcc warnings  | Lints clean; CUDA / Triton sources follow the project style guide              | Reviewer can add a new fused op in < 1 day via the `Kernel` protocol                                 |

### 4.3 Anti-patterns (auto-deduct)

These come directly from the learning repo's `rubric.md` and the
project's "common mistakes" catalog. A reviewer should grep for them
explicitly.

- **`time.perf_counter` for GPU timing** (grep `src/bench/` for
  `perf_counter`): correctness violation on the bench contract.
  Inherited from project-01 § 4.3.
- **Warmup < 50 iterations** (grep `src/bench/` for `warmup=`):
  same.
- **Numerics test asserts only `allclose`, not the percentile bound**:
  silent drift goes undetected.
- **`torch.cat` in the KV-cache update path**: O(N²) decoding cost
  per request; flag as not-shippable for the serving gate.
- **Materializing the full softmax-normalized attention matrix in
  the custom attention kernel**: defeats the purpose of the
  exercise (cf. § 2.3).
- **Custom GEMM hand-rolled when cuBLAS would do**: lost
  code-quality points (cf. § 2.5); production code rides the
  library.
- **Custom kernel claims "always faster" with no shape sweep**:
  shape-window README is missing or wrong.
- **`--use_fast_math` enabled on the production wheel without a
  numerical-stability review**: a real source of silent fp16
  inaccuracy; treat as a finding (cf. § 5.4).
- **No quarantine flow; failed kernels silently dropped**: same as
  project-01 anti-pattern.

### 4.4 Stretch / distinction bonuses

These are scored but don't count toward Pass:

- **Warp-specialized variant on H100** (producer-consumer warps
  using `wgmma` async tensor-core MMA, per the NVIDIA Hopper Tuning
  Guide): cite the measured speedup and which kernel benefited.
- **`cp.async` (Ampere) / TMA (Hopper) load path**: most relevant
  for the GEMM-epilogue and attention kernels; cite the Nsight
  Compute `smsp__inst_executed_pipe_*` metric that moved.
- **FP8 path on H100** (cf. mod-008 advanced topics): tensor-core
  FP8 attention or GEMM with stated tolerance and measured speedup.
- **Triton autotune + persistent reduction patterns**: shipped with
  the autotune config logged so the cache is reproducible.
- **CUDA Graphs capture** over the full transformer block stitching
  the custom kernels together: cold/warm latency comparison
  reported.

<!-- spec-pin: confirm the exact B-numbering of the stretch
bonuses in the project-02 rubric.md. -->

## 5. Common mistakes

These are the failure modes the reference graders see repeatedly.
They're grouped by phase so a reviewer can locate the cause quickly.

### 5.1 Build and packaging

- **Wrong `-gencode` arch flags.** The candidate builds with the
  default `-arch=sm_60` and the kernel runs but at JIT-compiled
  fallback speed. Always emit explicit per-arch flags for the SKUs
  the manifest claims to support.
- **PTX-only build with no SASS.** Without a matching SASS for the
  target SM, the driver JIT-compiles on first kernel launch and the
  bench measures the JIT, not the kernel. Confirm with
  `cuobjdump --list-text build/*.so`.
- **Mismatched PyTorch ABI.** `torch.utils.cpp_extension` builds
  must use the same ABI flag (`_GLIBCXX_USE_CXX11_ABI`) as the
  PyTorch wheel they will be loaded into. Mismatch produces an
  obscure `undefined symbol` at import time.

### 5.2 Numerics

- **fp16 variance accumulation overflow.** The candidate computes
  variance directly in fp16 for the LayerNorm/RMSNorm reduction;
  for hidden dim ≥ 4096 the accumulator overflows on certain
  inputs. Fix: cast to fp32 inside the reduction (cf. § 2.4).
- **Denormal-handling drift.** A kernel with `--use_fast_math`
  flushes denormals to zero and silently changes the output. Catch
  with the percentile-error check, not just `allclose`.
- **Numerics test on wrong dtype.** Comparing two fp16 outputs to
  each other passes trivially; the reference must be fp32, with the
  candidate's fp16 output cast up for comparison.
- **Masking bugs in Triton.** Loading past `mask=cols < N` with no
  `other=` clause leaves uninitialized values in the
  accumulator. The mod-004 ex-05 reference shows the correct
  pattern (`tl.load(..., mask=mask, other=0.0)`).

### 5.3 Attention kernel

- **Online-softmax recurrence implemented in fp16.** The running
  max `m` and running sum `l` must be fp32; in fp16 the
  cross-block correction `exp(m_prev - m_new)` underflows for
  realistic attention scores.
- **Materializing the full attention matrix to shared memory.**
  Defeats the purpose; cf. § 2.3.
- **No causal mask handling, or a causal mask that double-masks the
  block diagonal.** Common off-by-one; spot-check with `is_causal=True`
  against the reference and look for an off-by-one diff on the
  diagonal element.
- **Backward pass not implemented when the spec required it.**
  Silent failure under `loss.backward()`; cf. § 2.9.

### 5.4 `--use_fast_math` and other "free" flags

- **`--use_fast_math` shipped on the production wheel without
  review.** This flag enables flush-to-zero denormals, less
  accurate `expf` / `sqrtf` / division, and other approximations.
  It is appropriate for **some** kernels (e.g., an attention kernel
  where the softmax already loses precision) and **wrong** for
  others (a LayerNorm where the rstd matters). The reference
  policy: flag is enabled per-kernel, not globally, and the
  numerics report shows the delta with and without it.
- **`--ftz=true` paired with a numerics test that runs on
  zero-mean inputs.** The test inputs never exercise the denormal
  range, so the flag's effect is invisible. Add a targeted test
  on a denormal-range input before trusting the flag.

### 5.5 GEMM-epilogue

- **Hand-rolled GEMM instead of cuBLAS-with-epilogue.** Cf. § 2.5.
- **Epilogue fused after the GEMM with a separate kernel
  launch.** The "fusion" is fake — bias and activation are still
  reading the GEMM output from HBM. Verify with the NCU memory
  trace: a true fused epilogue shows one kernel; the fake one
  shows two with a DRAM round-trip between them.
- **GELU approximated by `tanh` form vs exact form.** Both are
  documented; the reviewer should confirm which one the reference
  PyTorch op uses and match it. A mismatch on the GELU formula is
  a `numerics_report` finding.

### 5.6 KV-cache update

- **`torch.cat` in the per-step update path.** Cf. § 2.6.
- **Off-by-one on the write offset.** The kernel reads the cache
  before the update commits the new token; the attention output
  for the just-generated token is then garbage. Catch with a
  golden-trace test that compares full-rewrite output to
  incremental-update output for a known sequence.
- **No bounds check on the maximum cache length.** A long-context
  request silently writes past the allocated cache buffer and
  segfaults the worker. Always bounds-check.

### 5.7 LayerNorm / RMSNorm

- **One-pass variance using `E[X^2] - E[X]^2`.** Numerically
  unstable for inputs with large means. Use the two-pass
  formulation (mean first, then variance from centered inputs).
- **Skipping the affine `w`/`b`.** Some implementations
  conflate the normalization with a no-op affine; if `w`/`b` are
  trainable, missing them silently breaks training.
- **RMSNorm-specific: forgetting to divide by `sqrt(N)`.** The
  reference RMSNorm `rstd = 1.0 / sqrt(mean(x^2) + eps)` is
  *dividing by `N` then sqrt*. Implementations that
  drop the `/ N` produce values that scale with hidden dimension.

### 5.8 Bench / report

- **No clock-state record in the report.** Inherited from
  project-01 § 5.8; reviewers can't tell whether the bench was
  honest.
- **Roofline plotted as a generic illustration.** The point is to
  show each kernel's actual `(arithmetic_intensity,
  achieved_GFLOPs)` point against the hardware ridge, and to
  explain *why* each optimization moved the kernel. Inherited from
  project-01.
- **Standalone speedup quoted; embedded speedup hidden.** The
  embedded number is the production-relevant one; cf. § 2.2.

### 5.9 Cross-cutting

- **Claiming a stretch result without a measurement.** "Could use
  TMA" is not a bonus; TMA shipped with measured cycle-level
  metrics is. Inherited from project-01.
- **Custom kernel ships without the shape-window README.** The
  single most reliable signal that the candidate has not actually
  swept the input space. Cf. § 2.7.
- **Hardware-specificity unstated.** A kernel tuned for A100 that
  the manifest claims runs on every GPU is a documentation lie.
  The kernel README's `## Hardware tested` section is mandatory.

## 6. References

### Project artifacts (paired learning repo)

- [`projects/project-02-gpu-optimization/README.md`](https://github.com/ai-infra-curriculum/ai-infra-performance-learning/tree/main/projects/project-02-gpu-optimization) — high-level overview, learning outcomes, success criteria.
- `requirements.md`, `architecture.md`, `STEP_BY_STEP.md`, `rubric.md`, `deliverables/README.md` under the same directory — the canonical contract.
  <!-- spec-pin: when these files are confirmed in the learning
  repo, replace this collapsed reference with the individual links
  used in project-01 § 6. -->

### Related module solutions (this repo)

- [`modules/mod-001-gpu-fundamentals/SOLUTION.md`](../../modules/mod-001-gpu-fundamentals/SOLUTION.md) — roofline and the vocabulary every speedup claim depends on.
- [`modules/mod-002-cuda-programming/SOLUTION.md`](../../modules/mod-002-cuda-programming/SOLUTION.md) — the CUDA primitive catalog this project builds from; the "when is hand-written CUDA worth it?" framing is the project's organizing question.
- [`modules/mod-003-performance-profiling/SOLUTION.md`](../../modules/mod-003-performance-profiling/SOLUTION.md) — Nsight Systems / Nsight Compute / PyTorch Profiler discipline; the bench-runner contract is inherited.
- [`modules/mod-004-transformer-optimization/SOLUTION.md`](../../modules/mod-004-transformer-optimization/SOLUTION.md) — FlashAttention, Triton kernel authoring, and the operator catalog the project's kernels target.
- [`modules/mod-008-advanced-topics/SOLUTION.md`](../../modules/mod-008-advanced-topics/SOLUTION.md) — warp specialization, CUDA Graphs, FP8 — the source for the stretch bonuses.
- [`projects/project-01-model-optimization/SOLUTION.md`](../project-01-model-optimization/SOLUTION.md) — the bench-runner contract, manifest discipline, and quarantine flow are inherited verbatim; do not re-derive.
- [`SOLUTION_OVERVIEW.md`](../../SOLUTION_OVERVIEW.md) — track-wide design philosophy ("measure before you optimize", "verify model quality after every change", "hardware specificity is a feature").

### Official standards and primary sources

These are the authoritative documents the reference solution defers
to. Citations are by document family — pin the specific version that
matches the repo's `requirements.md` dependency table when reviewing
a particular submission.

- **NVIDIA CUDA C++ Programming Guide** — authoritative for the
  CUDA memory hierarchy (global / shared / register), warp
  primitives (`__shfl_*_sync`), occupancy semantics, async-copy
  (`cp.async`) on Ampere, and warp specialization / `wgmma` /
  TMA on Hopper.
- **NVIDIA CUDA C++ Best Practices Guide** — the source of the
  vectorized-load pattern (`float4`, `__ldg`) and the occupancy
  vs. ILP discussion the kernel READMEs should reference.
- **NVIDIA Nsight Compute and Nsight Systems User Guides** — the
  source of truth for the metric names cited in the roofline
  commentary (e.g., `dram__throughput.avg.pct_of_peak_sustained_elapsed`,
  `sm__throughput.avg.pct_of_peak_sustained_elapsed`,
  `smsp__inst_executed_pipe_tensor_op_hmma.sum.peak_sustained_active`).
- **NVIDIA A100 / H100 product briefs and datasheets** — for peak
  bf16/fp16/fp8 TFLOPs and HBM bandwidth used in the roofline.
- **NVIDIA cuBLAS / cuBLASLt documentation** — `cublasLtMatmul`
  with epilogue enums (`CUBLASLT_EPILOGUE_BIAS_GELU`,
  `CUBLASLT_EPILOGUE_BIAS_GELU_AUX`, etc.) is the production
  reference for the fused GEMM-epilogue path (§ 2.5).
- **PyTorch documentation**:
  - `torch.utils.cpp_extension` — `load()` (JIT development path)
    and `CUDAExtension` / `BuildExtension` (`setup.py` production
    path) used in § 2.8.
  - `torch.autograd.Function` — the documented integration point
    for custom CUDA / Triton ops with autograd, used in § 2.9.
  - `torch.cuda.Event` — the documented GPU-timing primitive used
    in the bench-runner (§ 2.2).
  - `torch.nn.functional.scaled_dot_product_attention` — the
    reference attention path the project's custom kernel is
    measured against (§ 2.3).
- **Triton documentation and tutorials** — `@triton.jit`,
  `tl.load` / `tl.store` masking semantics, `tl.sum` reductions,
  autotune. The fused LayerNorm tutorial in the Triton repository
  is the reference template for § 2.4.
- **FlashAttention paper** — Dao, Fu, Ermon, Rudra, Ré,
  "FlashAttention: Fast and Memory-Efficient Exact Attention with
  IO-Awareness", 2022 — for the online-softmax recurrence and
  tiling pattern used in § 2.3. The follow-on FlashAttention-2
  (Dao, 2023) and FlashAttention-3 (Hopper-specific, Shah et al.,
  2024) describe the warp-specialization patterns relevant to the
  stretch bonus.

### Foundational papers cited in the project spec

<!-- spec-pin: confirm the project-02 README's reference list;
the entries below are the standard CUDA-for-transformers reading
that the reference solution defers to. -->

- Dao et al., "FlashAttention: Fast and Memory-Efficient Exact
  Attention with IO-Awareness", 2022.
- Dao, "FlashAttention-2: Faster Attention with Better Parallelism
  and Work Partitioning", 2023.
- Shah et al., "FlashAttention-3: Fast and Accurate Attention with
  Asynchrony and Low-precision", 2024 — Hopper-specific patterns
  (warp specialization, TMA).
- Tillet, Kung, Cox, "Triton: An Intermediate Language and Compiler
  for Tiled Neural Network Computations", MAPL 2019 — the language
  the LayerNorm / RMSNorm reference kernel is written in.
- Kwon et al., "Efficient Memory Management for Large Language
  Model Serving with PagedAttention", SOSP 2023 — the systems
  context for the KV-cache update path in § 2.6.

### Cross-track pointers

- `engineer-solutions/mod-107-gpu-computing/exercise-02-cuda-kernel`
  — the full PyTorch C++ extension reference (`vector_add.cu` +
  `my_ops.cpp` + `setup.py` + `bench.py`) that the project's
  packaging path builds on (cf. mod-002 ex-05 README).
- `engineer-solutions/mod-110-llm-infrastructure/exercise-06-inference-optimization-llm`
  — the end-to-end LLM inference optimization chain that
  downstream project-03 builds on; the kernels delivered here are
  the operators that vLLM / TensorRT-LLM are themselves shipping.
- `senior-engineer-solutions/projects/project-201-distributed-training/SOLUTION.md`
  — the multi-GPU training counterpart; this project deliberately
  stays single-GPU.
- `architect-solutions/projects/project-301-enterprise-mlops/SOLUTION.md`
  — architecture-level cost / capacity reasoning that consumes
  the kernel speedups produced here.
