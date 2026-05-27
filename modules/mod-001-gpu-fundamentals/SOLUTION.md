# SOLUTION — GPU Fundamentals

> Read this *after* you have worked through the exercises. The
> exercise solutions show *what* the numbers are; this document
> explains *why* the metrics that matter on a GPU are different
> from the ones that matter on a CPU, and which mental models pay
> off for the rest of the curriculum.

## What this module is really teaching

Most engineers approach GPUs as "CPUs but parallel" and bounce off
the abstractions for weeks. The actual mental model you need is:

- A GPU is a **bandwidth machine**, not a clock-speed machine.
- An H100 streaming multiprocessor (SM) can issue 4 warp
  instructions per cycle across 132 SMs at 1.8 GHz — but only if
  memory feeds keep up.
- Most real workloads sit in the **memory-bound** half of the
  roofline. Spending a week shaving compute when the kernel is
  memory-bound is the most common mistake in performance work.

The five exercises in this module are scaffolding to internalize
that single insight. Once you reach for the roofline diagram
before reaching for nsys, the module has done its job.

## Architectural decisions and *why*

### Decision 1: Arithmetic-intensity-first analysis (roofline before profiler)

Exercise 04 deliberately asks you to compute arithmetic intensity
*before* you profile anything. The order matters: if you profile
first, you'll chase whatever the profiler highlights — usually a
hot kernel — without ever asking whether that kernel *can* go
faster on this hardware. The roofline tells you the ceiling. The
profiler tells you the floor. You optimize the gap.

**Anti-pattern to avoid**: "I saw a 12ms kernel in nsys, let me
optimize that one." If that kernel is already at 95% of memory
bandwidth, your day is spent for nothing.

### Decision 2: Warp divergence treated as a first-class metric

Exercise 05 makes you count divergent branches per warp explicitly.
The reason: modern transformer kernels divide neatly along
warp-sized tile boundaries, but custom kernels (your kernels)
won't, and a single `if (x < threshold)` inside a warp can halve
your throughput. The reference solution's analyzer is intentionally
mechanical — counting divergent paths is more important than fancy
visualization at this stage.

### Decision 3: Peak-throughput calculation done from spec sheets, not nvidia-smi

Exercise 02 walks you through computing peak TFLOPS and HBM
bandwidth from the H100 spec sheet by hand. This is on purpose.
Engineers who only ever read nvidia-smi never learn to spot when
their workload is delivering 30% of theoretical and could be at
80% with a different memory layout. The arithmetic — `2 × SMs ×
clock × tensor-core throughput-per-clock` — should feel as natural
as computing CPU GHz × cores × IPC.

### Decision 4: Occupancy is a tool, not a target

The occupancy calculator (exercise 03) is a debugging aid, *not* a
goal. Many beginner-tier GPU resources tell you to "maximize
occupancy"; this is wrong. A kernel at 50% occupancy with full
memory-bandwidth utilization is faster than the same workload at
100% occupancy with register spills. The reference solution
intentionally computes occupancy *and* records the register count
and shared-memory pressure that caused it, so the trade-off is
explicit.

### Decision 5: CPU-vs-GPU comparison kept honest

Exercise 01 measures the same workload on both CPU and GPU and
prints a speedup. The temptation is to pick a workload where the
GPU wins by 100x (large matmul) and pretend that's representative.
The reference solution runs three workloads — large matmul, small
matmul, sequential reduction — to demonstrate that the speedup
depends entirely on whether the work can be saturated with parallel
threads. A small matmul running 0.8x on GPU (yes, *slower*) is the
single most clarifying number a junior performance engineer can
encounter.

## Trade-offs we deliberately accepted

### No CUDA code in this module

This module is purely measurement and analysis. The CUDA code lives
in mod-002. The split is deliberate: trying to learn occupancy
calculation while debugging a kernel that won't compile is a recipe
for confusion. Master the metrics first; you'll write better
kernels when you do.

### Roofline computed for a single GPU

The reference roofline assumes one H100. Multi-GPU rooflines exist
(they account for NVLink + PCIe + collective overhead) and live in
mod-006 — distributed inference — where they pay off. Until then,
single-GPU is the right scope.

### nvidia-smi vs nvprof/nsys: introduced in mod-003

This module shows nvidia-smi only as a sanity check ("is the GPU
even running?"). Profiler-driven analysis belongs in mod-003. You
need a mental model before tools become useful; otherwise you stare
at flamegraphs without knowing what's actionable.

## Common mistakes graders see

1. **Reporting "I optimized X by 3x" without saying which roofline
   regime X is in**: 3x in the compute-bound regime is impressive;
   3x in the memory-bound regime usually means the baseline was
   broken.
2. **Treating SM count and warp count as the same thing**: 132 SMs,
   4 warp schedulers each, 32 threads per warp. Mixing these up
   produces nonsense peak numbers.
3. **Computing peak FLOPs using the *wrong* tensor-core mode**: FP16
   tensor cores hit 989 TFLOPS on H100; FP8 hits ~2000; FP32 hits
   67. Always cite the precision in your peak number.
4. **Conflating "high occupancy" with "good"**: it's a hint, not a
   verdict.
5. **Forgetting that L2 is 50 MB on H100**: many workloads that look
   memory-bound on paper actually fit in L2 once tiled correctly.
6. **Comparing GPU latency to CPU throughput**: a single GPU launch
   has 5-10µs of fixed cost. For tiny ops, the CPU wins on latency
   even if the GPU wins on throughput.

## When to go beyond this implementation

- Add an **interactive roofline visualizer** that takes nsys output
  and overlays each kernel's `(intensity, performance)` point on
  the roofline diagram. This is the next exercise (mod-003 ex-05).
- Compute peak numbers for **multiple precisions side-by-side**
  (FP32/TF32/BF16/FP16/FP8/INT8) so the precision/throughput
  trade-off is visible. This sets up mod-005 (compression).
- Generalize the occupancy calculator to handle **CUDA Graphs**
  (mod-008 ex-01), where the launch cost disappears but the
  occupancy ceiling still binds.

## Related curriculum touchpoints

- `performance/mod-002-cuda-programming` — write the kernels whose
  metrics you've just learned to read.
- `performance/mod-003-performance-profiling` — apply nsys/ncu/PyTorch
  Profiler to confirm the roofline analysis with real numbers.
- `performance/mod-005-model-compression` — every quantization
  choice moves your workload along the roofline; this module is the
  language you need to describe the move.
- `engineer/mod-107-gpu-computing/exercise-01-gpu-introspection` —
  the working pynvml CLI that turns these metrics into a live
  introspection tool.
