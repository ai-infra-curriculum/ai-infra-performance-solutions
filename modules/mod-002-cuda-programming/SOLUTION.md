# SOLUTION — CUDA Programming

> Read this *after* you have written and benchmarked your own
> versions of each kernel. This document explains *why* the
> reference kernels are shaped the way they are, what the common
> bad patterns are, and where to draw the line between hand-written
> CUDA and library-provided kernels.

## What this module is really teaching

Most production AI workloads do **not** need you to write CUDA from
scratch — cuBLAS, cuDNN, FlashAttention, and Triton already cover
80% of the operations you'll ever care about. This module exists so
you can answer two questions confidently:

1. **When is hand-written CUDA worth it?** (Almost never. The
   exceptions are surgical.)
2. **What is the library actually doing under the hood?**

If you finish this module able to read FlashAttention's source and
recognize the tiling pattern, the module has done its job. You will
write very little custom CUDA in your career, but you'll read CUDA
every week.

## Architectural decisions and *why*

### Decision 1: Vector-add as the "hello world" — and why it's misleading

Exercise 01 (vector-add) is the canonical CUDA tutorial. The
reference solution intentionally adds a second variant that
**vectorizes** the load (`float4` instead of `float`) and shows the
2-3x speedup. The reason: a naive `kernel<<<N/256, 256>>>` reaches
maybe 60% of peak bandwidth; the vectorized version reaches 90%+.
A junior engineer who only sees the naive version learns the wrong
lesson — that "GPU goes brrr" — instead of learning that *every*
GPU kernel needs to wring the last bit of bandwidth out of the
memory subsystem.

**Anti-pattern to avoid**: shipping the naive version as
"production-ready" because the unit test passes. CUDA correctness
and CUDA performance are different bugs.

### Decision 2: Tiled matmul that beats cuBLAS by 0% on purpose

Exercise 02 (tiled matmul) deliberately targets a hand-written
kernel that reaches 50-70% of cuBLAS performance, not 100%. The
reason: hitting cuBLAS-grade performance requires WMMA / tensor-core
intrinsics, register-level tiling, async copies, and split-K
strategies that take weeks to implement correctly. The teaching
goal is to internalize the **two-level tiling pattern** (block
tile → warp tile → thread tile) that every modern GEMM uses, not
to recreate cuBLAS.

The exercise asks you to also benchmark cuBLAS on the same
operation. When the student sees their carefully tuned kernel at
3.2 TFLOPS and cuBLAS at 280 TFLOPS, they internalize *why* "just
call the library" is the right default.

### Decision 3: Reduction as the warp-shuffle teaching vehicle

Exercise 03 (reduction) walks through four versions:

1. Naive `atomicAdd` (slow, correct).
2. Shared-memory reduction with `__syncthreads()`.
3. Warp-shuffle reduction (`__shfl_down_sync`) within a warp.
4. Multi-block reduction with grid-stride loop.

Each version is 2-4x faster than the previous. The teaching value
is in the **progression**, not the final kernel. Reduction is the
canonical example of "a problem that looks trivial but exposes
every CUDA primitive you'll ever need" — synchronization, shared
memory, warp primitives, grid-stride loops, atomic operations.

**Anti-pattern to avoid**: skipping straight to the warp-shuffle
version. Without seeing the naive `atomicAdd` melt down at scale,
you won't believe the warp version's speedup is real.

### Decision 4: Occupancy tuning is a real exercise, not a footnote

Exercise 04 (occupancy tuning) takes a working kernel and asks you
to vary block size, register count (via `__launch_bounds__`), and
shared memory usage. The lesson: occupancy is a sensitivity
analysis, not a single number. The reference solution sweeps block
sizes from 64 → 1024 and produces a plot — the curve is rarely
monotonic, and the sweet spot depends on the GPU. Future-you
debugging a slow kernel will run this sweep instinctively.

### Decision 5: PyTorch C++ extension as the "real" deliverable

Exercise 05 ties the module together by wrapping a CUDA kernel as a
PyTorch C++ extension. This is where 99% of production CUDA lives:
not in standalone binaries, but glued into a PyTorch training or
inference pipeline. The reference solution uses
`torch.utils.cpp_extension.load()` for development and
`setup.py` for production builds. The exercise teaches the **two
boundary conditions** that bite engineers later:

1. CUDA stream synchronization with PyTorch's default stream.
2. autograd integration (so `loss.backward()` actually works
   through your custom op).

Without those two, the extension "works" in the forward pass and
silently corrupts training in the backward pass.

## Trade-offs we deliberately accepted

### No PTX inline assembly

PTX is the right answer for a handful of operations (cooperative
groups, specific tensor-core ops not exposed by intrinsics), but
99% of CUDA code shouldn't touch it. The exercises stay in CUDA C++
because the time investment per insight is much better.

### No multi-GPU primitives

NCCL collectives, peer-to-peer memory access, and unified memory
all live in mod-006 (distributed inference). The reason: single-GPU
performance must be solid before multi-GPU complexity helps.

### Triton skipped here, addressed in mod-004

Triton is the modern alternative to hand-written CUDA for ML
kernels (FlashAttention 2, Mamba, etc. are all Triton). Mod-004
ex-05 covers Triton specifically — the teaching there is "Triton
gives you 90% of CUDA's performance with 30% of the code." Putting
Triton in mod-002 would dilute the "this is what CUDA actually
looks like" goal.

## What the tests cover (and why)

The reference test suites focus on **numerical correctness against
a CPU reference** and **bandwidth utilization above a threshold**
(typically 70-80% of theoretical peak). They deliberately do not
test for an absolute time, because GPU clock boost behavior makes
those tests flaky. The threshold-based check catches real
regressions (a code change that broke the vectorized load) without
false-positives from thermal throttling.

## Common mistakes graders see

1. **Forgetting `cudaDeviceSynchronize()` before timing**: your
   kernel launch returns instantly; your measurement is meaningless
   without a sync.
2. **Allocating per-kernel scratch memory**: `cudaMalloc` inside a
   loop is the single largest source of slowdown in junior CUDA
   code. Pre-allocate once.
3. **Shared memory bank conflicts**: writing `tile[threadIdx.y][threadIdx.x]`
   instead of `tile[threadIdx.x][threadIdx.y]` halves your shared
   memory bandwidth and barely shows up in nsys without `ncu`.
4. **Forgetting to set `--use_fast_math` or its non-existence
   matters less than they think**: most ML workloads are
   bandwidth-bound, not transcendental-function-bound.
5. **Comparing kernel time without warm-up**: the first launch
   includes JIT compilation and lazy module loading. Always discard
   the first run.
6. **Using `printf` inside kernels in production builds**: works
   fine in tests, melts performance silently because each warp
   serializes the printf queue.

## When to go beyond this implementation

- Port one kernel to **Triton** (mod-004 ex-05) and compare the LOC
  vs performance trade-off.
- Add **CUDA Graphs** capture around the launches (mod-008 ex-01)
  for ~5µs per-launch savings — meaningful for small batch
  inference.
- Profile with **ncu** at the metric level (`sm__inst_executed`,
  `dram__bytes_read`) to confirm your bandwidth claims.

## Related curriculum touchpoints

- `performance/mod-001-gpu-fundamentals` — the metrics you're now
  optimizing against.
- `performance/mod-003-performance-profiling` — nsys/ncu workflows
  to confirm your kernel hits its roofline.
- `performance/mod-004-transformer-optimization` — production
  kernels (FlashAttention, paged attention) that build on these
  primitives.
- `engineer/mod-107-gpu-computing/exercise-02-cuda-kernel` — the
  PyTorch-wrapped CUDA extension scaffold.
