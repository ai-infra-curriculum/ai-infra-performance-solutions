# Exercise 01 — When does the GPU actually help? (SOLUTION)

> Paired learning file: `ai-infra-performance-learning/modules/mod-001-gpu-fundamentals/exercises/exercise-01-cpu-gpu-comparison`
> Module rationale: see [`../SOLUTION.md`](../SOLUTION.md) — "Decision 5: CPU-vs-GPU comparison kept honest".

## 1. Solution overview

The exercise asks learners to measure the *same* workload on CPU
and GPU and report a speedup. The reference answer deliberately
runs **three** workloads, not one, because the single most common
mistake at this level is generalizing from a single large matmul:

| Workload | What it stresses | Expected qualitative result |
|---|---|---|
| Large dense matmul (e.g. `4096 × 4096`) | Tensor-core throughput, HBM bandwidth | GPU wins by a large factor |
| Small matmul (e.g. `64 × 64`) | Kernel-launch fixed cost vs. actual work | GPU is at parity or **slower** than CPU |
| Sequential reduction without parallel reduction tree | Serial dependency that cannot be fanned out | GPU underperforms expectations; CPU may be competitive |

The point of running all three is that "GPU speedup" is a property
of the *workload* (parallel saturation + arithmetic intensity),
not of the hardware. A single number is misleading; a comparison
table forces honest reasoning.

### Classification: this is an implementation + analysis exercise

The deliverable is runnable timing code **and** a written
interpretation. Code without interpretation fails the exercise;
interpretation without supporting numbers also fails.

## 2. Worked answer or implementation

### 2.1 Reference measurement harness (PyTorch)

PyTorch is the standard toolkit for this exercise because its
CUDA event API is the documented way to time GPU work without
polluting the measurement with host/device synchronization
overhead (see PyTorch CUDA semantics docs linked in §6).

```python
"""cpu_gpu_bench.py — reference timing harness for exercise 01.

Runs three workloads on CPU and GPU and prints a comparison.
Requires: torch >= 2.0 with CUDA. Falls back gracefully when no
GPU is available (prints an explanation instead of fake numbers).
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass

import torch


@dataclass
class Result:
    workload: str
    device: str
    size: str
    seconds: float
    tflops: float | None = None


def _time_cpu(fn, repeats: int) -> float:
    # Warm up once to let any lazy BLAS loads happen.
    fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return statistics.median(samples)


def _time_gpu(fn, repeats: int) -> float:
    # CUDA events are the documented way to measure GPU-side time
    # without being contaminated by host-side scheduling jitter.
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / 1000.0)  # ms -> s
    return statistics.median(samples)


def bench_matmul(n: int, device: str, repeats: int = 25) -> Result:
    dtype = torch.float32
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    def _run() -> None:
        _ = a @ b

    seconds = _time_gpu(_run, repeats) if device == "cuda" else _time_cpu(_run, repeats)
    flops = 2.0 * n * n * n  # 2*n^3 FLOPs for dense GEMM
    return Result(
        workload="matmul",
        device=device,
        size=f"{n}x{n}",
        seconds=seconds,
        tflops=flops / seconds / 1e12,
    )


def bench_sequential_reduction(n: int, device: str, repeats: int = 25) -> Result:
    # Intentionally serial: a Python-level loop prevents the GPU
    # from running a parallel reduction tree. This is the "GPU
    # cannot save you" workload.
    x = torch.randn(n, device=device)

    def _run() -> None:
        acc = torch.zeros((), device=device)
        for i in range(n):
            acc = acc + x[i]

    seconds = _time_gpu(_run, repeats) if device == "cuda" else _time_cpu(_run, repeats)
    return Result(workload="serial_reduce", device=device, size=f"n={n}", seconds=seconds)


def main() -> None:
    cuda_ok = torch.cuda.is_available()
    if not cuda_ok:
        print("No CUDA device. Rerun on a machine with a GPU to produce a")
        print("comparison. Reporting CPU-only numbers would hide the")
        print("workload-dependence this exercise is teaching.")
        return

    workloads = [
        ("large_matmul", lambda d: bench_matmul(n=4096, device=d)),
        ("small_matmul", lambda d: bench_matmul(n=64, device=d, repeats=200)),
        ("sequential_reduction", lambda d: bench_sequential_reduction(n=1024, device=d)),
    ]

    print(f"{'workload':<22} {'cpu_s':>10} {'gpu_s':>10} {'speedup':>10}")
    for name, fn in workloads:
        cpu_r = fn("cpu")
        gpu_r = fn("cuda")
        speedup = cpu_r.seconds / gpu_r.seconds
        print(f"{name:<22} {cpu_r.seconds:>10.4f} {gpu_r.seconds:>10.4f} {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
```

### 2.2 How to interpret the three rows

Expected qualitative outcome on a modern server GPU (H100, L40S,
A100, etc.) paired with a modern multi-core x86 CPU:

- **`large_matmul` (4096×4096)**: GPU wins by one to three orders
  of magnitude. The workload is large enough to saturate the
  tensor cores; the kernel launch cost is amortized across the
  work.
- **`small_matmul` (64×64)**: GPU speedup collapses, often below
  1.0×. The actual compute is in the microsecond range while
  each GPU kernel launch carries fixed overhead measured in
  microseconds too (NVIDIA CUDA C++ Programming Guide notes
  that per-launch overhead is non-negligible for sub-millisecond
  kernels — see §6).
- **`sequential_reduction`**: the per-element Python loop issues
  one kernel per element. The GPU cannot form a parallel
  reduction tree because the graph is serial at the framework
  level. CPU often wins outright. This is the clarifying
  negative result the module calls out in
  [`../SOLUTION.md`](../SOLUTION.md) — "Decision 5".

### 2.3 Required written analysis (prose the learner must produce)

The written answer must contain **at minimum**:

1. A table of the three measured timings with CPU/GPU columns and
   a `speedup` column.
2. For each row, one sentence identifying whether the workload is
   compute-bound, memory-bound, or launch-overhead-bound.
3. A one-paragraph conclusion that answers the exercise title:
   *the GPU helps when the workload can be fanned out across
   enough threads to amortize launch overhead and saturate at
   least one of the two roofline ceilings* (compute or memory
   bandwidth).

### 2.4 Known hardware context the learner should cite

The exact numbers vary with hardware, but the reference solution
should name the measurement setup explicitly. Recommended
template (fill with the learner's actual device):

- GPU model and VBIOS / driver (from `nvidia-smi`)
- CPU model and core count (`lscpu`)
- Precision used (default here is FP32; mention if TF32/BF16 was
  enabled)
- PyTorch version and whether cuDNN/cuBLAS auto-tuning was on

### 2.5 Why no fixed reference speedup number is quoted

A single "representative" CPU/GPU speedup table (e.g. "H100 vs
32-core Sapphire Rapids achieves Nx on 4096 matmul") is
deliberately not pinned here because we do not have a verified
first-party benchmark to cite. Learners run the harness on their
own hardware; the point of this exercise is their measurement,
not a fixed reference number.

## 3. Validation steps

A graded submission should be reproducible. The reviewer runs:

1. **Static check**: `python -m py_compile cpu_gpu_bench.py` — no
   syntax errors; imports resolve on a machine with
   `torch>=2.0`.
2. **Dry run on CPU**: run the harness without a GPU present.
   The program must print the "no CUDA" message and exit
   cleanly, not raise. (Tests the fallback branch.)
3. **GPU run**: on the target machine, execute the harness. All
   three workload rows must appear.
4. **Sanity bounds** for the table:
   - `large_matmul` GPU time should be << CPU time. A speedup
     < 2× on a modern server GPU with a 4096 matmul usually
     indicates the GPU was not warm, cuBLAS was not selected,
     or the CPU timing captured only cache-resident data.
   - `small_matmul` GPU time of less than ~30 µs is suspicious —
     a single kernel launch alone is typically in the low-µs
     range, so sub-30µs end-to-end timings usually mean the
     measurement is being optimized away or the stream is not
     synchronized.
   - `sequential_reduction` CPU time should be much less than
     GPU time; the opposite result indicates the Python-level
     loop was not actually iterated.
5. **Written interpretation present**: the three required prose
   elements from §2.3 are in the submission.

## 4. Rubric or review checklist

Total: 10 points. Pass threshold: 7.

| Criterion | Points | What earns full credit |
|---|---|---|
| Correct use of `torch.cuda.Event` + `torch.cuda.synchronize` | 2 | Both `start.record()`/`end.record()` and an explicit `synchronize` before reading `elapsed_time`. No reliance on `time.perf_counter` for GPU-side timing. |
| Three distinct workloads measured | 2 | At least one large-parallel, one small-parallel, and one serial workload. Varying only matrix size does not count. |
| Warm-up run included before timed samples | 1 | At least one untimed invocation per workload before measurement; prevents one-off allocator / autotune cost from skewing the first sample. |
| Multiple samples + median (or mean + stddev) reported | 1 | Single-sample timings are rejected — GPU timing is noisy at the microsecond scale. |
| Speedup column computed correctly | 1 | `speedup = cpu_time / gpu_time`. Not the inverse. |
| Written interpretation covers all three rows | 2 | §2.3 items 1–3 all present. |
| Hardware / software context recorded | 1 | GPU model, CPU model, PyTorch version, and precision stated. |

### Bonus (+1, max 10)

Learner adds a fourth row with a *mixed* workload (e.g. elementwise
op over a large tensor) and correctly classifies it as
memory-bandwidth-bound — demonstrating they understand the
roofline model introduced in exercise 04.

## 5. Common mistakes

Observed repeatedly in submissions:

1. **Timing GPU work with `time.perf_counter()` only.** Without
   `torch.cuda.synchronize()`, the host returns immediately after
   enqueueing the kernel, so the "timing" measures launch
   latency, not kernel runtime. See PyTorch CUDA semantics docs
   (§6) — this is why `torch.cuda.Event` exists.
2. **Reporting a single speedup number and generalizing.** The
   whole point of the exercise is that there is no single number.
3. **Forgetting the warm-up.** The first invocation pays for
   allocator setup, cuBLAS kernel selection, and sometimes JIT
   compilation. This first-sample cost is often 10× the steady
   state and will dominate a median of too few samples.
4. **Running the "small" workload with too few repeats.** At the
   microsecond scale, you need hundreds of samples for the median
   to stabilize. 3 samples is noise.
5. **Comparing GPU throughput to CPU latency (or vice versa).**
   A single GPU launch has fixed overhead. For tiny ops, CPU
   wins on latency even if GPU wins on throughput — this is a
   real result, not a measurement error. Module-level
   [`../SOLUTION.md`](../SOLUTION.md) lists this explicitly.
6. **Using FP32 CPU numbers against TF32 GPU numbers without
   saying so.** TF32 on tensor cores has very different peak
   than FP32 on CUDA cores; apples-to-oranges comparisons
   inflate the speedup number and obscure the real lesson.
7. **Letting `torch.compile` or cuDNN autotune run inside the
   timed region.** Autotune can launch dozens of candidate
   kernels on the first call. Warm up until median stabilizes.

## 6. References

**Official project documentation (primary sources):**

- NVIDIA CUDA C++ Programming Guide — kernel launch semantics,
  grid/block sizing, host-device synchronization behavior.
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA CUDA Runtime API — events and timing
  (`cudaEventRecord`, `cudaEventElapsedTime`).
  <https://docs.nvidia.com/cuda/cuda-runtime-api/>
- PyTorch CUDA semantics documentation — asynchronous execution,
  `torch.cuda.Event`, `torch.cuda.synchronize`. This is the
  official PyTorch guidance on how to time CUDA work correctly.
  <https://pytorch.org/docs/stable/notes/cuda.html>
- PyTorch `torch.cuda.Event` API reference.
  <https://pytorch.org/docs/stable/generated/torch.cuda.Event.html>

**Local exercise context:**

- [`../SOLUTION.md`](../SOLUTION.md) — module rationale, including
  "Decision 5: CPU-vs-GPU comparison kept honest" and the common
  mistakes list the rubric above pulls from.
- [`../exercise-04/`](../exercise-04/) — roofline analysis; the
  bonus item above and several rubric lines assume the roofline
  vocabulary from exercise 04.
- [`./README.md`](./README.md) — short-form cross-reference to the
  engineer-tier working ResNet training script that runs on both
  CPU and GPU.
