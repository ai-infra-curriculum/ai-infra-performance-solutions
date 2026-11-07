# Exercise 05 — End-to-End Optimization Cycle

## 1. Solution overview

This exercise is the synthesis of Exercises 01–04. The learner is
given a training loop with **three layered bottlenecks** (host-side
starvation, a memory-bound kernel, and a hidden host sync) and must
fix them in the correct order, re-profiling between fixes.

The teaching point: **the bottleneck shifts as you fix things.** A
clever kernel optimization applied before the DataLoader is fixed
produces 0 % end-to-end speedup, because the GPU is already idle 40 %
of the time. The reference deliverable is a **speedup table** that
shows the contribution of each fix in isolation and in sequence, so
the learner can see the shift happen.

See the module-level SOLUTION.md, *Decision 5*, for the pedagogy.

## 2. Worked answer / implementation

### 2.1 Baseline measurement

Before changing anything, record a steady-state baseline with the
same methodology that will be used after each fix. This is the only
way to attribute speedup correctly.

```python
import torch
import torch.cuda as cuda

def steady_state_step_time(train_step, iters=20, warmup=5):
    for _ in range(warmup):
        train_step()
    cuda.synchronize()
    start, end = cuda.Event(enable_timing=True), cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        train_step()
    end.record()
    cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms/step
```

Record alongside:

- `nsys` trace of ~5 steady-state steps.
- PyTorch Profiler TensorBoard Overview percentages.
- `torch.cuda.memory_stats()` peak allocated / reserved.

### 2.2 The fix order

The bottlenecks must be addressed in the order nsys reveals them.
The reference sequence:

**Fix 1 — DataLoader starvation (host-bound).**
nsys shows wide CPU-side gaps between kernel launches. Fixes:
- Increase `DataLoader(num_workers=N, persistent_workers=True)` so
  workers survive across epochs.
- Use `pin_memory=True` and `batch.to(device, non_blocking=True)`
  to overlap HtoD with compute (requires pinned memory for
  `non_blocking` to actually be async).
- Move cheap augmentation onto the GPU where possible.

**Fix 2 — Memory-bound kernel (compute-side).**
Once the GPU is fed, nsys shows the top-time kernel back to back.
ncu Speed-of-Light confirms memory-bound with a low
`smsp__sass_average_data_bytes_per_sector_mem_global_op_ld` (narrow
bytes-per-sector → uncoalesced loads). Fixes:
- Make the contiguous axis the innermost loop axis (`.contiguous()`
  on a reshape that otherwise strides unfavorably).
- Replace an index-select + reduction with a fused op where
  available (e.g., `torch.nn.functional.embedding_bag`).
- Enable Tensor Core paths (`torch.set_float32_matmul_precision("high")`
  for matmuls; use `bfloat16` / `float16` where the model supports it;
  on PyTorch 2.x, let `torch.compile` fuse small ops).

**Fix 3 — Hidden host sync in the backward path.**
nsys shows a tall `cudaStreamSynchronize` bar near the end of the
step. The usual culprits:
- `.item()` or `.cpu()` on a scalar loss for logging, inside the
  timed loop.
- `print(tensor)` inside the step (forces a sync to format the
  tensor on the host).
- `torch.cuda.synchronize()` left over from debugging.
- Gradient-norm clipping implemented as
  `grad.norm().item() > threshold` instead of
  `torch.nn.utils.clip_grad_norm_`.

Fix: move the `.item()` call outside the hot loop (e.g., log every
N steps, async via a queue) or compute the gradient-norm check on
the device.

### 2.3 Speedup attribution table

The required artifact. Each row is one configuration; "Step time"
is from §2.1, "GPU busy %" is the TensorBoard Overview "Kernel"
bucket, and "Peak reserved" is `reserved_bytes.all.peak`.

| Configuration | Step time (ms) | GPU busy % | Peak reserved (GiB) | Speedup vs. baseline |
|---|---|---|---|---|
| Baseline | `<B>` | `<G0>` | `<R0>` | 1.00× |
| + DataLoader fix | `<T1>` | `<G1>` | `<R1>` | `<B/T1>` |
| + Kernel fix | `<T2>` | `<G2>` | `<R2>` | `<B/T2>` |
| + Host-sync fix | `<T3>` | `<G3>` | `<R3>` | `<B/T3>` |
| Kernel fix **only** (no others) | `<Tk>` | `<Gk>` | `<Rk>` | `<B/Tk>` |
| Host-sync fix **only** | `<Th>` | `<Gh>` | `<Rh>` | `<B/Th>` |

The last two rows are the teaching rows: they usually show that
each individual lower-level fix is small until the higher-level fix
is in place.

### 2.4 Re-profile between fixes

A short checklist to run after each change:

1. Re-run `nsys` on 5 steady-state steps. Compare the step-time
   histogram shape, not just the mean.
2. Re-run the PyTorch Profiler Overview. If the step-time pie chart
   shifted (e.g., DataLoader bucket shrank, Kernel bucket grew),
   the fix landed where expected.
3. If one kernel now dominates, update the Exercise 02 target and
   re-run ncu against it.
4. Record peak allocated and reserved; large memory regressions
   often follow from "obvious" speedups (`torch.compile` can
   increase peak memory, bf16 cuts it).

## 3. Validation steps

1. `baseline.nsys-rep`, `after-dl.nsys-rep`, `after-kernel.nsys-rep`,
   `after-sync.nsys-rep` all exist and each represents the same
   number of steady-state steps.
2. For each configuration, the TensorBoard Overview page renders
   and the "Kernel" percentage is recorded in the speedup table.
3. The speedup table contains at least one row where an individual
   fix shows a smaller speedup in isolation than when combined with
   earlier fixes (the "bottleneck shifts" lesson).
4. Peak reserved memory is recorded for every row — regressions are
   flagged explicitly.
5. No `torch.cuda.synchronize()` or `.item()` remains inside the
   timed loop in the final configuration (greppable).
6. Final numerical outputs (loss, metric) are within tolerance of
   the baseline — speedups must not come from accidental behavioral
   changes.

## 4. Rubric / review checklist

| Item | Pass |
|---|---|
| Baseline recorded **before** any change | [ ] |
| Three distinct bottlenecks identified from profiling evidence | [ ] |
| Fixes applied in nsys-driven order (host → kernel → sync) | [ ] |
| Speedup table includes isolated-fix rows, not only sequential | [ ] |
| Step time measured with `cudaEvent` (not `time.time()`) | [ ] |
| GPU-busy % cited per configuration, not only step time | [ ] |
| Peak memory tracked across configurations | [ ] |
| Final behavior-equivalence check (loss / metric vs. baseline) | [ ] |
| Report explicitly names the "bottleneck shift" observed | [ ] |
| No `synchronize` / `.item()` / `print(tensor)` remains in hot loop | [ ] |

## 5. Common mistakes

- **Fixing the kernel first.** The kernel fix looks dramatic in
  isolation (ncu confirms 2× improvement) but moves end-to-end step
  time by near zero because the GPU was idle. Learners who skip the
  isolated-fix rows in the table miss the lesson.
- **Measuring with `time.time()`.** CUDA launches are asynchronous;
  `time.time()` around a `forward` call measures *enqueue* time, not
  compute time.
- **Not synchronizing before taking end time.** Even with
  `cudaEvent`, the final `cuda.synchronize()` is required before
  `elapsed_time` to be meaningful for end-to-end steps.
- **Forgetting `persistent_workers=True`.** Workers restarted every
  epoch amortize their startup cost over the epoch; on short epochs
  this looks like a persistent DataLoader bottleneck even after the
  fix.
- **`pin_memory=True` without `non_blocking=True`** (or vice versa).
  You need both for the HtoD copy to actually overlap with compute.
- **Enabling `torch.compile` and claiming the kernel fix.** Compile
  can fuse ops and change kernel names in the trace; attribute the
  speedup to the fix you tested, not a mix.
- **Ignoring numerical regressions after lowering precision.** bf16
  / fp16 fixes should include a tolerance check on the loss or
  downstream metric.

## 6. References

- PyTorch, *Performance Tuning Guide* (authoritative list of the
  patterns used in this exercise — DataLoader, non-blocking HtoD,
  `set_to_none`, Tensor Core precision, `channels_last`):
  <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html>
- PyTorch, *torch.utils.data.DataLoader* reference
  (`num_workers`, `persistent_workers`, `pin_memory`):
  <https://pytorch.org/docs/stable/data.html>
- PyTorch, *CUDA semantics* (asynchronous execution, streams,
  events — the reason `time.time()` is wrong for GPU timing):
  <https://pytorch.org/docs/stable/notes/cuda.html>
- PyTorch, *torch.compile* overview:
  <https://pytorch.org/docs/stable/generated/torch.compile.html>
- NVIDIA, *Nsight Systems User Guide* (for the per-fix re-traces):
  <https://docs.nvidia.com/nsight-systems/UserGuide/>
- NVIDIA, *Nsight Compute User Guide* (for the per-fix kernel
  verification): <https://docs.nvidia.com/nsight-compute/NsightCompute/index.html>
- Module-level rationale and the "bottleneck shift" pedagogy:
  [`modules/mod-003-performance-profiling/SOLUTION.md`](../SOLUTION.md)
  — *Decision 5: End-to-end optimization as the synthesis*.
- Companion exercises in this module:
  [`exercise-01`](../exercise-01/SOLUTION.md),
  [`exercise-02`](../exercise-02/SOLUTION.md),
  [`exercise-03`](../exercise-03/SOLUTION.md),
  [`exercise-04`](../exercise-04/SOLUTION.md).
