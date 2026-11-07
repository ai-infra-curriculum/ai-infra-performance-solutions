# Exercise 02 — Stream Overlap (Compute / Transfer Pipelining)

## 1. Solution overview

Use multiple CUDA streams so the next batch's **host-to-device
copy** runs concurrently with the current batch's **forward/
backward compute**, instead of the two phases running serially on
the default stream. The learner wires up a prefetch stream and a
compute stream, synchronizes them with explicit events (or
`wait_stream`), and demonstrates a measurable step-time reduction
on a workload where the HtoD copy is a visible fraction of the
step.

Two ideas the exercise is really testing:

1. **Streams give concurrency, not parallelism of the same kind as
   threads.** Work issued on different CUDA streams may execute
   concurrently on the GPU if resources allow (copy engines +
   compute SMs are separate units on NVIDIA GPUs). This is why
   HtoD-copy / compute overlap works at all — the copy engine
   drives the PCIe/NVLink transfer while the SMs run the kernel.
2. **Overlap only helps where there is slack to hide.** If compute
   per batch already dwarfs the transfer, the "overlap" saves
   nothing — the bottleneck is the compute stream. The learner is
   expected to measure before claiming a win.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md), Decision 2 ("Stream overlap —
concurrency without parallelism"). The stub for this exercise lives
in [`README.md`](./README.md).

## 2. Worked answer or implementation

### 2.1 Reference pattern

The pattern below separates **prefetch** (HtoD copy for batch N+1)
from **compute** (forward/backward for batch N). The two streams
share data via an event: compute waits for the prefetch of the
batch it is about to consume.

```python
# modules/mod-008-advanced-topics/exercise-02/stream_overlap.py
import torch


def run_overlapped(model, loader, device="cuda"):
    """Train one epoch with next-batch HtoD overlapping current compute.

    Assumes `loader` yields CPU tensors in pinned memory so that
    `.to(device, non_blocking=True)` is actually asynchronous.
    """
    copy_stream    = torch.cuda.Stream(device=device)
    compute_stream = torch.cuda.Stream(device=device)

    it = iter(loader)
    try:
        first_cpu = next(it)
    except StopIteration:
        return

    # Prefetch the first batch onto the copy stream.
    with torch.cuda.stream(copy_stream):
        current_gpu = first_cpu.to(device, non_blocking=True)
    copy_done = torch.cuda.Event()
    copy_done.record(copy_stream)

    for next_cpu in it:
        # Issue the HtoD for batch N+1 on the copy stream.
        with torch.cuda.stream(copy_stream):
            next_gpu = next_cpu.to(device, non_blocking=True)
        next_copy_done = torch.cuda.Event()
        next_copy_done.record(copy_stream)

        # Compute on batch N — but only after its copy finished.
        compute_stream.wait_event(copy_done)
        with torch.cuda.stream(compute_stream):
            out  = model(current_gpu)
            loss = out.sum()
            loss.backward()

        # Rotate: batch N+1 becomes the new "current".
        current_gpu = next_gpu
        copy_done   = next_copy_done

    # Drain the last batch.
    compute_stream.wait_event(copy_done)
    with torch.cuda.stream(compute_stream):
        out = model(current_gpu)
        loss = out.sum()
        loss.backward()

    torch.cuda.current_stream().wait_stream(compute_stream)
```

Key points:

- `torch.cuda.Stream()` creates a non-default stream. Operations
  issued inside `torch.cuda.stream(s)` are enqueued on `s` rather
  than on the current (default) stream.
- `copy_stream.record(event)` / `compute_stream.wait_event(event)`
  make the dependency explicit. Without this, the compute kernel
  may start before the copy finishes and read uninitialized
  memory. (See PyTorch CUDA semantics: cross-stream dependencies
  must be stated with events/`wait_stream`.)
- `non_blocking=True` is only meaningful when the source tensor
  lives in **pinned** host memory. For a `DataLoader`, set
  `pin_memory=True`. Without pinning, the copy degenerates to a
  synchronous one and the overlap disappears.
- The driver code at module end (`current_stream().wait_stream(compute_stream)`)
  ensures any downstream code (optimizer step, logging) does not
  race ahead of the final backward.

### 2.2 Measurement harness

A benchmark that reports both variants side-by-side on the same
model and data; the learner should produce something equivalent.

```python
# modules/mod-008-advanced-topics/exercise-02/bench_overlap.py
import time, torch

def bench(fn, iters=50, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / iters  # ms/iter


def serial_step(model, cpu_batch, device):
    gpu = cpu_batch.to(device, non_blocking=True)
    out = model(gpu); out.sum().backward()


def overlapped_step(model, cpu_a, cpu_b, device, copy_s, compute_s):
    with torch.cuda.stream(copy_s):
        b_gpu = cpu_b.to(device, non_blocking=True)
    evt = torch.cuda.Event(); evt.record(copy_s)
    compute_s.wait_event(evt)
    with torch.cuda.stream(compute_s):
        a_gpu = cpu_a.to(device, non_blocking=True)
        out = model(a_gpu); out.sum().backward()
```

What the learner should **report**, not just produce:

- Model and batch shape.
- Whether `pin_memory=True` was used.
- Per-iter ms, serial vs overlapped, with the number of warmup
  iterations and the number of measured iterations.
- The ratio of transfer time to compute time (profile it with
  Nsight Systems or the PyTorch profiler). The expected overlap
  speedup is bounded by **min(transfer_time, compute_time)** — the
  slower of the two sets the floor. If compute ≫ transfer, the
  speedup is small and that is the correct answer.

### 2.3 How to confirm the overlap actually happened

A green "I got a speedup" is not sufficient — the speedup could
come from caching or a different kernel schedule. Verify the
overlap directly:

- **Nsight Systems (`nsys profile`)**: the timeline should show
  copy-engine activity and SM kernel activity on different rows
  with visible time-overlap bars. No overlap on the timeline = no
  overlap in practice, regardless of what wall-clock shows.
- **PyTorch profiler with CUDA activity**: export the trace
  (`torch.profiler.profile(..., with_stack=True)`) and open it in
  `chrome://tracing`. The `memcpy_h2d` events should overlap the
  kernel events.

## 3. Validation steps

1. **Dependencies and environment**
   - `python -c "import torch; print(torch.cuda.is_available())"` →
     `True`.
   - GPU has at least one copy engine (all modern NVIDIA GPUs do;
     H100 / A100 have multiple) — otherwise stream overlap for
     HtoD cannot happen regardless of code.
2. **Static review of the implementation**
   - Two distinct `torch.cuda.Stream()` instances exist (copy and
     compute).
   - Every cross-stream read is guarded by `record` + `wait_event`
     or an explicit `wait_stream`.
   - `DataLoader` (or the equivalent host batch source) uses
     `pin_memory=True`.
   - The final compute result is awaited before the next epoch /
     optimizer step (no silent races).
3. **Correctness**
   - Train for a fixed seed for the same number of iterations in
     both the serial and overlapped variants. Final loss values
     should match to within numerical noise. A divergence means a
     missing sync — the overlapped run is using stale memory.
4. **Performance**
   - Overlapped `ms/iter` is ≤ serial `ms/iter` on at least one
     workload the learner profiled. The learner reports the ratio
     and, if the speedup is near-zero, explains it (compute-bound
     workload — expected).
5. **Overlap evidence**
   - Learner shows a profiler screenshot or exported trace with
     copy and compute on overlapping time bars.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | Two distinct streams | Code creates `Stream()` for copy and another for compute; default stream not used for either phase |
| 2 | Explicit cross-stream sync | `record` + `wait_event` (or `wait_stream`) before the compute reads the prefetched tensor |
| 3 | Pinned host memory | DataLoader / source tensors are pinned; `non_blocking=True` is used in `.to(device)` |
| 4 | Both variants benchmarked | Same model, same data, same iter count, same warmup, reported side-by-side |
| 5 | Overlap verified visually | Learner attaches a profiler trace (Nsight Systems or PyTorch profiler) showing concurrent copy + compute |
| 6 | Correctness preserved | Serial and overlapped runs produce matching loss trajectories under a fixed seed |
| 7 | Honest reporting | If compute ≫ transfer and the overlap bought nothing, the learner states this rather than fabricating a speedup |
| 8 | Final-batch drain | Epoch end awaits the last compute (`wait_stream(compute_stream)`) before optimizer / logging |

Scoring guidance:

- 7–8 passes: understands CUDA concurrency model; move on to
  exercise 03 (NCCL tests).
- 5–6 passes: implementation is right but the measurement is
  weak — have the learner redo the benchmark with warmup and
  attach the profile.
- ≤ 4 passes: stream semantics not yet internalized (usually
  a missing event or unpinned memory). Send them back to the
  PyTorch CUDA-semantics docs before signing off.

## 5. Common mistakes

1. **Non-pinned host memory.** `.to(device, non_blocking=True)`
   silently becomes blocking when the source is pageable memory.
   The overlap collapses and the benchmark shows "no difference."
   Set `pin_memory=True` on the `DataLoader` or allocate with
   `torch.empty(..., pin_memory=True)`.
2. **No cross-stream synchronization.** Reading the prefetched
   tensor on the compute stream without a `wait_event` /
   `wait_stream` races against the copy. Symptom: intermittent
   NaNs, wrong losses, or garbage outputs. The code may "work"
   on a small example and fail on a larger one.
3. **Using the default stream for one phase.** The default stream
   has implicit synchronization behavior that defeats overlap in
   several configurations. Put **both** phases on explicit
   non-default streams.
4. **Claiming a speedup without profiling.** A wall-clock
   improvement can come from kernel caching, JIT warmup, or
   allocator reuse — not the overlap. The timeline in Nsight
   Systems is the ground truth for whether copy and compute ran
   concurrently.
5. **Forgetting the final drain.** The last iteration's compute
   runs on `compute_stream`, but the next thing that runs
   (optimizer step, metric logging) may be on the default stream.
   Without `current_stream().wait_stream(compute_stream)` the
   optimizer may read stale gradients.
6. **Capturing the pipeline inside a CUDA Graph without care.**
   Graph capture is sensitive to multi-stream event dependencies.
   The module-level SOLUTION notes that mixing `torch.compile` and
   graph capture is fragile; the same caveat applies to capturing
   a multi-stream pipeline (see module-level SOLUTION, Decision 1).

## 6. References

- PyTorch "CUDA semantics" — streams, events, and when
  `non_blocking=True` is actually non-blocking:
  https://pytorch.org/docs/stable/notes/cuda.html
- PyTorch API reference — `torch.cuda.Stream`,
  `torch.cuda.Event`, `torch.cuda.stream` context manager:
  https://pytorch.org/docs/stable/cuda.html
- NVIDIA CUDA C++ Programming Guide — asynchronous concurrent
  execution, streams, and events (the primitives PyTorch wraps):
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- Nsight Systems user guide — reading the copy-engine and SM
  activity rows to confirm overlap:
  https://docs.nvidia.com/nsight-systems/
- PyTorch Profiler recipe — exporting a trace for Chrome
  `tracing` viewing:
  https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
- Local module context —
  [`modules/mod-008-advanced-topics/SOLUTION.md`](../SOLUTION.md),
  Decision 2 ("Stream overlap — concurrency without
  parallelism") and the module-level anti-pattern list on missing
  synchronization.
- Local stub —
  [`modules/mod-008-advanced-topics/exercise-02/README.md`](./README.md)
  shows the minimal snippet this solution elaborates.
