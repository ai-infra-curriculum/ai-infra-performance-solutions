# SOLUTION — Performance Profiling

> Read this *after* you have profiled a real workload end-to-end.
> The exercise solutions show how the tools work; this document
> explains *why* the profiling workflow is shaped the way it is and
> which signals matter when you have ten minutes to find the
> bottleneck.

## What this module is really teaching

Profiling on a GPU is not "run perf and read the flamegraph." It
is a **layered investigation** that switches tools based on what
you suspect:

| Symptom | Right tool |
|---|---|
| "I don't know what's slow" | nsys (timeline) |
| "Kernel X is slow, but why?" | ncu (per-kernel deep dive) |
| "Python is slow" | PyTorch Profiler |
| "Out of memory" | torch.cuda.memory snapshot |
| "Training and inference both slow" | end-to-end (e2e) profile |

The mistake juniors make is reaching for `ncu` first (because the
flamegraphs look impressive) and burning two hours on a kernel
that turns out to be 1% of the runtime. The reference solutions
enforce the order: **timeline first, then deep dive**.

## Architectural decisions and *why*

### Decision 1: nsys timeline as the first artifact (always)

Exercise 01 (nsys trace) is intentionally placed before all other
tools. The reason: a 30-second timeline tells you whether the
bottleneck is:

- **Host-side** (CPU is slow, GPU is starving) — show as gaps
  between kernel launches.
- **Memory-bound** (HtoD/DtoH copies dominate) — visible as fat
  copy bars on the memory rows.
- **Compute-bound** (kernels back-to-back, no gaps) — go to ncu.
- **Synchronization-bound** (lots of `cudaStreamSynchronize` waits)
  — usually a Python-side `.item()` or `.cpu()` hidden in a hot
  path.

Without that bird's-eye view, you'll optimize the wrong layer.

**Anti-pattern to avoid**: profiling for two hours in ncu, finding a
30% improvement on kernel X, and discovering after deployment that
the workload was 80% Python overhead. The 30% gain becomes 3%.

### Decision 2: ncu used surgically, not broadly

Exercise 02 (ncu deep dive) only kicks in after nsys has identified
*which* kernel deserves attention. The reference workflow runs
`ncu --set full -k <kernel-name> -c 1` to profile a single launch
of the targeted kernel. Running `ncu --set full` over an entire
training step takes 30-60 minutes and produces 50 MB of data — most
of which is irrelevant to the question you started with.

The deep-dive metrics that matter:
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — compute
  utilization.
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — memory
  utilization.
- `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld`
  — memory access pattern (uncoalesced loads kill performance).
- `l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st` — shared
  memory bank conflicts.

Five metrics tell you 90% of what you need. The full report is for
fine-tuning, not for finding the bottleneck.

### Decision 3: PyTorch Profiler as the Python boundary tool

Exercise 03 (PyTorch profiler) lives at a different abstraction
level than nsys. nsys sees CUDA streams; PyTorch profiler sees
`aten::matmul` and `aten::index_select` calls. For ML workloads,
the right diagnostic is usually:

1. PyTorch profiler ⇒ which op is dominant?
2. nsys ⇒ what is that op actually doing under the hood?
3. ncu ⇒ if the kernel is bad, why?

The reference solution emits the profiler trace as a Chrome
tracing JSON (`prof.export_chrome_trace("trace.json")`) so you can
load it in `chrome://tracing` or Perfetto. This works in
environments where you can't run nsys (CI runners, cloud notebooks)
and is the right "first profile" for most students.

### Decision 4: Memory snapshot as a separate skill

Exercise 04 (memory snapshot) uses
`torch.cuda.memory._record_memory_history()`. This is the **only**
tool that explains *why* you OOM'd at step 4231 when the model fits
in memory. The reference solution captures the snapshot before the
OOM, then visualizes it with `torch.cuda._snapshot()` to find:

- Memory fragmentation (small free chunks adding up to "enough"
  memory but no contiguous block).
- Tensor lifetime issues (activations retained across iterations
  by accident).
- Optimizer-state explosion (Adam has 2x model size in optimizer
  state; with mixed precision, double again).

This is its own exercise because the workflow is genuinely
different from kernel-level profiling — you're looking for what's
holding memory across time, not what's slow right now.

### Decision 5: End-to-end optimization as the synthesis

Exercise 05 (end-to-end optimization) deliberately presents a real
training loop with three independent bottlenecks layered on top of
each other:

1. DataLoader is slow (host-side starvation).
2. Forward pass has an uncoalesced load.
3. Backward pass has an unnecessary host sync.

The student must fix them **in the right order** — fixing the
kernel before the DataLoader produces 0% improvement because the
GPU was already idle 40% of the time. The reference solution emits
a "speedup table" showing the contribution of each fix in
isolation, so the lesson sticks: **the bottleneck shifts as you
fix things**, and you need to re-profile after each change.

## Trade-offs we deliberately accepted

### nsys focused on a single training step

Tracing a full epoch produces unreadable timelines and 1-2 GB
files. The reference workflow traces 5-10 steps (after warm-up),
which is enough to see steady-state behavior and small enough to
open in the nsys UI.

### No off-the-shelf profiling dashboard

Tools like Weights & Biases System Metrics or TensorBoard's
profile tab cover similar ground with prettier UIs. The reference
solution sticks to nsys/ncu/PyTorch Profiler because those
artifacts are what NVIDIA's own engineers look at when triaging
performance bugs. The dashboards are downstream consumers; learning
the source first means the dashboards make sense later.

### Linux assumed; macOS skipped

CUDA profiling on macOS is dead since CUDA 11.x. The reference
workflow assumes Linux + CUDA 12.x. Students on macOS run the
exercises in a CUDA container or a cloud GPU box.

## Common mistakes graders see

1. **Profiling cold cache**: the first iteration includes lazy
   module loads, JIT compilation, and cuDNN benchmark search.
   Always profile after a warm-up loop.
2. **Treating CUDA Graphs as a free speedup**: graphs only help
   when launch overhead is significant (small ops, high batch
   count). Wrapping a forward pass that already amortizes launches
   well buys nothing.
3. **Ignoring host-side overhead**: 90% of slow training loops on
   medium GPUs are bottlenecked on the DataLoader or `.item()` /
   `.cpu()` calls. Always check the host bar in nsys.
4. **Profiling with `torch.profiler.with_stack=True` and complaining
   about overhead**: stack traces have a 5-20% perf cost. Turn them
   off for steady-state measurement.
5. **Using `time.time()` for kernel timing**: kernel launches are
   asynchronous. Use `torch.cuda.Event` with `record()` /
   `elapsed_time()` or `cudaEventRecord` directly.
6. **Reading "100% GPU utilization" from nvidia-smi as success**:
   nvidia-smi's utilization metric is "fraction of time *any* SM
   was active." A kernel that uses one SM out of 132 reads as
   100%.

## When to go beyond this implementation

- Pipe nsys output into a roofline tool to mark each kernel's
  position on the chart (mod-001 ex-04).
- Add **continuous profiling** in production (Parca, Pyroscope) for
  steady-state regression detection.
- Build a **per-step trace diff** (today's profile vs. yesterday's)
  to catch performance regressions on the same workload — a CI gate
  every serious ML platform has.

## Related curriculum touchpoints

- `performance/mod-001-gpu-fundamentals` — the roofline model that
  tells you whether a slow kernel can even *be* faster.
- `performance/mod-007-production-deployment` — production
  profiling and SLO-based regression detection.
- `engineer/mod-107-gpu-computing/exercise-07-gpu-memory-profiling`
  — the working scripts and dashboards.
- `engineer/mod-108-monitoring-observability/exercise-02-ml-model-monitoring`
  — turning steady-state profile artifacts into alerts.
