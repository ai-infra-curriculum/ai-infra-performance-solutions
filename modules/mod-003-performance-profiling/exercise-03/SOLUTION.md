# Exercise 03 — PyTorch Profiler and TensorBoard

## 1. Solution overview

`torch.profiler.profile` instruments a PyTorch program at the ATen /
autograd / Python-op boundary. Unlike `nsys`, which sees CUDA streams
and kernels, the PyTorch Profiler sees calls such as `aten::matmul` or
`aten::index_select` and attaches a Python stack to each call. That
makes it the right first lens for *ML* workloads: it answers "which
op is dominant" before you need to know anything about CUDA kernels.

The reference workflow (see also `exercise-03/profile_script.py`)
emits traces in two forms simultaneously:

1. A **TensorBoard** trace directory so the trace opens in the
   PyTorch Profiler plugin.
2. A **Chrome tracing JSON**, usable in `chrome://tracing`,
   [Perfetto UI](https://ui.perfetto.dev/), or any viewer that reads
   the Chrome Event Format. This is important for environments where
   `nsys` is not installed (CI runners, cloud notebooks).

## 2. Worked answer / implementation

### 2.1 Reference skeleton

The file `exercise-03/profile_script.py` contains the minimal skeleton.
The expanded reference pattern is:

```python
import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)


def profile_step(model, data_iter, optimizer, device):
    # schedule: skip the first iteration (lazy init), warm up for a few,
    # then record a short active window, repeated once.
    prof_schedule = schedule(
        skip_first=1, wait=1, warmup=2, active=3, repeat=1,
    )

    def on_ready(prof):
        # TensorBoard directory handler.
        tensorboard_trace_handler("./profiler_log")(prof)
        # Chrome tracing JSON for environments without TensorBoard.
        prof.export_chrome_trace("./profiler_log/chrome_trace.json")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=on_ready,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # enable only when hunting a specific hotspot
    ) as prof:
        for step, batch in enumerate(data_iter):
            batch = batch.to(device, non_blocking=True)
            out = model(batch)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            prof.step()  # advance the schedule state machine
            if step >= prof_schedule.num_steps - 1:
                break

    # Steady-state summary table for the README/report.
    print(
        prof.key_averages(group_by_input_shape=True)
        .table(sort_by="cuda_time_total", row_limit=10)
    )
```

Design notes:

- **`schedule(...)`** is the mechanism that makes the profiler usable
  on real workloads. Without a schedule, every step is recorded and
  the trace balloons. The `wait → warmup → active → repeat` state
  machine captures a bounded window.
- **`record_shapes=True`** makes the summary table group by tensor
  shape — essential when the same op is called with many shapes.
- **`profile_memory=True`** adds per-op allocator deltas (useful
  pointer to Exercise 04).
- **`with_stack=False`** by default. Enabling it adds 5–20 % overhead
  because each op records a Python stack; turn it on only when you
  need the source line for a specific hotspot.

### 2.2 Viewing the trace

**TensorBoard**:

```bash
pip install torch-tb-profiler
tensorboard --logdir=./profiler_log
```

The *PyTorch Profiler* tab contains:

- **Overview** — step-time breakdown into Kernel / Memcpy / Memset /
  Runtime / DataLoader / CPU exec. The first screen that tells you
  whether the workload is GPU-bound or CPU-bound.
- **Operator view** — per-op CPU and CUDA time, aggregated by shape.
- **Kernel view** — per-kernel GPU time, equivalent to the top of a
  `nsys` kernel summary.
- **Trace view** — timeline, equivalent to `chrome://tracing`.
- **Memory view** — timeline of allocator usage (requires
  `profile_memory=True`).

**Chrome tracing / Perfetto**:

```text
chrome://tracing → Load → ./profiler_log/chrome_trace.json
# or
https://ui.perfetto.dev → Open trace file → chrome_trace.json
```

### 2.3 Reading the output

The steady-state `key_averages` table is sorted by
`cuda_time_total`. The columns that matter:

- **Self CUDA %** — fraction of GPU time spent *inside* the op, not
  counting children. The right column to sort by when hunting
  dominant ops.
- **CUDA total %** — fraction including children (useful on
  autograd nodes that wrap many sub-ops).
- **CPU total** — if this is close to the step time while CUDA total
  is small, you are CPU-bound.
- **Input Shapes** — present when `record_shapes=True`. Reveals
  whether a single op is slow because of a specific shape (e.g., an
  embedding lookup with a huge vocab).

### 2.4 Report template

```
Workload: <model, batch size, device>
Schedule: skip_first=1 wait=1 warmup=2 active=3 repeat=1
Captured steps: <active window description>
Top 3 ops by Self CUDA time:
  1. <op name> <shape> — <ms> (<%>)
  2. <op name> <shape> — <ms> (<%>)
  3. <op name> <shape> — <ms> (<%>)
Step-time breakdown (TensorBoard Overview):
  Kernel: <%>  Memcpy: <%>  DataLoader: <%>  CPU exec: <%>  Other: <%>
Classification: <GPU-bound | CPU/DataLoader-bound | mixed>
Hand-off to next tool: <nsys step profile | ncu on kernel X | fix DataLoader>
```

## 3. Validation steps

1. `python -c "import torch, torch.profiler"` succeeds.
2. After running the script, `./profiler_log/` contains at least one
   `*.pt.trace.json` TensorBoard file **and** `chrome_trace.json`.
3. `tensorboard --logdir=./profiler_log` serves the *PyTorch
   Profiler* tab without "no data" errors. (If missing, install
   `torch-tb-profiler`.)
4. The TensorBoard **Overview** page shows a non-empty step-time
   breakdown pie chart.
5. The `key_averages` printed table lists at least one op with
   nonzero `cuda_time_total`.

## 4. Rubric / review checklist

| Item | Pass |
|---|---|
| `schedule(...)` is used (not per-step unbounded profiling) | [ ] |
| At least one iteration is skipped before `active` window | [ ] |
| Both TensorBoard and Chrome traces are emitted | [ ] |
| `record_shapes=True` in the submitted config | [ ] |
| `with_stack=True` is **off** for the headline measurement | [ ] |
| Top-3 ops reported with shapes and % of step time | [ ] |
| TensorBoard Overview breakdown cited in the write-up | [ ] |
| Classification handed off to the correct next tool | [ ] |
| No `torch.cuda.synchronize()` inside the timed loop | [ ] |

## 5. Common mistakes

- **Profiling without a schedule.** The trace covers every step,
  including startup, and is dominated by warm-up noise.
- **Treating `cpu_time_total` as "wall clock on CPU."** CUDA launches
  are asynchronous; `cpu_time_total` is time spent in the op's CPU
  dispatch, not time spent waiting for the kernel.
- **Leaving `with_stack=True` on for headline numbers.** Stack
  capture adds material overhead and distorts the very numbers you
  are reporting.
- **Measuring with `time.time()` around kernels.** Kernels are
  asynchronous. Use `torch.cuda.Event` with `record()` /
  `elapsed_time()` (after `synchronize()`) for ad-hoc timing, or let
  the profiler handle it.
- **Ignoring the DataLoader bucket in the Overview tab.** On medium
  GPUs, DataLoader dominates most "slow training" reports. Check it
  before descending into kernel-level tools.
- **Not calling `prof.step()`.** Without `prof.step()` the scheduler
  stays in its initial phase and no trace is emitted.

## 6. References

- PyTorch, *torch.profiler API reference*:
  <https://pytorch.org/docs/stable/profiler.html>
- PyTorch, *Profiling your PyTorch Module* tutorial:
  <https://pytorch.org/tutorials/beginner/profiler.html>
- PyTorch, *PyTorch Profiler With TensorBoard* tutorial:
  <https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html>
- `torch-tb-profiler` TensorBoard plugin:
  <https://github.com/pytorch/kineto/tree/main/tb_plugin>
- Perfetto UI (Chrome Event Format viewer):
  <https://ui.perfetto.dev/>
- Module-level rationale for using the PyTorch Profiler as the
  "Python boundary" tool:
  [`modules/mod-003-performance-profiling/SOLUTION.md`](../SOLUTION.md)
  — *Decision 3: PyTorch Profiler as the Python boundary tool*.
- Local skeleton:
  [`exercise-03/profile_script.py`](profile_script.py).
