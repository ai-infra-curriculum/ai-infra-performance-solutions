# Exercise 01 — nsys Trace and Bottleneck Identification

## 1. Solution overview

The goal of this exercise is to produce a Nsight Systems (`nsys`)
timeline of a training or inference step and use it to classify the
workload's bottleneck into one of four buckets: **host-bound**,
**memory-copy-bound**, **compute-bound**, or **synchronization-bound**.

The reference workflow deliberately starts with a system-wide timeline
rather than per-kernel metrics. A timeline is the only artifact that
shows kernels, CPU work, and memory copies on a common time axis, so
it is the correct first lens when "we don't yet know what's slow"
(see the module-level SOLUTION.md, *Decision 1*).

## 2. Worked answer / implementation

### 2.1 Capture the trace

Profile a representative script after warm-up. Trace only a handful
of steady-state steps so the resulting `.nsys-rep` stays small enough
to open in the GUI.

```bash
# One-shot capture of a training step.
nsys profile \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
  --output=trace \
  --force-overwrite=true \
  --capture-range=nvtx \
  --capture-range-end=stop \
  python train.py
```

If the script does not yet emit NVTX ranges, drop `--capture-range` and
instead start/stop the collector with `--delay=10 --duration=5` so the
trace covers 5 seconds of steady state after a 10 s warm-up.

Add NVTX markers around the step boundary to make the timeline readable:

```python
import torch.cuda.nvtx as nvtx

for step, batch in enumerate(loader):
    nvtx.range_push(f"step_{step}")
    out = model(batch)
    loss = criterion(out)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    nvtx.range_pop()
```

### 2.2 Summarize from the CLI

Before opening the GUI, triage from the terminal:

```bash
nsys stats \
  --report cuda_gpu_kern_sum \
  --report cuda_gpu_mem_time_sum \
  --report cuda_api_sum \
  trace.nsys-rep
```

What each report answers:

- `cuda_gpu_kern_sum` — which kernels spent the most GPU time (the
  short-list for Exercise 02).
- `cuda_gpu_mem_time_sum` — total time in `cudaMemcpy*` operations.
  If this is more than ~10–15 % of the step, the workload is
  transfer-bound.
- `cuda_api_sum` — time spent in CUDA API calls on the host, including
  `cudaStreamSynchronize`, `cudaDeviceSynchronize`, and
  `cudaMemcpyAsync`. Large synchronize time means the host is waiting
  on the device (or vice-versa).

### 2.3 Classify the bottleneck

Open the trace with `nsys-ui trace.nsys-rep` and compare the timeline
against the four signatures:

| Signature in timeline | Classification | Next step |
|---|---|---|
| Gaps between kernels; CPU row shows Python frames filling the gap | **Host-bound** | Fix DataLoader / Python op; re-trace. |
| Fat HtoD / DtoH bars dominate; kernels wait on copies | **Transfer-bound** | Move preprocessing to GPU, pin memory, use `non_blocking=True`, or overlap with CUDA streams. |
| Kernels back-to-back with little white space between them | **Compute-bound** | Pick the top-time kernel and go to Exercise 02 (ncu deep dive). |
| Many short kernels followed by tall `cudaStreamSynchronize` / `cudaDeviceSynchronize` bars | **Sync-bound** | Audit for hidden `.item()`, `.cpu()`, `print(tensor)`, `torch.cuda.synchronize()` in the hot path. |

### 2.4 Report template

The expected learner deliverable is a short classification note. A
passing write-up contains:

```
Workload: <script, model, batch size, GPU>
Warm-up / measured steps: <e.g., skip 3 steps, measure 5>
Top-3 kernels by GPU time: <from cuda_gpu_kern_sum>
HtoD + DtoH time per step: <ms> (<%> of step)
CPU-side gap time per step: <ms> (<%> of step)
Classification: <host / transfer / compute / sync>
Rationale: <one sentence tying the timeline evidence to the label>
Next action: <re-trace / ncu kernel X / fix DataLoader / etc.>
```

## 3. Validation steps

1. `nsys --version` prints a Nsight Systems version. If the command is
   missing, install the CUDA Toolkit or the standalone Nsight Systems
   package.
2. `trace.nsys-rep` exists and is at least a few hundred kilobytes.
   Empty or 0-byte traces usually mean the target exited before the
   collector attached — rerun without `--capture-range` to confirm.
3. `nsys stats --report cuda_gpu_kern_sum trace.nsys-rep` produces a
   non-empty kernel table. An empty table means no CUDA kernels ran
   during the capture window (common with `--capture-range=nvtx` if
   NVTX ranges are missing).
4. Opening in `nsys-ui` shows at least one CUDA HW row with kernels
   and one CUDA API row with host-side calls.
5. The classification label in the report matches the timeline
   evidence cited (grader sanity check).

## 4. Rubric / review checklist

| Item | Pass |
|---|---|
| Trace was captured after warm-up (not iteration 0) | [ ] |
| Trace covers multiple steps or ≥ 1 s of steady state | [ ] |
| NVTX step boundaries make the timeline readable | [ ] |
| At least one `nsys stats` report cited in the write-up | [ ] |
| Top-3 kernels listed with their share of GPU time | [ ] |
| HtoD/DtoH time and CPU-gap time estimated numerically | [ ] |
| Bottleneck classified using the four-bucket taxonomy | [ ] |
| "Next action" step named and tied to the evidence | [ ] |
| `nvidia-smi` "GPU utilization" is **not** cited as evidence | [ ] |

## 5. Common mistakes

- **Tracing iteration 0.** Lazy module loads, cuDNN benchmark search,
  and JIT compilation make the first iteration unrepresentative. Add
  a warm-up loop and start the collector after it.
- **Running `nsys` over an entire epoch.** Multi-gigabyte traces that
  fail to open. Capture 5–10 steady-state steps instead.
- **Treating `nvidia-smi`'s utilization column as a bottleneck
  signal.** It reports the fraction of time *any* SM was active, not
  how much of the GPU was in use. A single-SM kernel reads as 100 %.
- **Jumping to `ncu` without a timeline.** You end up optimizing
  kernels that are 1 % of the runtime. Timeline first, deep dive
  second.
- **Forgetting `--trace=osrt`.** Without OS runtime trace you lose
  visibility into CPU-side waits and can't diagnose host-bound
  workloads.

## 6. References

- NVIDIA, *Nsight Systems User Guide* — overview, CLI flags, and
  report list: <https://docs.nvidia.com/nsight-systems/UserGuide/>
- NVIDIA, *Nsight Systems — Profiling from the CLI* (nsys profile /
  nsys stats): <https://docs.nvidia.com/nsight-systems/UserGuide/index.html#cli-profiling>
- NVIDIA, *NVTX for CUDA C++ and Python* (range annotations used in
  this exercise): <https://docs.nvidia.com/cuda/profiler-users-guide/index.html#nvtx>
- Module-level rationale for timeline-first profiling:
  [`modules/mod-003-performance-profiling/SOLUTION.md`](../SOLUTION.md)
  — *Decision 1: nsys timeline as the first artifact*.
