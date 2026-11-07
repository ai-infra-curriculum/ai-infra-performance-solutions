# Exercise 02 — ncu Deep Dive on a Single Kernel

## 1. Solution overview

Once Exercise 01 has identified a top-time kernel, Nsight Compute
(`ncu`) explains *why* that kernel is slow. The reference workflow is
**surgical**: profile a small number of launches of a named kernel,
read the Speed-of-Light (SOL) section to determine whether the kernel
is compute- or memory-bound, then use targeted metrics to find the
specific inefficiency (uncoalesced loads, shared-memory bank
conflicts, low occupancy, warp divergence).

Running `ncu --set full` across a whole training step is almost always
wrong — it is slow, produces tens of megabytes of data, and buries the
answer (see module-level SOLUTION.md, *Decision 2*).

## 2. Worked answer / implementation

### 2.1 Pick the kernel from the nsys short-list

From Exercise 01, you already have the top-time kernel name. Resolve
its unique identifier with:

```bash
ncu --list-sets                 # confirm which sections exist
ncu --query-metrics             # discover metric names
```

### 2.2 Capture a focused report

Profile only the first few launches of the targeted kernel:

```bash
ncu \
  --set full \
  --kernel-name regex:"<kernel_name_substring>" \
  --launch-count 3 \
  --launch-skip 5 \
  --target-processes all \
  --export report \
  python train.py
```

Key flags:

- `--set full` — collect all sections, including SOL, memory workload
  analysis, scheduler stats, and source-level sampling.
- `--kernel-name regex:...` — restrict capture to kernels whose name
  matches. For PyTorch, the names are often mangled (e.g.,
  `at::native::vectorized_elementwise_kernel`), so a short substring
  usually suffices.
- `--launch-count` / `--launch-skip` — skip warm-up launches, capture
  a bounded number.
- `--export report` — write `report.ncu-rep` for later analysis in
  `ncu-ui` or the CLI.

### 2.3 Read the Speed-of-Light first

Open the terminal summary:

```bash
ncu -i report.ncu-rep --section SpeedOfLight --page details
```

SOL reports two headline numbers:

| SOL field | Meaning |
|---|---|
| **SM % of peak** | Fraction of achievable compute throughput used. |
| **Memory % of peak** | Fraction of achievable DRAM bandwidth used. |

Decision rules on an NVIDIA GPU of any recent generation:

- **Compute % high, Memory % low** → compute-bound. Look for FMA /
  tensor-core throughput, warp occupancy, instruction mix.
- **Memory % high, Compute % low** → memory-bound. Look for
  coalescing, cache hit rates, and whether the kernel is fetching
  more bytes than it needs.
- **Both low** → latency-bound or launch-bound. Common with very small
  kernels; usually fixed by batching work, using CUDA Graphs, or
  fusing into a larger kernel.

### 2.4 Targeted follow-up metrics

After SOL narrows the category, pull the metrics that explain *why*:

```bash
ncu -i report.ncu-rep --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_ld,\
smsp__sass_average_data_bytes_per_sector_mem_global_op_st,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st,\
smsp__warps_active.avg.pct_of_peak_sustained_active
```

Interpretation guide:

| Metric | What "bad" looks like | Likely cause / fix |
|---|---|---|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | Very low | Not saturating compute — look elsewhere. |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | ≥ 80 % of peak | Memory-bound; reduce bytes fetched or improve reuse. |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld` | Well below sector size (32 B) | Uncoalesced loads; fix access pattern. |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_*` | Nonzero | Shared-memory bank conflicts; pad arrays or change stride. |
| `smsp__warps_active.avg.pct_of_peak_sustained_active` | Low | Low occupancy; check register/shared-memory usage with `ncu --section LaunchStats`. |

The Speed-of-Light-Roofline section also places the kernel on a
roofline chart, which pairs directly with the roofline work in
`mod-001-gpu-fundamentals`.

### 2.5 Report template

```
Kernel: <name as reported by ncu>
Launches captured: <N>
SM % of peak: <x>
Memory % of peak: <y>
Bound type: <compute | memory | latency>
Primary inefficiency: <uncoalesced loads | bank conflicts |
  low occupancy | warp divergence | other>
Metric evidence: <metric name + value>
Proposed fix: <concrete change to kernel or caller>
```

## 3. Validation steps

1. `ncu --version` prints a Nsight Compute version.
2. `report.ncu-rep` exists and is at least a few hundred kilobytes.
   Empty reports usually mean the kernel never matched the
   `--kernel-name` regex.
3. `ncu -i report.ncu-rep --section SpeedOfLight` prints an SOL
   table with both SM and Memory throughput values.
4. The SOL classification (compute vs. memory) is consistent with
   the follow-up metrics (e.g., a "memory-bound" label is supported by
   a high `dram__throughput` percentage).
5. The report cites at least one metric value, not just a category.

## 4. Rubric / review checklist

| Item | Pass |
|---|---|
| Target kernel came from an Exercise-01 nsys short-list | [ ] |
| `--launch-skip` used to skip warm-up launches | [ ] |
| `--launch-count` kept the capture bounded (not every launch) | [ ] |
| SOL section inspected before anything else | [ ] |
| Classification (compute / memory / latency) stated explicitly | [ ] |
| At least one SOL percentage reported with its metric name | [ ] |
| Follow-up metric evidence matches the classification | [ ] |
| Proposed fix is specific (not "optimize the kernel") | [ ] |
| Report does not generalize from a single launch (≥ 3 captured) | [ ] |

## 5. Common mistakes

- **Running `ncu --set full` on a whole training run.** You will wait
  tens of minutes and produce an unreadable report. Scope with
  `--kernel-name` + `--launch-count`.
- **Profiling a kernel that is 1 % of the step.** `ncu` findings on a
  minor kernel do not move end-to-end latency. Pick from the top of
  the `nsys` short-list.
- **Reading the "Issue Slot Utilization" number as "the GPU is 100 %
  used."** Issue utilization reports how often the scheduler issued
  *any* instruction — it says nothing about math-vs-memory balance.
  Use SOL for that.
- **Fixing shared-memory bank conflicts before confirming the kernel
  is memory-bound.** The textbook optimization often changes nothing
  because the bottleneck was compute all along.
- **Comparing absolute GB/s to the GPU's marketing peak.** `ncu`'s
  "percentage of peak" is relative to *sustained* peak, which is the
  honest ceiling. Do not override it with a datasheet number.
- **Ignoring the "Launch Statistics" section on tiny kernels.** If
  the kernel runs for microseconds and SOL shows both compute and
  memory low, the kernel is launch-bound; batching or CUDA Graphs is
  usually the right answer, not a micro-optimization inside the
  kernel.

## 6. References

- NVIDIA, *Nsight Compute User Guide*:
  <https://docs.nvidia.com/nsight-compute/NsightCompute/index.html>
- NVIDIA, *Nsight Compute Metrics Reference*:
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-reference>
- NVIDIA, *Kernel Profiling Guide* (Speed-of-Light, memory workload
  analysis, roofline):
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>
- Module-level rationale for surgical `ncu` use:
  [`modules/mod-003-performance-profiling/SOLUTION.md`](../SOLUTION.md)
  — *Decision 2: ncu used surgically, not broadly*.
- Pairs with `mod-001-gpu-fundamentals` roofline exercises.
