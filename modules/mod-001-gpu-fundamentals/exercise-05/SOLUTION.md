# Exercise 05 — Warp divergence analyzer (SOLUTION)

> Paired learning file: `ai-infra-performance-learning/modules/mod-001-gpu-fundamentals/exercises/exercise-05-warp-divergence-analyzer`
> Module rationale: see [`../SOLUTION.md`](../SOLUTION.md) — "Decision 2: Warp divergence treated as a first-class metric".

## 1. Solution overview

A **warp** is a group of 32 threads on NVIDIA GPUs that execute
in lockstep (SIMT). When threads in the same warp take different
branches of a conditional, the hardware serializes the paths —
each path runs with the inactive threads masked off. This is
**warp divergence**, and it directly reduces the useful throughput
of the warp scheduler. See the CUDA C++ Programming Guide,
"SIMT Architecture" section (linked in §6).

The exercise asks learners to build an *analyzer* — a tool that,
given a kernel, identifies which branches are likely to diverge
and quantifies the divergence. The reference solution combines
two complementary approaches:

1. **Static analysis**: scan the kernel source for conditional
   expressions and classify whether each condition is likely
   warp-uniform or lane-dependent (divergent).
2. **Dynamic measurement**: run Nsight Compute on the kernel and
   extract the divergence metrics the profiler records.

The module SOLUTION.md deliberately keeps this exercise mechanical
— "counting divergent paths is more important than fancy
visualization at this stage".

### Classification: this is an implementation + analysis exercise

The learner must produce a working analyzer (static or
profiler-driven, or both) plus a short written interpretation
explaining what counts as divergent and why.

## 2. Worked answer or implementation

### 2.1 Reference worked example: a kernel that diverges and one that doesn't

```cuda
// diverges: branch depends on the lane ID within the warp.
__global__ void divergent_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((threadIdx.x & 1) == 0) {       // even lanes
        y[i] = x[i] * 2.0f;
    } else {                            // odd lanes
        y[i] = x[i] + 1.0f;
    }
}

// warp-uniform: branch depends on blockIdx (constant across the warp).
__global__ void uniform_kernel(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if ((blockIdx.x & 1) == 0) {        // whole-block decision
        y[i] = x[i] * 2.0f;
    } else {
        y[i] = x[i] + 1.0f;
    }
}
```

The two kernels compute the same per-element result but differ
dramatically in divergence behavior: the first splits every warp
in half (two serialized passes); the second commits every warp
uniformly to one branch.

### 2.2 Static analyzer — reference implementation

The static analyzer is a small Python tool that reads a `.cu`
file, finds `if` conditions, and classifies each by what the
condition depends on. The intent is pedagogical — a real
production tool would parse a proper AST (e.g. Clang/libclang)
rather than use regexes.

```python
"""warp_divergence_analyzer.py — pedagogical static analyzer.

Scans a CUDA source file for conditional expressions and flags
those likely to cause warp divergence based on which per-thread
identifiers appear in the condition.

Heuristic (documented in CUDA C++ Programming Guide §SIMT):
- threadIdx.x / threadIdx.y / threadIdx.z are per-thread values.
  Lane ID (threadIdx.x & 31) differs within a warp -> divergent.
- blockIdx.*, gridDim.*, blockDim.* are warp-uniform within a
  single warp -> not divergent (whole warp takes one path).
- Values loaded from global memory can go either way; flag as
  "data-dependent, unknown".
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


# Order matters: the first regex that matches wins.
LANE_DEPENDENT = re.compile(r"\bthreadIdx\.(x|y|z)\b")
WARP_UNIFORM = re.compile(r"\b(blockIdx|blockDim|gridDim)\.(x|y|z)\b")
DATA_DEPENDENT = re.compile(r"\b[A-Za-z_][A-Za-z_0-9]*\s*\[")


@dataclass
class Finding:
    line: int
    condition: str
    verdict: str          # "divergent" | "uniform" | "data-dependent"
    reason: str


def classify(condition: str) -> tuple[str, str]:
    if LANE_DEPENDENT.search(condition):
        return "divergent", "condition references threadIdx; lanes differ within a warp"
    if WARP_UNIFORM.search(condition):
        return "uniform", "condition references only block/grid identifiers; warp-uniform"
    if DATA_DEPENDENT.search(condition):
        return "data-dependent", "condition reads a value; cannot statically prove uniformity"
    return "data-dependent", "no recognized SIMT identifier; cannot statically classify"


IF_RE = re.compile(r"\bif\s*\((?P<cond>[^;{]+?)\)\s*[{]?")


def analyze(source: str) -> list[Finding]:
    findings: list[Finding] = []
    for lineno, line in enumerate(source.splitlines(), start=1):
        for m in IF_RE.finditer(line):
            cond = m.group("cond").strip()
            verdict, reason = classify(cond)
            findings.append(Finding(lineno, cond, verdict, reason))
    return findings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="path to a .cu/.cuh file")
    args = parser.parse_args()
    text = args.source.read_text()
    findings = analyze(text)

    counts = {"divergent": 0, "uniform": 0, "data-dependent": 0}
    for f in findings:
        counts[f.verdict] += 1

    print(f"File: {args.source}")
    print(f"Branches found: {len(findings)}")
    print(f"  divergent:      {counts['divergent']}")
    print(f"  uniform:        {counts['uniform']}")
    print(f"  data-dependent: {counts['data-dependent']}")
    print()
    for f in findings:
        print(f"  line {f.line:>4}  [{f.verdict:<15}] if ({f.condition})")
        print(f"              reason: {f.reason}")


if __name__ == "__main__":
    main()
```

**Expected output on the two kernels in §2.1** (combined into one
file):

```text
Branches found: 4
  divergent:      1
  uniform:        1
  data-dependent: 2
```

(The two `i >= n` guards are flagged "data-dependent" because `n`
is a kernel argument whose relationship to lane ID is not
statically obvious. This is honest — the analyzer admits what it
cannot prove.)

### 2.3 Dynamic measurement — Nsight Compute

Nsight Compute exposes divergence directly. Two metrics are
sufficient to compute a divergence ratio:

- `smsp__sass_branch_targets.sum` — total branch targets executed.
- `smsp__sass_branch_targets_threads_divergent.sum` — branch
  targets where threads in the warp went to different targets
  (i.e. divergent).

```bash
ncu --metrics \
    smsp__sass_branch_targets.sum,\
smsp__sass_branch_targets_threads_divergent.sum \
    --launch-skip 0 --launch-count 1 \
    ./a.out
```

Divergence ratio:

```text
divergence_ratio = smsp__sass_branch_targets_threads_divergent.sum
                 / smsp__sass_branch_targets.sum
```

`ncu --section SourceCounters` provides a wider report that also
surfaces per-line source attribution, which is how learners
locate the specific `if` in the kernel.

> The exact metric names above follow NVIDIA's Nsight Compute
> metrics reference (linked in §6). Metric names are
> version-stable but learners on older Nsight Compute versions
> may need to verify via `ncu --query-metrics | grep branch`.

> A published "divergence ratio above X% means you should
> rewrite the kernel" threshold is not pinned here — the module
> README mentions a 5% rule of thumb but we have not found that
> number in an authoritative NVIDIA source. Treat any absolute
> threshold as hardware- and kernel-dependent; compare against
> an alternative implementation instead.

### 2.4 Required written interpretation

The submitted analysis must contain:

1. A list of the kernels analyzed and, for each, the number of
   static branches classified as divergent / uniform /
   data-dependent.
2. For at least one kernel, a before/after: rewrite one divergent
   branch so it becomes warp-uniform (for example, via a lookup
   table indexed by the lane ID, or by sorting work so that
   whole warps share the same branch direction) and quantify
   the change either with the static analyzer or with Nsight
   Compute.
3. One sentence tying the rewrite back to the module's framing:
   the goal is not to eliminate every `if`, only those where
   lanes within a warp disagree.

## 3. Validation steps

1. **Static check**: `python -m py_compile warp_divergence_analyzer.py`
   — no syntax errors; standard library only, no runtime
   dependencies.
2. **Analyzer self-test**: run the analyzer against the §2.1 pair
   of kernels. The `threadIdx`-based kernel must be flagged
   "divergent"; the `blockIdx`-based kernel must be flagged
   "uniform". If either verdict is wrong, the heuristic in
   `classify()` is broken.
3. **Nsight Compute path** (when GPU present):
   - Compile the demo kernels.
   - Run the `ncu` command in §2.3.
   - Confirm that the divergent kernel reports a non-zero value
     for `smsp__sass_branch_targets_threads_divergent.sum` and
     the uniform kernel reports close to zero.
4. **Before/after demonstration**: the rewrite in §2.4 item 2 is
   present and the metrics (static or dynamic) support the
   claim.
5. **Honest handling of unknown cases**: for data-dependent
   conditions, the tool must report "data-dependent", not
   silently classify as uniform. A solution that labels every
   unclassifiable branch as "uniform" fails validation because
   it underreports divergence.

## 4. Rubric or review checklist

Total: 10 points. Pass threshold: 7.

| Criterion | Points | What earns full credit |
|---|---|---|
| Correctly identifies `threadIdx`-dependent branches as divergent | 2 | The reference kernel in §2.1 (or an equivalent constructed example) is flagged divergent. |
| Correctly identifies `blockIdx` / `gridDim` / `blockDim` branches as warp-uniform | 2 | The matching uniform kernel in §2.1 is flagged non-divergent, with reasoning. |
| Reports a "data-dependent / cannot classify" category | 1 | Tool does not force a binary verdict on expressions it cannot analyze. |
| Produces a per-kernel summary count (divergent/uniform/unknown) | 1 | A summary section, not just a dump of per-line findings. |
| Demonstrates one before/after rewrite | 2 | A branch is rewritten from divergent to warp-uniform and the change is quantified (static count change OR Nsight Compute metric change). |
| Written interpretation present | 1 | §2.4 items 1–3 all present. |
| Nsight Compute command or reasoning included | 1 | At least cites the correct metric names and the `ncu` invocation; running it is not required if GPU access is unavailable but the reasoning must be correct. |

### Bonus (+1, max 10)

Learner implements the analyzer against a real CUDA AST (e.g.
`clang.cindex` / libclang) instead of regex, correctly handling
nested conditionals and ternary operators. This earns credit by
*correctness*, not by complexity — a buggy AST implementation
scores lower than a correct regex implementation.

## 5. Common mistakes

1. **Treating every `if` as divergent.** Most kernels have many
   conditionals, and most are warp-uniform (they key off block or
   grid identifiers). Flagging everything inflates the finding
   count and makes the tool useless.
2. **Ignoring the lane-ID vs. warp-ID distinction.**
   `threadIdx.x >> 5` (warp ID within block) is warp-uniform;
   `threadIdx.x & 31` (lane ID within warp) is lane-dependent.
   A tool that cannot tell these apart is naive — call them both
   "divergent" and you miss the point; call them both "uniform"
   and you miss the actual bug.
3. **Measuring the whole program instead of the kernel.** `ncu`
   defaults can collect across multiple kernels; use
   `--launch-skip` / `--launch-count` or a kernel-name filter to
   scope the measurement.
4. **Confusing "branch count" with "divergence".** A kernel can
   have many branches, all warp-uniform, and have zero
   divergence. Report the ratio, not the raw count.
5. **Not providing a before/after.** A tool that reports
   divergence but never shows that rewriting improves things has
   not demonstrated understanding. The point of the metric is to
   drive a change.
6. **Assuming `__syncwarp` cures divergence.** It synchronizes
   the warp; it does not re-converge diverged lanes into a
   single branch. The fix is the control-flow rewrite, not an
   intrinsic.
7. **Using CPU-side timing to "detect" divergence.** Divergence is
   an SM-level scheduling effect; host-side wall time will not
   isolate it from launch overhead or memory stalls. Use Nsight
   Compute metrics or static analysis — not `time.perf_counter`.

## 6. References

**Official project documentation (primary sources):**

- NVIDIA CUDA C++ Programming Guide — "SIMT Architecture" and
  "Warp Divergence" sections. Authoritative explanation of how
  the hardware handles intra-warp branch disagreement.
  <https://docs.nvidia.com/cuda/cuda-c-programming-guide/>
- NVIDIA Nsight Compute — user guide and metrics reference.
  Documents the `smsp__sass_branch_targets*` family of metrics
  used in §2.3.
  <https://docs.nvidia.com/nsight-compute/NsightCompute/>
  <https://docs.nvidia.com/nsight-compute/ProfilingGuide/>
- NVIDIA CUDA Best Practices Guide — "Control Flow" chapter,
  which discusses divergence avoidance patterns.
  <https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/>

**Local exercise context:**

- [`../SOLUTION.md`](../SOLUTION.md) — module rationale,
  specifically "Decision 2: Warp divergence treated as a
  first-class metric", which motivates the mechanical-analyzer
  approach used here.
- [`./README.md`](./README.md) — short-form pointer to the
  Nsight Compute source-counter section used in §2.3.
- [`../exercise-04/`](../exercise-04/) — roofline analysis. A
  kernel with high divergence often appears compute-bound in
  the profiler for the wrong reason; cross-reference when
  interpreting results.
