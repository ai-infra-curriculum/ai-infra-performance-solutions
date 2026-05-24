# Warp Divergence Analyzer — Solution

Use Nsight Compute's "Branch Stats" section to measure warp divergence per
kernel. A high divergence ratio means threads in a warp took different
branches; performance drops proportionally.

```bash
ncu --section SourceCounters python my_kernel.py
```

Look for: `smsp__sass_branch_targets_threads_divergent` > 5% of total.
