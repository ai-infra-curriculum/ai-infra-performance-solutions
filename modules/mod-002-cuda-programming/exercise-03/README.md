# Reduction — Solution

`reduction.cu` shows three variants:
- `reduce_naive`: divergent (every other thread on each iteration drops out)
- `reduce_shared`: sequential addressing (no divergence)
- `reduce_shfl`: warp-shuffle (no shared memory needed within a warp)

Typical: shfl variant within 5% of `torch.sum` at n=1M.
