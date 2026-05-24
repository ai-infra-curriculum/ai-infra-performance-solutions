# Tiled Matmul — Solution

`tiled_matmul.cu` shows the standard tiled implementation with `+1` shared-memory padding for bank-conflict avoidance and `#pragma unroll` on the inner loop.

Typical measured: ~50% of cuBLAS at N=4096 on L40S. The remaining gap requires Tensor Core fusion (CUTLASS / cuBLAS).
