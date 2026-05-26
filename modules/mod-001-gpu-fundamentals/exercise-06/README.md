# Tensor-Core Throughput Prediction — Solution

`tensor_core_throughput.py` implements the three required
functions using only the verified NVIDIA datasheet values from
the module's lecture notes.

## What it does

For a GEMM of size `M × K` (left) times `K × N` (right) at a
given precision, on either A100 or H100:

1. Computes arithmetic intensity `AI = 2*M*N*K / (bytes_per_elem * (M*K + K*N + M*N))`.
2. Classifies the GEMM as memory-bound or compute-bound by
   comparing AI to the GPU's ridge point on the FP16
   tensor-core path.
3. Predicts realistic attainable TFLOPS using the roofline
   (compute or memory ceiling, whichever is lower) with a
   default utilization factor of 0.85.

## Key worked numbers

The ridge points for FP16 tensor cores (numbers verified
against NVIDIA datasheets):

- **A100**: `312 TFLOPS / 2.04 TB/s ≈ 153 FLOP/byte`
- **H100**: `989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte`

This means H100 has a *higher* ridge — workloads that were
compute-bound on A100 may be memory-bound on H100.

## Operational implications

This prediction is the basis for the "should we use A100 or
H100?" decision. Workloads with:

- **Large GEMMs (high AI)**: H100 dramatically wins (~3×).
- **Small GEMMs (low AI)**: H100 wins less (memory bandwidth is
  ~1.6× higher, not 3×).
- **Mixed**: the dominant kernel's AI determines the answer.

## Cross-references

- Roofline analysis fundamentals: `exercise-04`.
- GPU generation specs: `lecture-notes/06-gpu-generations.md`.
- The matching senior-engineer "GPU fleet strategy" lab:
  `senior-engineer-learning/mod-203-gpu-computing/lab-05`.
