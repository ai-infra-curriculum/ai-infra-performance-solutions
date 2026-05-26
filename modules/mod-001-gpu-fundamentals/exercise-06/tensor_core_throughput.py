"""Exercise 6 solution — predict GEMM TFLOPS via roofline on tensor cores.

Uses only the verified NVIDIA datasheet values from lecture-notes/
06-gpu-generations.md. No estimates, no extrapolation.

Run:

    python check.py
"""

from __future__ import annotations


# Verified NVIDIA datasheet values (FP16 tensor-core, dense, no sparsity):
A100_TC_FP16_TFLOPS = 312.0
A100_BW_GBS = 2039.0

H100_TC_FP16_TFLOPS = 989.0
H100_BW_GBS = 3350.0


_GPU_SPECS = {
    "A100": (A100_TC_FP16_TFLOPS, A100_BW_GBS),
    "H100": (H100_TC_FP16_TFLOPS, H100_BW_GBS),
}


def gemm_arithmetic_intensity(M: int, N: int, K: int, bytes_per_elem: int) -> float:
    """Arithmetic intensity of an MxK times KxN GEMM, in FLOP/byte.

    AI = total_flops / total_bytes
       = 2*M*N*K / (bytes_per_elem * (M*K + K*N + M*N))
    """
    if M <= 0 or N <= 0 or K <= 0:
        raise ValueError("GEMM dimensions must be positive")
    if bytes_per_elem <= 0:
        raise ValueError("bytes_per_elem must be positive")
    total_flops = 2 * M * N * K
    total_bytes = bytes_per_elem * (M * K + K * N + M * N)
    return total_flops / total_bytes


def _ridge_point(gpu: str) -> float:
    """Compute ridge point (FLOP/byte) for a GPU's FP16 tensor-core path."""
    if gpu not in _GPU_SPECS:
        raise ValueError(f"Unknown GPU {gpu!r}; expected 'A100' or 'H100'")
    peak_tflops, peak_bw = _GPU_SPECS[gpu]
    # peak_tflops is TFLOPS = 10^12 FLOP/s
    # peak_bw is GB/s     = 10^9 byte/s
    # ridge_ai = peak_flops_per_s / peak_bytes_per_s
    #          = (peak_tflops * 1000) [GFLOP/s] / peak_bw [GB/s]
    #          = (peak_tflops * 1000) / peak_bw
    return (peak_tflops * 1000.0) / peak_bw


def classify_gemm(M: int, N: int, K: int, bytes_per_elem: int, gpu: str) -> str:
    """Return 'memory-bound' or 'compute-bound' for a GEMM on `gpu`.

    A GEMM is memory-bound if its arithmetic intensity is below the
    GPU's ridge point on the FP16 tensor-core path.
    """
    ai = gemm_arithmetic_intensity(M, N, K, bytes_per_elem)
    ridge = _ridge_point(gpu)
    return "memory-bound" if ai < ridge else "compute-bound"


def predict_tflops(
    M: int,
    N: int,
    K: int,
    bytes_per_elem: int,
    gpu: str,
    util: float = 0.85,
) -> float:
    """Predict realistic TFLOPS for the GEMM on `gpu`'s tensor cores.

    Uses the roofline: attainable performance is the minimum of the
    compute ceiling and the memory ceiling at the GEMM's arithmetic
    intensity. A utilization factor (default 0.85) accounts for the
    gap between roofline-peak and what well-tuned cuBLAS achieves on
    real workloads.
    """
    if gpu not in _GPU_SPECS:
        raise ValueError(f"Unknown GPU {gpu!r}; expected 'A100' or 'H100'")
    if not 0.0 < util <= 1.0:
        raise ValueError("util must be in (0, 1]")
    peak_tflops, peak_bw = _GPU_SPECS[gpu]
    ai = gemm_arithmetic_intensity(M, N, K, bytes_per_elem)

    # Both ceilings expressed in GFLOPS so we can compare apples-to-apples.
    compute_ceiling_gflops = peak_tflops * 1000.0
    memory_ceiling_gflops = peak_bw * ai
    attainable_gflops = min(compute_ceiling_gflops, memory_ceiling_gflops)

    realistic_tflops = util * attainable_gflops / 1000.0
    return realistic_tflops


if __name__ == "__main__":
    # Sanity check: a 4096x4096 FP16 GEMM is compute-bound on both A100 and H100.
    M = N = K = 4096
    BYTES = 2  # FP16

    print(f"GEMM {M}x{K} @ {K}x{N}, FP16:")
    print(f"  AI = {gemm_arithmetic_intensity(M, N, K, BYTES):.2f} FLOP/byte")
    for gpu in ("A100", "H100"):
        ridge = _ridge_point(gpu)
        cls = classify_gemm(M, N, K, BYTES, gpu)
        tflops = predict_tflops(M, N, K, BYTES, gpu)
        print(f"  {gpu}: ridge={ridge:.0f} FLOP/byte, {cls}, predicted={tflops:.1f} TFLOPS")
