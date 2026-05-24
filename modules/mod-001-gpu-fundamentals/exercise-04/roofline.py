"""Roofline classification: memory-bound vs compute-bound."""
def classify(flops: float, bytes: int, peak_tflops: float, peak_bw_gb_s: float) -> dict:
    """flops + bytes for one kernel invocation."""
    arith_intensity = flops / bytes if bytes else float("inf")
    ridge_intensity = (peak_tflops * 1e12) / (peak_bw_gb_s * 1e9)
    bound = "compute-bound" if arith_intensity > ridge_intensity else "memory-bound"
    achievable_tflops = min(peak_tflops, arith_intensity * peak_bw_gb_s / 1000)
    return {
        "arithmetic_intensity": round(arith_intensity, 2),
        "ridge_intensity": round(ridge_intensity, 2),
        "bound": bound,
        "achievable_tflops": round(achievable_tflops, 2),
    }


if __name__ == "__main__":
    import json
    # SGEMM N=4096: 2 * 4096^3 FLOPS; 3 * 4096^2 * 4 bytes
    n = 4096
    print(json.dumps(classify(
        flops=2 * n**3,
        bytes=3 * n**2 * 4,
        peak_tflops=91.6,    # L40S
        peak_bw_gb_s=864,
    ), indent=2))
