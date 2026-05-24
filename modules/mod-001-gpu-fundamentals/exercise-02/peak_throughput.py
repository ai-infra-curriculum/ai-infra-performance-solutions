"""Compute theoretical peak throughput for a GPU; compare to measured."""
SPECS = {
    "H100":  {"sm_count": 132, "fp32_tflops": 67,   "mem_gb_s": 3350},
    "A100":  {"sm_count": 108, "fp32_tflops": 19.5, "mem_gb_s": 1935},
    "L40S":  {"sm_count": 142, "fp32_tflops": 91.6, "mem_gb_s": 864},
    "T4":    {"sm_count": 40,  "fp32_tflops": 8.1,  "mem_gb_s": 320},
}


def report(gpu: str, measured_tflops: float, measured_bw_gb_s: float):
    s = SPECS[gpu]
    print(f"{gpu}: {measured_tflops:.1f} / {s['fp32_tflops']} TFLOPS "
          f"= {measured_tflops/s['fp32_tflops']*100:.0f}% of peak")
    print(f"{gpu}: {measured_bw_gb_s:.0f} / {s['mem_gb_s']} GB/s "
          f"= {measured_bw_gb_s/s['mem_gb_s']*100:.0f}% of peak")


if __name__ == "__main__":
    # Example: L40S with cuBLAS matmul + memcpy benchmark
    report("L40S", measured_tflops=85, measured_bw_gb_s=720)
