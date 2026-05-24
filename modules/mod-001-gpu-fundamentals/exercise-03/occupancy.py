"""Occupancy calculator: given a kernel's resource use, max concurrent warps/SM."""
from dataclasses import dataclass


@dataclass
class GPULimits:
    threads_per_sm: int
    threads_per_block: int
    blocks_per_sm: int
    registers_per_sm: int
    shared_mem_per_sm_kb: int


H100 = GPULimits(threads_per_sm=2048, threads_per_block=1024, blocks_per_sm=32,
                  registers_per_sm=65536, shared_mem_per_sm_kb=228)


def occupancy(block_size: int, regs_per_thread: int, smem_per_block_kb: int,
               gpu: GPULimits = H100) -> dict:
    blocks_by_threads = gpu.threads_per_sm // block_size
    blocks_by_regs = gpu.registers_per_sm // (regs_per_thread * block_size)
    blocks_by_smem = gpu.shared_mem_per_sm_kb // smem_per_block_kb if smem_per_block_kb > 0 else 1_000_000
    blocks_by_blocks_cap = gpu.blocks_per_sm

    active_blocks = min(blocks_by_threads, blocks_by_regs, blocks_by_smem, blocks_by_blocks_cap)
    active_threads = active_blocks * block_size
    active_warps = active_threads // 32
    max_warps = gpu.threads_per_sm // 32

    return {
        "active_blocks": active_blocks,
        "active_warps": active_warps,
        "max_warps": max_warps,
        "occupancy_pct": round(active_warps / max_warps * 100, 1),
        "limited_by": min(
            ("threads", blocks_by_threads), ("registers", blocks_by_regs),
            ("smem", blocks_by_smem), ("blocks_cap", blocks_by_blocks_cap),
            key=lambda x: x[1],
        )[0],
    }


if __name__ == "__main__":
    import json
    print(json.dumps(occupancy(block_size=256, regs_per_thread=64,
                                smem_per_block_kb=16), indent=2))
