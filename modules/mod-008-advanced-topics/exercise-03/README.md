# NCCL Tests — Solution

`nccl_bench.sh` runs all-reduce bandwidth measurement. Compare:
- Single-node 8×H100 NVLink: expect ~700-900 GB/s aggregate
- 2-node 4 GPUs each NDR IB: expect ~150-250 GB/s
- PCIe-only (no NVLink): expect ~50 GB/s

Big gap between NVLink and PCIe = TP across PCIe is a bad idea.
