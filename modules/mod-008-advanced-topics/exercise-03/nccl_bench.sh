#!/usr/bin/env bash
# NCCL all-reduce bandwidth test.
# Requires nccl-tests built; see https://github.com/NVIDIA/nccl-tests
set -euo pipefail

GPUS=${GPUS:-8}
SIZE=${SIZE:-1G}
mpirun -np $GPUS \
  all_reduce_perf -b 8 -e $SIZE -f 2 -g 1

# Typical: H100 NVLink ~700-900 GB/s; PCIe ~50 GB/s
