# Occupancy Tuning — Solution

Standard sweep pattern:
```bash
for block in 64 128 256 512; do
  for tile in 16 32 64; do
    BLOCK_SIZE=$block TILE=$tile ./matmul_benchmark
  done
done
```

Typical findings:
- TS=32 + block=1024 (32×32 threads): best on most NVIDIA GPUs
- TS=16: lower occupancy
- TS=64: too much shared memory; blocks/SM drops to 1
