# ncu Deep Dive — Solution

```bash
ncu --set full --kernel-id ::matmul:1 python train.py
ncu -i report.ncu-rep --section SpeedOfLight
```

The "GPU Speed of Light" section directly shows % of peak compute + memory.
