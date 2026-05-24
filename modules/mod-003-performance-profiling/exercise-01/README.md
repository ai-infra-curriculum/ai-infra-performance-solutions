# nsys Trace — Solution

```bash
nsys profile -o trace.nsys-rep python train.py
nsys stats trace.nsys-rep
nsys-ui trace.nsys-rep
```

Look for: top-time kernels in CUDA kernel summary; high `cudaMemcpy` time
= host↔device transfer bottleneck.
