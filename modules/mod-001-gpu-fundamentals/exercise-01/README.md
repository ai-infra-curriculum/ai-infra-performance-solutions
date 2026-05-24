# CPU vs GPU Comparison — Solution

Reference: [engineer-solutions/mod-107 ex-03 (pytorch-gpu-pipeline)](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-03-pytorch-gpu-pipeline) has a working ResNet training script you can run on CPU + GPU for direct comparison.

Typical findings for matmul (2048×2048):
- CPU (16 cores): ~5 GFLOPS
- L40S GPU: ~80 TFLOPS

~16,000× speedup for compute-heavy, memory-bound matmul.
