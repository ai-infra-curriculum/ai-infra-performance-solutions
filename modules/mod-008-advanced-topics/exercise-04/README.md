# MIG Partition — Solution

```bash
# Enable MIG on the host
sudo nvidia-smi -mig 1
# Create instances: 3 × 1g.5gb + 1 × 2g.10gb
sudo nvidia-smi mig -cgi 1g.5gb,1g.5gb,1g.5gb,2g.10gb -C
nvidia-smi mig -lgi
```

Companion: [engineer-solutions/mod-107 ex-10](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-10-gpu-sharing-strategies) for K8s device-plugin configs + Pod specs.
