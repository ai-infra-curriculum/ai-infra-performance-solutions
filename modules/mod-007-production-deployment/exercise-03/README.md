# Spot Resilience — Solution

See [engineer-solutions/mod-104 ex-15/karpenter/nodepool.yaml](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/blob/main/modules/mod-104-kubernetes/exercise-15-cluster-cost-optimization/karpenter/nodepool.yaml) for Karpenter spot + on-demand fallback.

Plus PodDisruptionBudget + graceful checkpoint pattern:
```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata: { name: training }
spec:
  minAvailable: 1
  selector: { matchLabels: { app: training } }
```
