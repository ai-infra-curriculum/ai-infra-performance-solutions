# Cold Start Mitigation — Solution

Three layers:
1. **Pre-pull image via DaemonSet**: image always present on every node
   ```yaml
   apiVersion: apps/v1
   kind: DaemonSet
   metadata: { name: prepull-vllm }
   spec:
     selector: { matchLabels: { app: prepull } }
     template:
       metadata: { labels: { app: prepull } }
       spec:
         containers:
           - name: pause
             image: vllm/vllm-openai:latest
             command: ["sleep", "infinity"]
             resources: { limits: { cpu: 10m, memory: 64Mi } }
   ```
2. **Pre-warm idle replicas**: keep min replicas above traffic floor
3. **Slow-start LB**: Envoy / NGINX gradual ramp on new replicas (20% / 30s)

Typical: cold-start time 6min → 90s.
