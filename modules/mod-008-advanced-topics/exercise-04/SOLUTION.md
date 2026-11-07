# Exercise 04 — MIG Partition (Hard-Isolated GPU Slices)

## 1. Solution overview

Put an NVIDIA GPU into **Multi-Instance GPU (MIG)** mode and
partition it into isolated slices that can be scheduled as
independent resources in Kubernetes. On A100 and H100, MIG
partitions the GPU at the hardware level: each GPU instance gets
its own memory slice, its own L2 cache slice, and a fixed share of
SMs. Processes on one instance cannot observe or interfere with
processes on another — the isolation is the point.

Two ideas the exercise is really testing:

1. **MIG is a scheduling decision, not a software one.** Profiles
   are chosen at GPU boot / reconfiguration time; you cannot
   resize a slice while workloads are running on the device. The
   learner must think about *who pays the drain cost* when the
   profile needs to change.
2. **MIG vs MPS is a tenancy question.** MIG gives hard memory
   isolation (right answer for multi-tenant production). MPS
   shares SMs with no memory isolation (right answer for
   cooperative single-tenant batch jobs). The module SOLUTION
   explicitly prefers MIG here; the learner must be able to say
   why.

Related module context: see the module-level
[`SOLUTION.md`](../SOLUTION.md), Decision 4 ("MIG partition — the
only way to share an H100 cleanly") and the trade-off section on
"MIG over MPS". The stub for this exercise lives in
[`README.md`](./README.md).

## 2. Worked answer or implementation

### 2.1 Target geometry

The stub geometry is **three small slices plus one medium slice**
on a single GPU:

- `1g.5gb` × 3 — three independent slices with 1/7th of the SMs
  and 5 GB of frame buffer each, for small-model inference.
- `2g.10gb` × 1 — one slice with 2/7th of the SMs and 10 GB of
  frame buffer, for a slightly larger model or batched traffic.

Profile names follow NVIDIA's `<N>g.<M>gb` convention: `N` is the
number of GPU-instance "slices" of compute and `M` is the memory
in GB. The exact profiles available depend on the GPU SKU — A100
40 GB, A100 80 GB, and H100 80 GB publish different profile tables
in the MIG User Guide. On H100 80 GB, instances scale up to
`7g.80gb` (the whole GPU as one MIG instance).

Pick the profile *once*, ideally from a documented site-wide table;
do not let individual pods pick arbitrary profiles, because every
reconfiguration requires draining the node (see §5).

### 2.2 Host-side enablement

Run on the GPU node, as root or via your existing GPU-operator /
Ansible path. The steps are: enable MIG mode, destroy any existing
instances, create GPU instances (GIs), create a compute instance
(CI) inside each GI.

```bash
# modules/mod-008-advanced-topics/exercise-04/mig_setup.sh
#!/usr/bin/env bash
# Enable MIG and create the target geometry on GPU index 0.
# IDEMPOTENT: destroys existing MIG config before recreating.
set -euo pipefail

GPU_INDEX="${GPU_INDEX:-0}"

# 1) Ensure no workloads are using the GPU — MIG mode change
#    requires an idle device. Drain Kubernetes pods first.
if nvidia-smi -i "$GPU_INDEX" --query-compute-apps=pid --format=csv,noheader | grep -q .
then
    echo "ERROR: processes still using GPU $GPU_INDEX; drain the node first." >&2
    exit 1
fi

# 2) Enable MIG mode. On some SKUs this requires a GPU reset or a
#    full node reboot; nvidia-smi will tell you which.
sudo nvidia-smi -i "$GPU_INDEX" -mig 1

# 3) Clear any stale instances left over from a previous config.
sudo nvidia-smi mig -i "$GPU_INDEX" -dci || true
sudo nvidia-smi mig -i "$GPU_INDEX" -dgi || true

# 4) Create the GPU instances (GIs) with -C so matching compute
#    instances (CIs) are auto-created in the default layout.
sudo nvidia-smi mig -i "$GPU_INDEX" \
    -cgi 1g.5gb,1g.5gb,1g.5gb,2g.10gb -C

# 5) Verify.
nvidia-smi mig -i "$GPU_INDEX" -lgi   # list GPU instances
nvidia-smi mig -i "$GPU_INDEX" -lci   # list compute instances
nvidia-smi -L                         # shows MIG-GPU UUIDs
```

Notes:

- `-cgi` (create GI) with `-C` auto-creates one CI per GI using
  the default compute slice layout. For custom CI layouts, use
  `-cci` explicitly after `-cgi` without `-C`.
- The profile list is **ordered placement-sensitive**: not every
  combination of profiles fits every GPU, because MIG slots have
  fixed positions on the die. The MIG User Guide publishes the
  valid placement tables per SKU. If `-cgi` rejects your list,
  reorder the profiles (largest first often works) or consult the
  placement table for that SKU.
- MIG mode persistence depends on the driver's persistence
  daemon. Without it, MIG mode can revert on reboot. Enable
  `nvidia-persistenced` on the node.

### 2.3 Expose slices to Kubernetes

Install the NVIDIA GPU Operator (or the standalone
`k8s-device-plugin` + `gpu-feature-discovery`) in **MIG strategy**
mode. Two strategies are documented:

- **`single`** — every pod on the node sees the same MIG profile
  (one profile per node). Simpler to schedule against; wastes
  capacity if your workload mix doesn't match the profile.
- **`mixed`** — multiple profiles coexist on the node. Pods
  request a specific profile name. This matches the geometry
  above (three `1g.5gb` + one `2g.10gb`).

Using `mixed` on the node, the resource names Kubernetes advertises
become `nvidia.com/mig-1g.5gb` and `nvidia.com/mig-2g.10gb`. Pod
specs request them like any other extended resource:

```yaml
# modules/mod-008-advanced-topics/exercise-04/pod-small.yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference-small
  labels: { tier: small }
spec:
  restartPolicy: Never
  containers:
    - name: server
      image: registry.internal/inference-small:latest
      resources:
        limits:
          nvidia.com/mig-1g.5gb: 1
        requests:
          nvidia.com/mig-1g.5gb: 1
```

```yaml
# modules/mod-008-advanced-topics/exercise-04/pod-medium.yaml
apiVersion: v1
kind: Pod
metadata:
  name: inference-medium
  labels: { tier: medium }
spec:
  restartPolicy: Never
  containers:
    - name: server
      image: registry.internal/inference-medium:latest
      resources:
        limits:
          nvidia.com/mig-2g.10gb: 1
        requests:
          nvidia.com/mig-2g.10gb: 1
```

Scheduling behaviour:

- Three `inference-small` pods and one `inference-medium` pod
  fit the node simultaneously.
- A fourth `inference-small` pod stays `Pending` with
  `Insufficient nvidia.com/mig-1g.5gb` — the node is full. Add a
  second node (or a second MIG-capable GPU) rather than resizing.
- Pods with `nvidia.com/gpu: 1` (whole-GPU request) will **not**
  schedule on a MIG-enabled node under the `mixed` strategy, and
  should not — the GPU is partitioned.

### 2.4 Reconfiguration workflow

MIG profile changes require the slices to be idle. The production
pattern is:

1. `kubectl cordon <node>` — stop new scheduling.
2. `kubectl drain <node> --ignore-daemonsets --delete-emptydir-data`
   — evict existing pods to their recreations on other nodes.
3. Re-run `mig_setup.sh` with the new profile list (or let the
   GPU Operator's `MigConfig` CR do it declaratively).
4. `kubectl uncordon <node>` — accept new pods against the new
   profiles.

If the GPU Operator is in use, the authoritative
mechanism is a `MigConfig` ConfigMap listing named profiles per
node label; the operator handles the cordon / drain / re-partition
sequence for you. The learner should at least *know this exists*
even if the hand-rolled script is the artifact.

## 3. Validation steps

1. **Pre-flight**
   - GPU model supports MIG: A100 or H100. `nvidia-smi -q |
     grep "MIG Mode"` returns a non-"N/A" line.
   - Driver is recent enough to list the target profile (profile
     tables widen with driver / firmware updates).
2. **Static review**
   - `mig_setup.sh` refuses to run when processes are still on
     the GPU (the `--query-compute-apps` check).
   - Profile list matches a valid placement for the SKU.
   - Pod manifests request the **MIG** resource name
     (`nvidia.com/mig-1g.5gb`), not the whole-GPU name
     (`nvidia.com/gpu`).
3. **Apply and verify**
   - `sudo bash mig_setup.sh` completes without error.
   - `nvidia-smi mig -lgi` lists 4 GPU instances matching the
     profile list.
   - `nvidia-smi -L` lists 4 `MIG-*` UUID lines under the parent
     GPU.
   - `kubectl describe node <node>` shows capacities for both
     `nvidia.com/mig-1g.5gb: 3` and `nvidia.com/mig-2g.10gb: 1`.
4. **Schedule the pods**
   - `kubectl apply -f pod-small.yaml` three times (edit the
     name) and `pod-medium.yaml` once. All four reach `Running`.
   - A fifth `pod-small` remains `Pending` with an "Insufficient"
     message that names the exhausted MIG resource.
5. **Isolation check**
   - In each `inference-small` pod, `nvidia-smi` shows a single
     MIG device of profile `1g.5gb` and no visibility into
     sibling instances' processes. Allocating memory beyond 5 GB
     in that pod fails with OOM — proving memory isolation.
6. **Teardown**
   - Drain the node and rerun `mig_setup.sh` with a different
     profile list to prove the reconfiguration path works end-to-
     end. A successful reconfiguration *while a pod was still
     scheduled* should fail or stall — the drain is required.

## 4. Rubric / review checklist

| # | Criterion | Pass signal |
|---|-----------|-------------|
| 1 | MIG mode enabled explicitly | Script calls `nvidia-smi -mig 1`; learner can state whether a GPU reset / reboot was required for this SKU |
| 2 | Idempotent setup | Script destroys existing CIs/GIs before recreating; re-running is safe |
| 3 | Valid placement for the SKU | Profile list in `-cgi` is a documented valid combination for the GPU model; reordering or alternatives are explained if the default fails |
| 4 | Kubernetes `mixed` strategy | GPU Operator (or device plugin) configured with the mixed MIG strategy; both resource names advertised |
| 5 | Pod specs use MIG resource name | Pods request `nvidia.com/mig-<profile>: 1`, not `nvidia.com/gpu: 1` |
| 6 | Isolation demonstrated | Learner shows an OOM at the slice's memory limit and shows a sibling instance unaffected |
| 7 | Reconfiguration path documented | Cordon → drain → re-partition → uncordon sequence stated (and ideally performed once) |
| 8 | MIG vs MPS justified | Learner can state why hard isolation (MIG) was the right choice for multi-tenant serving, and when MPS would be the right choice instead |

Scoring guidance:

- 7–8 passes: can own a MIG-partitioned fleet. Move on to
  exercise 05 (FP8 training).
- 5–6 passes: setup is correct but the operator story is weak —
  have the learner implement the reconfiguration workflow
  end-to-end before signing off.
- ≤ 4 passes: MIG is still an incantation. Walk through the
  module SOLUTION's Decision 4 and the MIG User Guide placement
  tables with them.

## 5. Common mistakes

1. **Trying to change the profile without a drain.** `nvidia-smi
   mig -dgi` / `-cgi` on an active GPU fails or (worse) succeeds
   partially, leaving the node in a mixed state. Always cordon
   and drain first.
2. **Picking an invalid placement.** Not every combination of
   profiles fits every GPU — MIG slots are positional. The MIG
   User Guide publishes placement tables per SKU; `-cgi` returns
   "Insufficient Resources" when the requested set is invalid.
   Treat that error as "wrong geometry," not "wrong hardware."
3. **Using MPS when MIG was needed.** NVIDIA MPS (Multi-Process
   Service) shares the GPU without memory isolation; a crashing
   client can take down co-tenants. For multi-tenant serving,
   MIG is the right default. The module SOLUTION makes this
   explicit (trade-offs section, "MIG over MPS").
4. **Requesting `nvidia.com/gpu: 1` on a MIG node.** The
   whole-GPU resource is not advertised when MIG is enabled under
   the `mixed` strategy, so the pod stays `Pending`. Fix the
   manifest, not the strategy.
5. **Hand-rolling MIG geometry per node when the GPU Operator is
   available.** Drift between nodes is the result; one node ends
   up with `3×1g.5gb + 1×2g.10gb` and another with
   `7×1g.10gb`, and pods schedule unpredictably. Use the operator's
   `MigConfig` custom resource so geometry is declarative.
6. **Forgetting persistence.** Without `nvidia-persistenced`
   enabled, MIG mode can revert on reboot and the node silently
   comes back as a whole GPU. Pods still scheduled by the
   advertised resource names then stall.
7. **Assuming slice `1g.5gb` is 1/7th of a GPU "evenly."** The
   compute slice is 1/7th of the SMs, but shared-resource
   contention (L2, memory bandwidth) is not perfectly linear.
   Benchmark the workload on a real slice; do not divide
   whole-GPU numbers by 7 and call it a plan.

## 6. References

- NVIDIA Multi-Instance GPU User Guide — concept, profiles per
  SKU, placement tables, and the `nvidia-smi mig` command family:
  https://docs.nvidia.com/datacenter/tesla/mig-user-guide/
- NVIDIA GPU Operator documentation — MIG strategies (`single`,
  `mixed`), `MigConfig` custom resource, and drain-and-
  reconfigure workflow:
  https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/
- NVIDIA `k8s-device-plugin` — resource advertisement and the
  MIG resource-name scheme:
  https://github.com/NVIDIA/k8s-device-plugin
- Kubernetes documentation — scheduling against extended
  resources (`nvidia.com/mig-*`), cordon and drain:
  https://kubernetes.io/docs/concepts/scheduling-eviction/
- Local module context —
  [`modules/mod-008-advanced-topics/SOLUTION.md`](../SOLUTION.md),
  Decision 4 ("MIG partition — the only way to share an H100
  cleanly") and the "MIG over MPS" trade-off.
- Local stub —
  [`modules/mod-008-advanced-topics/exercise-04/README.md`](./README.md)
  points at the engineer-track GPU-sharing exercise with its
  deployment examples.
