# SOLUTION — Advanced Topics

> Read this *after* everything else in the performance track. The
> exercises here are individually useful but their real value is
> in giving you the vocabulary to read the next 12 months of
> research papers and understand which of those papers belong in
> your production stack.

## What this module is really teaching

Mod-008 is the "frontier" module. The techniques here either
- exist in production at large labs (CUDA Graphs, FP8, NCCL
  tuning) but haven't fully landed in OSS frameworks, or
- exist in hardware (MIG, FP8 instructions) but require deliberate
  software work to exploit.

The goal is not mastery — these techniques will continue to
evolve — but **fluency**: when a vendor or a paper claims a
speedup, you should be able to read the claim and judge whether
it applies to your workload.

## Architectural decisions and *why*

### Decision 1: CUDA Graphs first — because the win is mechanical

Exercise 01 (CUDA Graphs) covers `torch.cuda.CUDAGraph` capture
and replay. The mechanism: capture a sequence of kernel launches
once, then replay the recorded graph in subsequent iterations.
The saving is the per-launch overhead — typically 5-10µs per
kernel — which adds up for workloads with many small kernels
(small-batch inference, decoding loops).

The reference solution measures the speedup on three workloads:
1. Large-batch training (no benefit; launches are already
   amortized).
2. Small-batch inference (5-15% speedup).
3. Decoding loop on small models (20-40% speedup; the launch cost
   was dominant).

The lesson: **CUDA Graphs are a launch-overhead optimization**.
They don't make kernels faster; they make Python and CUDA
runtime cheaper. Match the technique to the bottleneck.

**Anti-pattern to avoid**: capturing a graph that includes
control flow (`if` statements). The graph captures a single
execution path; control flow in the captured region produces
incorrect results on different inputs.

### Decision 2: Stream overlap — concurrency without parallelism

Exercise 02 (stream overlap) covers using multiple CUDA streams
to overlap compute with memory transfer (e.g., the next batch's
HtoD copy with the current batch's forward pass). The PyTorch
pattern uses non-default streams with `torch.cuda.stream()` and
explicit event-based synchronization.

The reference solution measures:
- Serial: HtoD → compute → DtoH → next batch. ~140 ms / batch.
- Overlapped: HtoD(batch N+1) || compute(batch N). ~95 ms / batch.

The speedup is bounded by the slowest stream. For workloads where
compute >> memory transfer (large models, GPU-resident data), the
overlap buys nothing. For data-loading-bottlenecked workloads
(small models, fast GPUs, slow disks), it's significant.

This is the **same idea** as pipelined data loaders, but at a
finer granularity (kernel-level rather than batch-level).

### Decision 3: NCCL tests as the multi-GPU correctness gate

Exercise 03 (NCCL tests) wires up `nccl-tests` (the
collective-comms test suite from NVIDIA) into the deployment
pipeline. The reason: a multi-GPU cluster that *looks* connected
(nvidia-smi shows NVLink as up) can fail collective tests if
firmware mismatches, PCIe topology issues, or driver bugs exist.
The first time this shows up under load is during a training run
that hangs, and debugging at that point is days of work.

The reference workflow runs `all_reduce_perf`, `all_gather_perf`,
and `reduce_scatter_perf` at cluster commissioning *and* on a
nightly schedule. A drop in collective bandwidth below threshold
(typically 80% of theoretical NVLink bandwidth) is an alert.

**Anti-pattern to avoid**: assuming hardware is healthy because
the deploy succeeded. Healthy hardware passes nccl-tests at
expected bandwidth; "deploy succeeded" only means the manifests
applied.

### Decision 4: MIG partition — the only way to share an H100 cleanly

Exercise 04 (MIG partition) covers NVIDIA's Multi-Instance GPU
mode: an H100 can be partitioned into up to 7 isolated GPU
slices, each with its own memory and compute. This is the
**right answer** for serving multiple small models on the same
physical GPU without noisy-neighbor problems.

The trade-off: MIG slices are fixed at GPU boot time. You can't
resize a slice without rebooting the GPU (which means draining
all pods on that node). The reference solution shows the
deployment pattern:

1. Boot the GPU with a chosen MIG profile (e.g., `7g.80gb` for
   one slice or `1g.10gb` × 7 for many small slices).
2. Expose slices as separate `nvidia.com/mig-1g.10gb` resources
   in Kubernetes.
3. Schedule pods against specific MIG profiles via resource
   requests.

This is the only technique on this list that's a **scheduling
question, not a software question**.

### Decision 5: FP8 training — the precision frontier

Exercise 05 (FP8 training) covers FP8 mixed-precision on H100
using NVIDIA's Transformer Engine. The hardware does FP8 matmuls
at ~2 PFLOPS (vs ~1 PFLOPS BF16 on the same chip). The catch is
**calibration**: FP8 has only 256 representable values, so the
per-tensor scale factors must be tracked dynamically to avoid
overflow/underflow.

The reference solution uses Transformer Engine's
`fp8_autocast()` context and shows the per-tensor scale tracking
in tensorboard logs. The measured speedup on a 7B model
fine-tuning is typically 1.3-1.5x with negligible quality impact
*if calibration converges*. The exercise emphasizes
**always evaluating** because a silent FP8 underflow can degrade
the model without throwing an error.

**Anti-pattern to avoid**: enabling FP8 because the hardware
supports it, without measuring quality. The hardware is fast; the
math is sometimes wrong.

## Trade-offs we deliberately accepted

### H100/Hopper-only assumptions

FP8 instructions, the larger L2 cache, and the per-SM scheduling
model that makes CUDA Graphs less essential are all H100
features. A100 deployments need different choices (BF16 instead of
FP8, more aggressive use of CUDA Graphs because launch overhead is
relatively larger).

### NCCL exclusively

The exercises don't cover MSCCL, NCCL-OFI plug-ins for InfiniBand,
or RCCL (AMD). NCCL is the de facto standard; the techniques
generalize but the names change.

### MIG over MPS

NVIDIA MPS (Multi-Process Service) provides a different sharing
model (shared SMs, no memory isolation). MIG is the right choice
for multi-tenant production because memory isolation is a hard
requirement; MPS is the right choice for single-tenant batch jobs
that don't need isolation. The exercise covers MIG only.

## Common mistakes graders see

1. **CUDA Graph capture with `torch.compile`**: both transform the
   graph; the interaction is fragile. Use one or the other per
   region.
2. **Stream overlap without explicit synchronization**: the
   captured stream finishes "later than expected" and the
   subsequent kernel reads stale data. Always synchronize
   explicitly.
3. **Treating NCCL bandwidth as a static number**: it depends on
   message size. A 1 KB all-reduce hits 5% of peak; a 1 GB
   all-reduce hits 95%. Always specify the message size with the
   reported bandwidth.
4. **Provisioning MIG slices on a node that needs to be reshaped
   later**: every MIG profile change requires a node drain. Pick
   the profile carefully and document it.
5. **Trusting FP8 results without evaluation**: per-tensor scale
   drift produces silent garbage. Always run an eval after each
   FP8 training run.

## When to go beyond this implementation

- Try **MSCCL** for custom collective topologies on bespoke
  hardware (DGX SuperPOD, custom NVL switch topologies).
- Use **CUDA Cooperative Groups** for kernel-level grid
  synchronization without the launch overhead of multiple kernels.
- Experiment with **FP4** (Blackwell) when the hardware lands.
- Move to **kernel fusion via Triton or CUTLASS** for the kernels
  the standard libraries don't cover.

## Related curriculum touchpoints

- `performance/mod-002-cuda-programming` — the CUDA primitives
  underlying graphs, streams, and cooperative groups.
- `performance/mod-005-model-compression` — FP8 sits alongside
  INT8/INT4 in the precision menu.
- `performance/mod-006-distributed-inference` — NCCL is the
  fabric layer beneath every multi-GPU deployment.
- `engineer/mod-107-gpu-computing/exercise-04-multi-gpu-training`
  — the working multi-GPU training example that uses these
  primitives.
