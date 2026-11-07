# Exercise 04 — CUDA Memory Snapshot

## 1. Solution overview

`torch.cuda.memory._record_memory_history` records every allocation,
free, and allocator event with a Python stack. `torch.cuda.memory._dump_snapshot`
writes the recording to a `.pickle` file that can be loaded in the
official visualizer at <https://pytorch.org/memory_viz>.

This is the **only** tool that answers "why did we OOM at iteration
4231 when the model fits in memory": it shows what tensors were
alive at that moment, when they were allocated, and which Python
frame allocated them. Kernel-level profilers cannot answer that
question (see module-level SOLUTION.md, *Decision 4*).

The reference workflow has three variants:

1. **OOM capture** — automatic snapshot on `CUDAOutOfMemoryError`,
   via the `max_entries` + `oom_observer` API.
2. **Steady-state capture** — snapshot at a chosen checkpoint, used
   to diagnose fragmentation or unexpected tensor retention.
3. **Growth capture** — periodic snapshots to show whether
   per-iteration memory is growing (leak) or stable (healthy).

## 2. Worked answer / implementation

### 2.1 Enable history recording

```python
import torch

# Enable the allocator history *before* the training loop.
torch.cuda.memory._record_memory_history(
    enabled="all",   # record allocations, frees, and segment events
    context="all",   # capture Python stacks at every event
    stacks="python", # cheaper than "all"; switch to "all" for C++ traces
    max_entries=100_000,  # ring buffer; raise for long runs
)
```

Important:

- `_record_memory_history` is marked private in PyTorch's public API
  and may change between versions; pin the PyTorch version you test
  against. The memory visualizer page at
  <https://pytorch.org/memory_viz> is the officially supported
  consumer of this format.
- Recording has overhead; disable it for steady-state throughput
  measurements.

### 2.2 Dump on OOM

Wrap the training loop:

```python
def dump_on_oom(path="oom_snapshot.pickle"):
    try:
        yield
    except torch.cuda.OutOfMemoryError:
        torch.cuda.memory._dump_snapshot(path)
        raise
```

Or register an observer that runs before the allocator raises:

```python
def _oom_observer(device, alloc, device_alloc, device_free):
    torch.cuda.memory._dump_snapshot("oom_snapshot.pickle")

torch.cuda._C._cuda_attach_out_of_memory_observer(_oom_observer)
```

### 2.3 Dump at a checkpoint

For non-OOM investigations (fragmentation, retention), dump at a
known iteration and stop recording:

```python
for step, batch in enumerate(loader):
    train_step(batch)
    if step == 50:
        torch.cuda.memory._dump_snapshot("step50.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)
        break
```

### 2.4 Visualize

Load the pickle at <https://pytorch.org/memory_viz>. Four views are
available on that page:

| View | Question it answers |
|---|---|
| **Active Memory Timeline** | How does allocated memory evolve over time? Any long-lived growth? |
| **Allocator State History** | Where are the segments, and how fragmented is each one? |
| **Active Cached Segment Timeline** | How much reserved-but-unused memory is sitting in the cache? |
| **Allocator Settings** | Which `PYTORCH_CUDA_ALLOC_CONF` settings were in effect (e.g. `expandable_segments`)? |

Cross-reference with `torch.cuda.memory_stats()` for scalar totals:

```python
stats = torch.cuda.memory_stats()
print({k: stats[k] for k in [
    "allocated_bytes.all.peak",
    "reserved_bytes.all.peak",
    "num_alloc_retries",
    "num_ooms",
]})
```

If `reserved_bytes.all.peak - allocated_bytes.all.peak` is large (and
grows), the allocator is fragmenting. `num_alloc_retries > 0` means
the allocator had to compact or release cache to satisfy a request —
a reliable early signal of fragmentation pressure.

### 2.5 The three bugs this exercise hunts

The reference workflow trains learners to recognize three signatures:

1. **Memory fragmentation.** Peak allocated is well below peak
   reserved, yet an allocation fails. The timeline shows many small
   gaps between live segments. Mitigations: set
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, reduce the
   variance in tensor sizes across a step, or call
   `torch.cuda.empty_cache()` at well-chosen boundaries (sparingly).
2. **Tensor lifetime leak.** The active-memory timeline shows a
   staircase: allocations at step N remain live at step N+1. Common
   causes include storing per-step tensors in a Python list for
   logging, retaining the graph for backward when it is not needed
   (`loss.backward(retain_graph=True)`), or holding a `.detach()`
   reference without calling `.cpu()`/`del`.
3. **Optimizer-state explosion.** Adam / AdamW maintain first- and
   second-moment buffers of model size; the sum of parameters +
   gradients + two moment buffers can be ~4× the model's parameter
   size in fp32. Mixed precision stores an fp32 master copy on top.
   The reference mitigation path is `AdamW8bit`
   (bitsandbytes) or ZeRO-style optimizer sharding.

### 2.6 Report template

```
Scenario: <OOM | fragmentation | leak | optimizer-state>
Snapshot(s): <file list>
Peak allocated / reserved: <X> / <Y> GiB
num_alloc_retries / num_ooms: <A> / <B>
Dominant allocating frame: <file:line (from memory_viz stack panel)>
Lifetime pattern observed: <growth / stable / staircase>
Diagnosis: <one sentence>
Fix applied: <config change, code change, or allocator flag>
Peak after fix: <X'> / <Y'> GiB
```

## 3. Validation steps

1. `python -c "import torch; print(torch.cuda.is_available())"`
   prints `True`.
2. After running the recorder, the target `.pickle` file is non-empty.
3. Uploading the pickle to <https://pytorch.org/memory_viz> renders
   a timeline without "no events" errors.
4. `torch.cuda.memory_stats()` shows nonzero `num_allocs` during the
   recording window.
5. For the OOM variant, the OOM traceback is still raised after the
   snapshot is written (so the program fails loudly as expected).
6. Recording is disabled (`_record_memory_history(enabled=None)`)
   before any throughput measurement.

## 4. Rubric / review checklist

| Item | Pass |
|---|---|
| `_record_memory_history` enabled before the first allocation of interest | [ ] |
| Snapshot file written and loads in `pytorch.org/memory_viz` | [ ] |
| Report cites peak allocated **and** peak reserved | [ ] |
| Report identifies which of the three signatures applies | [ ] |
| At least one allocator metric from `memory_stats` cited | [ ] |
| Dominant allocating Python frame named (from the visualizer) | [ ] |
| Fix is specific to the identified signature | [ ] |
| Recording turned off before any throughput measurement | [ ] |
| No use of `torch.cuda.empty_cache()` inside a hot loop | [ ] |

## 5. Common mistakes

- **Calling `torch.cuda.empty_cache()` every step to "fix" memory.**
  It returns reserved blocks to the driver but does not shrink peak
  allocated, and it stalls the allocator. Use only at coarse
  boundaries (epoch end, eval start).
- **Reading peak memory from `nvidia-smi`.** `nvidia-smi` reports
  what the driver has given to the process, which includes the
  allocator cache. Use `torch.cuda.memory_stats()` for the
  allocator view.
- **Confusing reserved and allocated.** Reserved is what the
  allocator holds; allocated is what tensors use. Fragmentation is
  the gap.
- **Forgetting to disable history recording.** Throughput numbers
  measured with recording enabled are systematically slow and not
  representative.
- **Tuning `PYTORCH_CUDA_ALLOC_CONF` without a snapshot.** You will
  rediscover someone else's blog post rather than your own workload's
  signature. Snapshot first.
- **Blaming the model when the leak is in logging.** Appending
  `loss` (a tensor) instead of `loss.item()` (a float) to a list is
  the canonical staircase-shaped leak.

## 6. References

- PyTorch, *Understanding CUDA Memory Usage* (official recipe for
  `_record_memory_history` and the visualizer):
  <https://pytorch.org/docs/stable/torch_cuda_memory.html>
- PyTorch, *CUDA Memory Management* (allocator behavior and
  `memory_stats` fields):
  <https://pytorch.org/docs/stable/notes/cuda.html#memory-management>
- PyTorch memory visualizer: <https://pytorch.org/memory_viz>
- PyTorch, *torch.cuda.memory_stats* reference:
  <https://pytorch.org/docs/stable/generated/torch.cuda.memory_stats.html>
- Module-level rationale:
  [`modules/mod-003-performance-profiling/SOLUTION.md`](../SOLUTION.md)
  — *Decision 4: Memory snapshot as a separate skill*.
