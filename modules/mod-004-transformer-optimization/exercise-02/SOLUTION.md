# Exercise 02 — `torch.compile` for Transformer Inference

## 1. Solution overview

The exercise asks learners to apply `torch.compile` to a transformer
inference path, measure the steady-state speedup against eager mode,
and account for the cold-start compile tax. The reference solution
wraps an existing attention/MLP block, warms the compiled graph,
reports both compile time and steady-state throughput, and checks
whether input-shape variation causes silent recompilation.

Key teaching points:

- `torch.compile` returns a callable backed by TorchDynamo (graph
  capture) and TorchInductor (code generation), which fuses pointwise
  ops and lowers hot regions to Triton kernels.
- The speedup comes from kernel fusion and Python-overhead
  elimination, so it is largest on small-batch decode and on paths
  that are not already dominated by a single large matmul.
- Compilation is per-dynamic-shape. Variable sequence lengths at
  serving time can trigger recompiles unless `dynamic=True` is used
  or shapes are bucketed.

## 2. Worked answer or implementation

The script below is the canonical reference. It runs on a single GPU
with PyTorch ≥ 2.1 and expects `torch`, `torch.nn.functional`.

```python
# modules/mod-004-transformer-optimization/exercise-02/compile_demo.py
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        h = self.heads
        qkv = self.qkv(self.ln1(x)).reshape(b, n, 3, h, d // h)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = x + self.proj(attn)
        x = x + self.mlp(self.ln2(x))
        return x


def bench(fn, x, warmup: int = 5, iters: int = 50) -> float:
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main() -> None:
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16
    model = Block(dim=2048, heads=16).to(device=device, dtype=dtype).eval()
    x = torch.randn(4, 512, 2048, device=device, dtype=dtype)

    eager_ms = bench(model, x) * 1e3

    compiled = torch.compile(model, mode="reduce-overhead", fullgraph=True)
    t0 = time.perf_counter()
    compiled(x)                # triggers first compile
    torch.cuda.synchronize()
    compile_ms = (time.perf_counter() - t0) * 1e3
    compiled_ms = bench(compiled, x) * 1e3

    print(f"eager:     {eager_ms:6.2f} ms/iter")
    print(f"compile:   {compile_ms:7.1f} ms (one-time)")
    print(f"compiled:  {compiled_ms:6.2f} ms/iter")
    print(f"speedup:   {eager_ms / compiled_ms:4.2f}x steady-state")

    # Shape-variation probe: changing seq-length triggers recompile
    # unless dynamic=True is passed. Log via TORCH_LOGS=recompiles.
    y = torch.randn(4, 1024, 2048, device=device, dtype=dtype)
    t0 = time.perf_counter()
    compiled(y)
    torch.cuda.synchronize()
    print(f"shape-change recompile: {(time.perf_counter() - t0) * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
```

Decision rationale (why this shape of answer):

- `mode="reduce-overhead"` is the correct default for inference: it
  uses CUDA Graphs on supported paths and caches launches, which
  matters more for decode than for prefill.
- `fullgraph=True` forces TorchDynamo to error on graph breaks
  rather than silently fall back to eager. Learners should see the
  errors and fix them; a partial graph with Python islands negates
  most of the fusion benefit.
- The shape-change probe is deliberately not hidden behind
  `dynamic=True`. The point is to *expose* the recompile cost so the
  decision to bucket shapes or opt into dynamic shapes is informed.
- `bench()` averages over 50 iterations after 5 warmup iterations,
  which is enough to dilute CUDA Graph capture noise without masking
  real variation.

## 3. Validation steps

1. Run on a CUDA device with PyTorch ≥ 2.1:
   `python compile_demo.py`.
2. Confirm numeric equivalence of eager vs compiled on the same
   input:
   ```python
   out_eager = model(x)
   out_comp = compiled(x)
   torch.testing.assert_close(out_eager, out_comp, rtol=1e-2, atol=1e-2)
   ```
   bf16 tolerance is intentional; fp32 should match much more
   tightly.
3. Enable recompile logging to confirm whether a given workload
   triggers silent recompiles:
   `TORCH_LOGS=recompiles python compile_demo.py`.
4. Report the three numbers (eager latency, compile time, compiled
   latency) and the derived steady-state speedup. Any single number
   is insufficient.
5. For a serving scenario, bench at two or three representative
   `(batch, seq_len)` combinations; a win at one shape is not a win
   across the SLO distribution.

## 4. Rubric or review checklist

| Check | Weight | Pass criterion |
|---|---|---|
| Numeric equivalence | 20 | `assert_close` passes at the stated tolerance for the chosen dtype |
| Reports compile time separately from steady-state | 20 | First-call latency is excluded from the steady-state average and reported independently |
| Uses an explicit warmup loop | 10 | At least 3 warmup iterations before timed iterations; `cuda.synchronize()` before and after timing |
| Picks a defensible `mode` | 10 | `reduce-overhead` for inference, `max-autotune` for an offline pass; default is acceptable with justification |
| Probes shape variation | 15 | Submits at least one off-shape input and either reports the recompile cost or justifies `dynamic=True` |
| Reports speedup with workload context | 15 | `(batch, seq_len, dtype, GPU)` recorded next to any speedup number |
| Avoids `fullgraph=False` without justification | 10 | Graph breaks either prevented via `fullgraph=True` or each break is explained |

## 5. Common mistakes

1. **Timing including the first call.** The first forward pass runs
   TorchDynamo + TorchInductor and can take tens of seconds. Any
   average that includes it dwarfs the real per-iteration cost.
2. **Omitting `torch.cuda.synchronize()`.** CUDA kernels are
   asynchronous. Timing with `time.perf_counter()` and no sync
   measures launch latency, not execution.
3. **Not noticing recompiles on variable sequence length.** A chat
   server with prompts of length 37, 412, 1183, ... will recompile
   once per unseen length unless `dynamic=True` or explicit shape
   bucketing is used. Verify with `TORCH_LOGS=recompiles`.
4. **Comparing eager fp32 to compiled bf16.** A dtype change on its
   own can explain a 2x speedup. Compare like for like.
5. **Treating `@torch.compile` as free.** On short, memory-bound
   shapes with heavy Python control flow, the compile tax and CUDA
   Graph capture can overshadow the fusion gain. Measure before
   claiming a win.
6. **Trusting `fullgraph=False` silently.** Dynamo graph breaks drop
   the model back to eager at that point, and the user often does
   not notice. Compile with `fullgraph=True` first; relax only if
   the break is understood and acceptable.

## 6. References

- PyTorch docs — `torch.compile`:
  <https://pytorch.org/docs/stable/generated/torch.compile.html>
- PyTorch docs — Introduction to `torch.compile`:
  <https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html>
- PyTorch docs — TorchDynamo / TorchInductor overview:
  <https://pytorch.org/docs/stable/torch.compiler.html>
- PyTorch docs — Dynamic shapes:
  <https://pytorch.org/docs/stable/torch.compiler_dynamic_shapes.html>
- PyTorch docs — Troubleshooting `torch.compile` (recompiles,
  graph breaks):
  <https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html>
- PyTorch docs — `scaled_dot_product_attention`:
  <https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html>
- Module-level context: this solution composes with the
  FlashAttention baseline established in
  `modules/mod-004-transformer-optimization/exercise-01` and the
  module-level discussion in
  `modules/mod-004-transformer-optimization/SOLUTION.md`.
