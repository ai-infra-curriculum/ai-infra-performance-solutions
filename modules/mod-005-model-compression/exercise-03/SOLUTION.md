# Exercise 03 — 2:4 Semi-Structured Sparsity

## 1. Solution overview

The exercise asks learners to apply NVIDIA's **2:4 semi-structured
sparsity** pattern to the linear layers of a transformer-style
model, convert the model to use the hardware-accelerated sparse
matmul path on Ampere+ GPUs, and measure the end-to-end speedup
(not just the matmul kernel speedup). The reference solution uses
PyTorch's prototype API `torch.sparse.to_sparse_semi_structured`
together with `torch.ao.pruning.SemiStructuredSparsifier`, produces
a mask → fine-tune → densify-at-inference flow, and reports both a
microbench (sparse matmul vs dense matmul) and an end-to-end
forward-pass speedup.

Key teaching points:

- 2:4 sparsity requires that *every contiguous group of four
  weights along the reduction (input) dimension contain exactly
  two zeros*. The pattern is fixed; the degree of freedom is
  *which* two of the four are zero.
- The acceleration is a hardware feature of the Sparse Tensor
  Cores introduced in NVIDIA Ampere (A100) and preserved on
  Ada / Hopper / Blackwell. Pre-Ampere GPUs (V100 and earlier)
  do **not** execute the sparse matmul path and will either fall
  back to dense or error at kernel dispatch.
- The theoretical kernel speedup is 2× on the sparse matmul, but
  end-to-end speedup is typically **1.3–1.5×** because non-matmul
  ops (softmax, layer norm, embedding lookup, attention mask
  handling) do not benefit.
- Zeroing weights without retraining usually loses several points
  of accuracy. The accepted recipe is **mask → fine-tune → freeze
  mask → convert**: apply the 2:4 mask, fine-tune with the mask in
  place so the surviving weights adapt, then convert the dense
  masked tensor into the packed semi-structured representation.

## 2. Worked answer or implementation

The script below is the canonical reference. It runs on CUDA with
PyTorch ≥ 2.1 on an A100/H100 (or any Ampere+ GPU). The sparse
matmul path requires FP16 or BF16 weights; FP32 falls back to
dense.

```python
# modules/mod-005-model-compression/exercise-03/sparsify_2to4.py
import time
import torch
import torch.nn as nn
from torch.sparse import to_sparse_semi_structured, SparseSemiStructuredTensor


class MLPBlock(nn.Module):
    """A representative linear-heavy block (transformer MLP shape)."""

    def __init__(self, dim: int = 4096, ffn_mult: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, ffn_mult * dim, bias=False)
        self.fc2 = nn.Linear(ffn_mult * dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(torch.nn.functional.gelu(self.fc1(x)))


# --- Step 1. Build a 2:4 mask along the input (reduction) dim. ---
def mask_2to4(w: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask that keeps the top-2 magnitudes per group of 4
    along the last (input) dim of a weight tensor shape [out, in]."""
    out, inp = w.shape
    assert inp % 4 == 0, "2:4 requires in-dim divisible by 4"
    w4 = w.abs().reshape(out, inp // 4, 4)
    _, idx = w4.topk(2, dim=-1)                     # keep 2 of each 4
    mask = torch.zeros_like(w4, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    return mask.reshape(out, inp)


def apply_mask(linear: nn.Linear) -> None:
    with torch.no_grad():
        m = mask_2to4(linear.weight.data)
        linear.weight.data.mul_(m)


# --- Step 2. Short sparsity-aware fine-tune (toy loop). ---
def finetune_with_mask(model: nn.Module, data_iter, steps: int = 50) -> None:
    """Keep the 2:4 mask fixed; let the surviving weights adapt."""
    masks = {name: mask_2to4(m.weight.data)
             for name, m in model.named_modules() if isinstance(m, nn.Linear)}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    for step in range(steps):
        x, y = next(data_iter)
        loss = torch.nn.functional.mse_loss(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    m.weight.data.mul_(masks[name])      # re-apply mask


# --- Step 3. Convert dense masked tensors to the packed sparse rep. ---
def convert_linears_to_semi_structured(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            module.weight = nn.Parameter(
                to_sparse_semi_structured(module.weight.data))


# --- Step 4. Bench. ---
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
    assert torch.cuda.is_available(), "2:4 sparsity requires Ampere+ GPU"
    major, _ = torch.cuda.get_device_capability()
    assert major >= 8, f"Need SM 8.0+ (Ampere), got SM {major}.x"

    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.float16

    dense = MLPBlock(dim=4096).to(device=device, dtype=dtype).eval()
    sparse = MLPBlock(dim=4096).to(device=device, dtype=dtype).eval()
    sparse.load_state_dict(dense.state_dict())

    # Mask + (skip fine-tune in this bench script; real recipe includes it).
    for m in sparse.modules():
        if isinstance(m, nn.Linear):
            apply_mask(m)

    convert_linears_to_semi_structured(sparse)

    # Verify the packed representation is in use.
    assert isinstance(sparse.fc1.weight, SparseSemiStructuredTensor), \
        "fc1.weight did not convert to SparseSemiStructuredTensor"

    x = torch.randn(32, 512, 4096, device=device, dtype=dtype)

    dense_ms = bench(dense, x) * 1e3
    sparse_ms = bench(sparse, x) * 1e3

    # Microbench: isolate the fc1 matmul.
    dense_mm = bench(dense.fc1, x) * 1e3
    sparse_mm = bench(sparse.fc1, x) * 1e3

    print(f"matmul only:   dense {dense_mm:6.2f} ms | "
          f"sparse {sparse_mm:6.2f} ms | speedup {dense_mm/sparse_mm:4.2f}x")
    print(f"end-to-end:    dense {dense_ms:6.2f} ms | "
          f"sparse {sparse_ms:6.2f} ms | speedup {dense_ms/sparse_ms:4.2f}x")

    out_d = dense(x); out_s = sparse(x)
    cos = torch.nn.functional.cosine_similarity(
        out_d.flatten(1), out_s.flatten(1)).mean().item()
    print(f"cosine similarity (no fine-tune): {cos:.4f}")


if __name__ == "__main__":
    main()
```

Decision rationale (why this shape of answer):

- **Magnitude-based `top-2 of 4` masking.** This is the simplest
  mask that satisfies the 2:4 pattern and is the one the NVIDIA
  Ampere Sparsity whitepaper uses as the baseline. More advanced
  recipes (SR-STE, movement pruning) can recover more accuracy but
  are out of scope for the exercise; the point is to produce a
  *valid* 2:4 mask and then adapt the surviving weights.
- **Mask along the input (reduction) dim.** The hardware groups
  are along the dimension being reduced in the matmul. A mask
  built along the output dim will still convert via
  `to_sparse_semi_structured`, but the packing will be wrong for
  acceleration and the speedup will collapse.
- **FP16 (or BF16), not FP32.** The Sparse Tensor Core path exists
  only for lower precisions. FP32 silently falls back to dense and
  the speedup disappears.
- **Separate microbench and end-to-end bench.** The module
  `SOLUTION.md` explicitly flags this as a required piece of
  intellectual honesty: the matmul kernel can hit near-2× and the
  end-to-end number still land around 1.3×, because non-matmul ops
  do not benefit. The exercise must show both numbers.
- **Fine-tune is sketched, not executed end-to-end.** A real
  recovery recipe needs task data, an eval harness, and usually
  thousands of steps; that lives in the training-side companion
  repo. The reference solution shows where it belongs and makes
  the mask-freeze-during-finetune step explicit.
- **Capability guard (`SM >= 8.0`).** Running the script on a V100
  silently falls back to dense and produces a confusing "1.0×
  speedup" result. The guard turns that failure into a clear
  error.

## 3. Validation steps

1. Run on an Ampere+ GPU with PyTorch ≥ 2.1:
   ```
   python sparsify_2to4.py
   ```
2. Confirm the packed representation is in use:
   ```python
   from torch.sparse import SparseSemiStructuredTensor
   assert isinstance(model.fc1.weight, SparseSemiStructuredTensor)
   ```
3. Confirm the mask is a valid 2:4 pattern on the original dense
   tensor:
   ```python
   w = linear.weight.data.view(linear.out_features, -1, 4)
   zeros_per_group = (w == 0).sum(dim=-1)
   assert (zeros_per_group == 2).all(), "Not a 2:4 pattern"
   ```
4. Verify the capability check actually fires:
   - On a V100 (SM 7.0) the assertion should trip before any
     kernel runs.
5. Record both numbers: **matmul-only speedup** (should approach
   but not exceed 2× on fp16 at large K) and **end-to-end forward
   speedup** (typically 1.3–1.5× on a transformer block). A single
   number is insufficient.
6. Profile with Nsight Compute (`ncu --kernel-id ::sparse`) to
   confirm that the kernels invoked contain the Sparse Tensor Core
   instructions (`mma.sp.*`). If `ncu` shows only dense `mma.sync`
   kernels, the packing is wrong or the backend fell back to dense.
7. Report the accuracy drop *after* a sparsity-aware fine-tune
   relative to the dense baseline on the target task. Do not
   report "no fine-tune" cosine similarity as the accuracy number;
   it under-represents what the recipe can actually achieve.

## 4. Rubric or review checklist

| Check | Weight | Pass criterion |
|---|---|---|
| Produces a valid 2:4 mask | 15 | For every group of 4 consecutive weights along the reduction dim, exactly 2 are zero |
| Masks the input (reduction) dim | 10 | Grouping is along `in_features`, not `out_features` |
| Uses FP16 or BF16 weights | 10 | Weight dtype is half-precision; FP32 variants are flagged as a fallback |
| Capability-gates the run | 5 | Script checks `cuda.get_device_capability()[0] >= 8` before proceeding |
| Converts via `to_sparse_semi_structured` | 10 | Final weight is a `SparseSemiStructuredTensor`, verified with `isinstance` |
| Mask preserved during fine-tune | 15 | The fine-tune loop re-applies the mask after each optimizer step (or uses an equivalent sparsifier that does) |
| Reports matmul **and** end-to-end speedup | 20 | Two separate numbers, labeled; end-to-end is not inflated to the matmul-only number |
| Hardware and dtype context recorded | 10 | GPU model, SM version, batch shape, and dtype appear next to any speedup number |
| Accuracy delta measured, not asserted | 5 | A real eval (or a documented placeholder) produces the "post-sparsity, post-fine-tune" number |

## 5. Common mistakes

1. **Claiming "2× speedup" from the kernel-only number.** The
   kernel can hit close to 2× on a large K; the transformer block
   containing it does not. The module `SOLUTION.md` lists this as
   the top anti-pattern for the exercise: always measure
   end-to-end.
2. **Running on pre-Ampere hardware.** On V100 or T4 the
   `to_sparse_semi_structured` call still succeeds in some PyTorch
   versions and the forward pass still runs, but the sparse matmul
   is dispatched as dense, so the "speedup" collapses to 1.0× (or
   worse, because of packing overhead).
3. **Masking along the wrong dimension.** The 2:4 groups must line
   up with the reduction dimension of the matmul (`in_features`
   for `nn.Linear`). A mask along `out_features` is technically a
   valid 2:4 pattern but is not the one the hardware expects.
4. **Pruning a dense model and shipping it without fine-tuning.**
   Zeroing half the weights of a trained transformer typically
   costs 5–15 points on downstream task metrics. The recipe is
   mask → **fine-tune with mask frozen** → convert. The reference
   solution frames this explicitly even though the bench script
   skips the fine-tune body.
5. **Not freezing the mask during fine-tune.** If the mask is
   recomputed each step from the current magnitudes, the model
   oscillates between patterns and never converges to a single 2:4
   layout. Pick the mask once (or on a schedule), then hold it
   fixed.
6. **Comparing FP32-dense to FP16-sparse latency.** The dtype
   change alone explains ~2× of the apparent speedup on compute-
   bound shapes. Compare like for like: FP16-dense vs FP16-sparse.
7. **Benchmarking without CUDA synchronization.** CUDA kernels are
   asynchronous; timing with `time.perf_counter()` and no
   `torch.cuda.synchronize()` measures launch latency, not
   execution.

## 6. References

- PyTorch tutorial — Accelerating BERT with semi-structured
  sparsity:
  <https://pytorch.org/tutorials/prototype/semi_structured_sparse.html>
- PyTorch docs — `torch.sparse.to_sparse_semi_structured`:
  <https://pytorch.org/docs/stable/generated/torch.sparse.to_sparse_semi_structured.html>
- PyTorch docs — `torch.ao.pruning` (sparsifiers):
  <https://pytorch.org/docs/stable/ao.html>
- NVIDIA developer blog — Accelerating inference with sparsity
  using the NVIDIA Ampere architecture:
  <https://developer.nvidia.com/blog/accelerating-inference-with-sparsity-using-ampere-and-tensorrt/>
- NVIDIA Ampere Architecture Whitepaper (section on Sparse Tensor
  Cores and the 2:4 pattern):
  <https://www.nvidia.com/en-us/data-center/nvidia-ampere-gpu-architecture/>
- NIST AI RMF (`measure` function): quantifying both the
  efficiency gain and the accuracy cost before adopting a
  compressed variant —
  <https://www.nist.gov/itl/ai-risk-management-framework>
- Module-level context: this solution sits beneath the "2:4
  sparsity as an Ampere/Hopper hardware feature" framing in
  `modules/mod-005-model-compression/SOLUTION.md`, which is where
  the matmul-vs-end-to-end honesty requirement comes from.
