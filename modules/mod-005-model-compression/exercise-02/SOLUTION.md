# Exercise 02 — INT8 Static Post-Training Quantization

## 1. Solution overview

The exercise asks learners to apply PyTorch post-training static
quantization (PTQ) to a pretrained vision model, calibrate it on a
small representative dataset, convert it to an INT8 graph, and
report the accuracy / size / latency trade-off against the FP32
baseline. The reference solution targets `torchvision`'s ResNet-50
with the `fbgemm` x86 backend (`qnnpack` is the ARM equivalent) and
produces a TorchScript-serialized INT8 model.

Key teaching points:

- *Static* PTQ quantizes **both weights and activations** to INT8.
  Calibration is required because activation scales are data-
  dependent; weights alone can be quantized dynamically with no
  calibration data.
- PyTorch's quantization flow is a three-step pipeline: (1) attach
  a `qconfig` (observer + fake-quant specification), (2) `prepare`
  the model so observers record activation statistics, (3) run
  calibration data through the prepared model, then `convert` to
  actual INT8 modules.
- The win is **4× smaller weights** (INT8 is one quarter of FP32)
  and roughly **2–3× faster** inference on x86 CPUs with
  oneDNN/fbgemm kernels, at the cost of a small accuracy drop
  (typically well under 1 pp of top-1 on ImageNet for ResNet-50).
- Static PTQ requires a `QuantStub` at the input and a
  `DeQuantStub` at the output; `torchvision` quantizable model
  variants already include these.

## 2. Worked answer or implementation

The script below is the canonical reference. It runs on CPU (fbgemm
is x86-only) with PyTorch ≥ 2.1 and `torchvision` ≥ 0.16. It uses
`torchvision.models.quantization.resnet50`, the quantizable variant
that already has fused `Conv+BN+ReLU` blocks and the stubs in place.

```python
# modules/mod-005-model-compression/exercise-02/quantize_static.py
import copy
import os
import time
import torch
import torch.ao.quantization as tq
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models.quantization import resnet50, ResNet50_QuantizedWeights


def build_calibration_loader(calib_dir: str, batch_size: int = 32) -> DataLoader:
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = ImageFolder(calib_dir, transform=tf)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=2)


@torch.inference_mode()
def calibrate(model: torch.nn.Module, loader: DataLoader,
              max_batches: int = 32) -> None:
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        model(x)


@torch.inference_mode()
def evaluate_top1(model: torch.nn.Module, loader: DataLoader,
                  max_batches: int = 50) -> float:
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


def model_size_mb(model: torch.nn.Module, path: str) -> float:
    torch.jit.save(torch.jit.script(model), path)
    size = os.path.getsize(path) / (1024 * 1024)
    return size


def bench(model: torch.nn.Module, x: torch.Tensor,
          warmup: int = 3, iters: int = 20) -> float:
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return (time.perf_counter() - t0) / iters


def main() -> None:
    torch.manual_seed(0)
    torch.backends.quantized.engine = "fbgemm"

    # 1. Build the FP32 baseline using the quantizable ResNet-50
    #    (has QuantStub / DeQuantStub and fuseable Conv+BN+ReLU).
    fp32 = resnet50(weights=ResNet50_QuantizedWeights.DEFAULT.url  # FP32 weights
                    if False else None, quantize=False).eval()
    # Load standard torchvision FP32 weights:
    from torchvision.models import resnet50 as plain_resnet50, ResNet50_Weights
    fp32_src = plain_resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).eval()
    fp32.load_state_dict(fp32_src.state_dict(), strict=False)

    # 2. Fuse Conv+BN+ReLU in-place for better INT8 performance.
    fp32.fuse_model(is_qat=False)

    # 3. Attach a per-tensor qconfig for fbgemm and prepare.
    prepared = copy.deepcopy(fp32)
    prepared.qconfig = tq.get_default_qconfig("fbgemm")
    tq.prepare(prepared, inplace=True)

    # 4. Calibrate with a small representative loader.
    calib_loader = build_calibration_loader(
        os.environ.get("CALIB_DIR", "data/imagenet_calib"))
    calibrate(prepared, calib_loader, max_batches=32)

    # 5. Convert to actual INT8 modules.
    int8 = tq.convert(prepared, inplace=False).eval()

    # 6. Report size, accuracy (small val slice), and latency.
    val_loader = build_calibration_loader(
        os.environ.get("VAL_DIR", "data/imagenet_val"))
    fp32_acc = evaluate_top1(fp32, val_loader, max_batches=50)
    int8_acc = evaluate_top1(int8, val_loader, max_batches=50)

    fp32_size = model_size_mb(fp32, "resnet50_fp32.pt")
    int8_size = model_size_mb(int8, "resnet50_int8.pt")

    dummy = torch.randn(1, 3, 224, 224)
    fp32_ms = bench(fp32, dummy) * 1e3
    int8_ms = bench(int8, dummy) * 1e3

    print(f"size:    fp32 {fp32_size:6.1f} MB | int8 {int8_size:6.1f} MB "
          f"| ratio {fp32_size/int8_size:4.2f}x")
    print(f"top-1:   fp32 {fp32_acc:6.4f} | int8 {int8_acc:6.4f} "
          f"| drop {(fp32_acc-int8_acc)*100:+.2f} pp")
    print(f"lat/im:  fp32 {fp32_ms:6.2f} ms | int8 {int8_ms:6.2f} ms "
          f"| speedup {fp32_ms/int8_ms:4.2f}x")


if __name__ == "__main__":
    main()
```

Decision rationale (why this shape of answer):

- **Quantizable `resnet50` variant, not plain `resnet50`.** The
  quantizable class already places `QuantStub`/`DeQuantStub` and
  exposes `fuse_model()`. Hand-rolling these on the plain variant
  works but is a large amount of boilerplate that distracts from
  the quantization concepts the exercise targets.
- **`fbgemm` backend, not `qnnpack`.** The exercise text assumes an
  x86 development machine; `qnnpack` is correct for mobile/ARM.
  The choice must be set via `torch.backends.quantized.engine`
  **and** reflected in the `qconfig`; the two must agree.
- **Per-tensor qconfig from `get_default_qconfig`.** This is the
  PyTorch-recommended starting point for static PTQ on CNNs. Per-
  channel weight quantization is the default inside the qconfig
  and matters for recovering accuracy; learners should not override
  it unless they know why.
- **32 calibration batches of 32 images = ~1024 samples.** The
  module-level discussion in `SOLUTION.md` calls out 128–512 as the
  normal range; the solution picks the upper end for a vision
  classifier with 1000-class activation spread. Diminishing returns
  kick in around 1024.
- **Serialize via `torch.jit.script` for the size measurement.**
  `state_dict` sizes are misleading for quantized models because
  quantized tensors pack differently; `torch.jit.save` of the
  scripted module reflects what a deployment would actually ship.
- **Benchmark at batch size 1.** The module `SOLUTION.md` warns that
  at batch size 1 the dequantization overhead can dominate and INT8
  may *not* beat FP32. The exercise asks learners to **measure and
  report** honestly rather than cherry-pick a batch size.

## 3. Validation steps

1. Run on x86 Linux with PyTorch ≥ 2.1 on CPU:
   ```
   CALIB_DIR=/path/to/calib VAL_DIR=/path/to/val \
     python quantize_static.py
   ```
2. Verify the three reported axes:
   - **Size ratio ≈ 4×** between FP32 and INT8 TorchScript files.
   - **Top-1 drop < 1 pp** on a 1600-image val slice for ResNet-50
     (ImageFolder with class subfolders; any 50 batches of 32).
   - **Latency speedup > 1×** at the batch size reported; if it is
     < 1× at batch size 1, re-run at batch size 8 and 32 and
     include both numbers.
3. Confirm the fbgemm engine is actually in use:
   ```python
   print(torch.backends.quantized.engine)          # "fbgemm"
   print(int8.conv1.weight().q_scheme())           # per_channel_affine
   ```
4. Sanity-check numeric equivalence on a single input:
   ```python
   out_fp32 = fp32(dummy)
   out_int8 = int8(dummy)
   cos = torch.nn.functional.cosine_similarity(
       out_fp32.flatten(1), out_int8.flatten(1)).mean()
   assert cos > 0.99, f"cosine similarity too low: {cos:.4f}"
   ```
5. Load the saved `resnet50_int8.pt` in a fresh process via
   `torch.jit.load` and confirm it runs without needing the original
   Python class definition — that is the point of scripting.

## 4. Rubric or review checklist

| Check | Weight | Pass criterion |
|---|---|---|
| Uses static PTQ (prepare → calibrate → convert), not dynamic | 15 | Code calls `tq.prepare` and runs calibration data through the prepared model before `tq.convert`; no `quantize_dynamic` |
| Backend engine matches qconfig | 10 | `torch.backends.quantized.engine` set to the same backend the qconfig was fetched for (`fbgemm` on x86, `qnnpack` on ARM) |
| Fuses Conv+BN+ReLU before preparing | 10 | `fuse_model()` (or an equivalent explicit `fuse_modules` call) runs before `prepare` |
| Calibration dataset is representative | 10 | At least 128 calibration samples drawn from the same distribution as validation; justified or logged |
| Reports size, accuracy, and latency together | 20 | All three numbers appear next to each other; latency reported alongside the batch size used |
| Honest latency reporting at B=1 | 10 | Does not hide the small- or inverted-speedup case at batch 1; if measured, it is reported |
| Serializes via TorchScript | 10 | `torch.jit.save(torch.jit.script(model), ...)` produces a loadable artifact in a fresh process |
| No training mutation during calibration | 10 | `model.eval()` and `torch.inference_mode()` or `torch.no_grad()` wrap the calibration pass; Dropout/BN do not update statistics |
| Clear FP32 baseline comparison | 5 | FP32 numbers measured on the same hardware, dataset slice, and batch size as the INT8 numbers |

## 5. Common mistakes

1. **Dynamic quantization substituted for static.** `quantize_dynamic`
   only quantizes weights and leaves activations in FP32. It needs
   no calibration data, which is why learners reach for it, but it
   does not answer the exercise and typically gives 1.3–1.8×
   speedup, not 2–3×.
2. **Skipping `fuse_model` / `fuse_modules`.** Unfused BatchNorm
   breaks quantized Conv optimization and inflates the latency of
   the INT8 model, sometimes to below FP32 at batch 1. Fuse before
   `prepare`.
3. **Mismatched backend and qconfig.** Setting
   `torch.backends.quantized.engine = "qnnpack"` but fetching
   `get_default_qconfig("fbgemm")` silently yields wrong scales and
   either a slow fallback or a hard error at `convert`. The two
   must agree.
4. **Calibrating with `torch.no_grad()` but without `model.eval()`.**
   BatchNorm still updates running stats in training mode, so the
   model drifts during calibration and the recorded activation
   ranges are inconsistent with the final weights.
5. **Reporting a size win from the state_dict instead of the
   serialized artifact.** State dict counting ignores packing /
   padding of INT8 tensors and overstates the compression ratio.
   Measure what a deployment would actually ship: the scripted
   TorchScript file size.
6. **Claiming a speedup without naming the hardware and batch
   size.** An "INT8 is 2.4× faster" claim without `(CPU, batch=32,
   threads=4)` is not reproducible. The module-level `SOLUTION.md`
   lists this as the top grading pitfall.
7. **Calibrating on data that doesn't look like the serving
   distribution.** Random ImageNet-val samples work for a generic
   ResNet, but an industrial-camera model calibrated on ImageNet
   will have activation ranges that are wrong for its real inputs.
   Representativeness beats sample count past ~128 images.

## 6. References

- PyTorch docs — Quantization overview:
  <https://pytorch.org/docs/stable/quantization.html>
- PyTorch docs — Quantization API (`torch.ao.quantization`):
  <https://pytorch.org/docs/stable/quantization-support.html>
- PyTorch tutorial — Static quantization with Eager Mode (ResNet):
  <https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html>
- PyTorch docs — Backend configuration (`fbgemm` / `qnnpack`):
  <https://pytorch.org/docs/stable/quantization-backend-configuration.html>
- torchvision docs — Quantized models:
  <https://pytorch.org/vision/stable/models.html#quantized-models>
- NIST AI RMF (`govern` + `measure`): establishing measurable
  accuracy/latency targets before accepting a quantized variant —
  <https://www.nist.gov/itl/ai-risk-management-framework>
- Module-level context: this solution sits beneath the INT8-as-
  boring-baseline framing in
  `modules/mod-005-model-compression/SOLUTION.md` and feeds the
  FP16 → INT8 → INT4-AWQ comparison table that the AWQ exercise
  (`exercise-01`) builds on top of.
