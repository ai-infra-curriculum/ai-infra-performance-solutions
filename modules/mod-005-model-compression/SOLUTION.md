# SOLUTION — Model Compression

> Read this *after* you have quantized and fine-tuned at least one
> real model. This document explains *why* the compression toolkit
> looks the way it does and which technique to reach for first.

## What this module is really teaching

Compression is the lever that lets a 70B model fit on a single
H100 (140 GB → 70 GB INT8 → 35 GB INT4), but every compression
technique trades **accuracy** for **resource savings**, and the
trade-off curves are non-obvious:

- INT8 PTQ is nearly free for most models (< 0.5% accuracy drop).
- INT4 with naive PTQ destroys models (5-20% accuracy drop).
- INT4 with AWQ/GPTQ recovers most of it (< 1% drop) but takes
  hours to calibrate.
- 2:4 sparsity gives 2x speedup on Ampere+ with ~1% accuracy drop
  *if* you fine-tune with sparsity-aware regularization.
- Distillation is the deepest compression but the highest cost.

The module exists to give you calibrated intuition about **which
lever to pull when**, not to teach a single technique.

## Architectural decisions and *why*

### Decision 1: AWQ before GPTQ — and both before naive INT4

Exercise 01 (AWQ quantize) puts AWQ first because the API is the
gentlest (just install `autoawq`, point it at a model, get an INT4
checkpoint) and the accuracy retention is the best for instruct-
tuned LLMs. GPTQ is older, slower to calibrate, and harder to
debug.

**Anti-pattern to avoid**: running `model.half().int4()` or
equivalent naive cast and shipping. Naive INT4 on a 70B model
typically loses 8-15 points of MMLU. AWQ/GPTQ on the same model
loses 0.5-2 points.

The reference solution always benchmarks against the FP16 baseline
on a real eval (MMLU, HellaSwag, GSM8K) — not just perplexity.
Perplexity is a calibration target, not an evaluation target.

### Decision 2: INT8 static quantization as the boring-but-correct baseline

Exercise 02 (INT8 static quantization) uses post-training static
quantization with a calibration dataset. This is the **boring**
optimization: it works on almost any model with almost no accuracy
loss and gives ~2x throughput on memory-bound workloads. It's
boring because it's been the industry default for years.

The teaching value: a working INT8 baseline before fancy INT4
techniques means you can quantify *what AWQ actually buys you*.
The reference solution shows the full hierarchy: FP16 (baseline),
INT8 (1.8x throughput, < 0.5% drop), INT4-AWQ (3.5x throughput,
< 1% drop). With those numbers in front of you, deployment
decisions become easy.

### Decision 3: 2:4 sparsity as an "Ampere/Hopper hardware feature"

Exercise 03 (2:4 sparsity) covers structured sparsity — every 4
consecutive weights have at least 2 zeros. This pattern is
hardware-accelerated on A100/H100 tensor cores and gives a true 2x
speedup *on the sparse matmul*. The catch: you must fine-tune
with sparsity-aware training; you can't prune a dense model and
expect quality to hold.

The reference solution uses NVIDIA's `apex` library to apply 2:4
masking during fine-tuning, then converts the model to use the
hardware-accelerated kernel at inference time. The measured
end-to-end speedup is typically 1.3-1.5x (not 2x), because not all
ops are matmuls — softmax, layernorm, and embedding lookups don't
benefit.

**Anti-pattern to avoid**: claiming "2x speedup" because the
matmul kernel got 2x faster. Always measure end-to-end.

### Decision 4: LoRA as the "compress fine-tuning" technique

Exercise 04 (LoRA fine-tune) is in this module deliberately, even
though LoRA is usually framed as a *training* technique. The
reason: LoRA is the compression of the *update*, not the base
model. A 70B model fine-tuned with LoRA (r=16) produces a 200 MB
adapter instead of a 140 GB checkpoint, and you can hot-swap
adapters at inference time to serve many fine-tunes from one base
model.

The reference solution shows:
1. LoRA fine-tuning with `peft`.
2. Adapter merging (`model.merge_and_unload()`) for production
   deployment.
3. Multi-LoRA serving (vLLM's adapter routing) for serving multiple
   fine-tunes from one GPU.

This is the lever that makes "personalized models" economically
viable at scale.

### Decision 5: Distillation as the deepest cut

Exercise 05 (distillation) is the only technique that *changes the
model architecture*. A 70B model distills into a 7B model that
inherits most of the larger model's capabilities at 10x the
throughput. The trade-off: weeks of training, careful data curation,
and irreversible architectural commitment.

The reference solution covers:
1. **Response-based** distillation (student matches teacher
   outputs on a curated dataset).
2. **Feature-based** distillation (student matches teacher hidden
   states at chosen layers).
3. **On-policy** distillation (student generates, teacher rates).

Most production distillations end up using a hybrid. The exercise
emphasizes that you measure quality on **task-specific evals**, not
just perplexity — a distilled chat model might match the teacher
on perplexity and fail on multi-turn reasoning.

## Trade-offs we deliberately accepted

### CUDA-only quantization paths

The exercises assume CUDA + GPU inference. CPU quantization (via
GGUF, llama.cpp) is a separate world with its own tooling. The
techniques are conceptually similar (group-quantization, AWQ
ports) but the kernel landscape is entirely different.

### No FP8 here

FP8 is covered in mod-008 ex-05 because it requires Hopper (H100)
or Blackwell hardware and the calibration story is still maturing
(per-tensor vs per-block scales, dynamic vs static). For most
production workloads in 2026, INT8 + INT4-AWQ cover the use cases
that FP8 would target.

### Calibration data: small but representative

The reference solution uses 128-512 calibration samples. More is
not better past a point — diminishing returns kick in around 1024,
and the calibration time grows linearly. What matters is
**distribution coverage**: 512 samples covering your real-world
prompt distribution beats 50,000 samples of WikiText.

## Common mistakes graders see

1. **Reporting "x% accuracy drop" without specifying the eval**:
   MMLU drop and GSM8K drop are different stories for the same
   quantization. Always specify.
2. **Quantizing the embedding layer too aggressively**: embeddings
   are sensitive to quantization; keep them in higher precision
   (FP16 or INT8) even when the rest of the model is INT4.
3. **Fine-tuning a quantized model**: usually wrong. Fine-tune
   first, then quantize (PTQ). Exception: QLoRA, which is its own
   workflow.
4. **Not benchmarking the dequantized path**: some quantization
   libraries silently dequantize to FP16 inside the kernel,
   nullifying the memory-bandwidth win. Verify with ncu.
5. **Comparing latency on different batch sizes**: an INT8 model
   often runs *slower* than FP16 at batch size 1 because the
   dequantization overhead dominates. The win is at larger batches
   where memory bandwidth limits throughput.
6. **Distilling without measuring the right capability**: a 7B
   student that matches a 70B teacher on perplexity may completely
   fail on long-context retrieval. Always eval on the *target
   use case*.

## When to go beyond this implementation

- Try **GPTQ-AWQ hybrids** (each layer chooses the better
  technique based on a small calibration sweep).
- Use **EAGLE / Medusa** speculative decoding with a distilled
  draft head as a more aggressive compression of the inference
  path.
- Move to **FP8** (mod-008 ex-05) for new H100/B200 deployments
  where the hardware does the heavy lifting.

## Related curriculum touchpoints

- `performance/mod-004-transformer-optimization` — the kernels you
  are now quantizing.
- `performance/mod-007-production-deployment` — choosing among
  compressed variants based on workload.
- `engineer/mod-110-llm-infrastructure/exercise-01-production-llm-serving`
  — adapter routing for multi-LoRA serving.
- `mlops/projects/project-2-model-serving` — registry + canary
  flow for promoting a quantized model.
