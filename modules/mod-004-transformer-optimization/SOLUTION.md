# SOLUTION — Transformer Optimization

> Read this *after* you have benchmarked the kernels and serving
> strategies yourself. This document explains *why* the modern
> LLM serving stack is shaped the way it is: which optimizations
> compose, which are mutually exclusive, and which marketing
> claims to trust.

## What this module is really teaching

Transformer inference is the single most-optimized workload in
modern computing. A naive PyTorch implementation hits maybe 5 TFLOPS
of useful work on an H100; production stacks (vLLM, TensorRT-LLM,
SGLang) hit 700+ TFLOPS for the same workload. The 100x gap comes
from a small handful of techniques layered together:

1. **FlashAttention** — fuses attention to avoid materializing the
   N×N attention matrix.
2. **KV cache** — reuses past-token computation across decoding
   steps.
3. **Continuous batching** — packs requests at different decode
   positions into one GPU launch.
4. **Paged attention** — manages KV cache like virtual memory.
5. **Prefix caching** — shares KV cache across requests with the
   same prompt prefix.
6. **Speculative decoding** — uses a small draft model to propose
   tokens for the large model to verify.

Understanding which of these are independent (you can layer them)
vs which compete (you usually pick one) is the goal of the module.

## Architectural decisions and *why*

### Decision 1: FlashAttention first — because it changes the asymptotic story

Exercise 01 (FlashAttention) is first because every other
optimization assumes you've eliminated the O(N²) memory footprint
of standard attention. Naive attention materializes `QK^T` (N×N
floats), which for N=8192 is 256 MB *per layer per batch element*
in fp16 — saturating HBM bandwidth on every iteration.

FlashAttention tiles the attention computation along the sequence
dimension and computes softmax incrementally, never materializing
the full attention matrix. The result is the same; the memory
footprint is O(N). This is the **single most important kernel** in
modern LLM inference, and it's worth understanding the tiling
pattern in detail before treating it as a black box.

The reference solution uses `flash-attn` (v2 or v3 depending on
GPU) and benchmarks it against `torch.nn.functional.scaled_dot_product_attention`
to confirm the speedup (typically 2-4x for long sequences).

### Decision 2: torch.compile as the "free win" — with caveats

Exercise 02 (torch.compile) shows that for the right workload,
adding `@torch.compile` to a PyTorch module gives a 1.5-3x speedup
with zero code changes. The compiler (TorchDynamo + TorchInductor)
fuses operations, eliminates Python overhead, and generates
optimized Triton kernels.

**The caveat that matters**: `torch.compile` is *not* always a
free win. It recompiles on shape changes, has long compile times
(30-60s for a large model on first run), and produces stack traces
that are nearly impossible to debug. For inference servers that
see varying input shapes (variable-length sequences), the
recompilation tax can dominate. The reference solution shows both
the speedup *and* the cold-start cost so the trade-off is
explicit.

**Anti-pattern to avoid**: putting `@torch.compile` everywhere and
shipping. Profile compile-time *and* steady-state separately.

### Decision 3: vLLM prefix caching as the production lever

Exercise 03 (vLLM prefix caching) is where the module crosses from
"clever kernels" to "system architecture." Prefix caching gives a
2-10x throughput improvement for **specific workloads** — those
where many requests share a long prompt prefix (chat with a
system prompt, RAG with a retrieved context). For workloads with
unique prompts (creative writing, completion), prefix caching buys
nothing.

The reference solution benchmarks the *same model* on:

- 1000 unique prompts (prefix cache hit rate ~0%, no speedup).
- 1000 requests sharing a 4000-token system prompt (hit rate ~99%,
  3-5x throughput).

The lesson: **measure your workload before turning on the
optimization**. The optimization isn't free — it costs GPU memory
that could have been a larger KV cache for active requests.

### Decision 4: Speculative decoding as the "you pay tokens to save tokens" trick

Exercise 04 (speculative decoding) is the most counterintuitive
optimization in the stack. You run a *small* model to propose
N tokens, then the *large* model verifies all N in a single forward
pass. If the small model agreed with the large one for K tokens,
you've generated K tokens at the cost of one large-model forward
pass.

The mathematics: speedup is bounded by the **acceptance rate**
(typically 0.6-0.85 for a well-chosen draft model) and the
**relative latency** of draft vs target. On H100, with a 1B draft
and 70B target, real speedups are 1.8-2.5x for decoding throughput
on well-aligned domains. On adversarial domains (draft model knows
nothing about), it's a slowdown.

**Anti-pattern to avoid**: deploying speculative decoding without
measuring the acceptance rate on your real traffic. The reference
solution emits the acceptance rate as a first-class metric so the
deployment can be evaluated honestly.

### Decision 5: Triton kernel as the "I can write this" exercise

Exercise 05 (custom Triton kernel) implements a fused
RMSNorm+SiLU+matmul kernel (the gate/up projection in Llama-style
MLPs) and benchmarks it against the unfused PyTorch version. The
speedup is typically 1.3-1.8x — modest, but instructive: it shows
how much *just removing intermediate writes to HBM* buys you.

The deeper teaching: Triton is the language that Hugging Face,
vLLM, and SGLang use to write their hot kernels. Reading and
modifying Triton is now a baseline skill for AI infrastructure
engineers.

## Trade-offs we deliberately accepted

### vLLM as the default serving framework

The exercises use vLLM rather than TensorRT-LLM or SGLang. The
reason: vLLM has the gentlest learning curve and the widest model
support. The optimizations covered (paged attention, continuous
batching, prefix caching) are the same across all three frameworks
— what changes is the deployment surface. Mod-007 (production
deployment) covers framework selection criteria; this module
focuses on the underlying techniques.

### Single-GPU focus

Tensor and pipeline parallelism live in mod-006. The reason: every
optimization here works on a single GPU, and the test setup is
much simpler. Once these are solid, scaling to multi-GPU is largely
about choosing the right collective strategy.

### Quantization deferred to mod-005

INT8 / FP8 / INT4 quantization is its own module (mod-005). The
reason: quantization changes the precision of *every* kernel, so
it's better treated as a cross-cutting transformation than as a
single exercise.

## Common mistakes graders see

1. **Comparing FlashAttention to naive attention at N=128**: at
   short sequences, the overhead of FlashAttention's tiling
   dominates and naive attention wins. FlashAttention's payoff is
   at N ≥ 1024.
2. **Reporting throughput without specifying batch size + sequence
   length**: "100 tok/s" is meaningless. Always report as
   `tokens / (batch_size × seconds)` with the (input, output)
   length explicit.
3. **Mistaking prefill TFLOPS for decode TFLOPS**: prefill is
   compute-bound (matmuls of size B×N×H); decode is memory-bound
   (matmuls of size B×1×H, dominated by weight reads). They have
   completely different bottlenecks.
4. **Forgetting the KV cache size**: at long context, the KV cache
   dwarfs the model weights. A 70B model in fp16 is 140 GB; the
   KV cache at 32k context with batch 32 is ~640 GB. Capacity
   planning that ignores this is wrong by 5x.
5. **Trusting `torch.compile()` after the first inference batch**:
   recompilations on shape change are silent. Log compile events.
6. **Conflating throughput and latency**: continuous batching
   improves throughput by *increasing* per-request latency through
   queueing. Both metrics must be on the SLO.

## When to go beyond this implementation

- Add **chunked prefill** so long-prompt requests don't block the
  decode loop.
- Implement **structured output** (constrained decoding with a
  grammar) and measure the cost — it bites latency more than most
  people expect.
- Move to **TensorRT-LLM** for production deployments where the
  10-30% additional throughput justifies the engineering cost.

## Related curriculum touchpoints

- `performance/mod-002-cuda-programming` — the CUDA primitives that
  underlie FlashAttention and the custom Triton kernel.
- `performance/mod-005-model-compression` — quantization as the
  composable next layer on top of these optimizations.
- `performance/mod-006-distributed-inference` — scaling these
  optimizations across multiple GPUs.
- `engineer/mod-110-llm-infrastructure/exercise-01-production-llm-serving`
  — the full production gateway that uses these techniques.
