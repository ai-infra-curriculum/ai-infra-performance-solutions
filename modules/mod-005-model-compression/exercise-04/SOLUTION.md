# Exercise 04 — LoRA Fine-Tune

## 1. Solution overview

The exercise asks learners to adapt a pretrained causal language
model to a new task using **Low-Rank Adaptation (LoRA)**, save the
resulting adapter as a small standalone artifact, and demonstrate
both (a) running inference with the adapter attached to the frozen
base model and (b) merging the adapter back into the base weights
for a single-checkpoint deployment path. The reference solution
uses Hugging Face `transformers`, `peft`, and `datasets` and keeps
the training loop small enough to fit in the module's
single-exercise scope.

Key teaching points:

- LoRA freezes the base weights `W` and learns two small matrices
  `A ∈ ℝ^{r×k}` and `B ∈ ℝ^{d×r}`, applying the update
  `ΔW = (α/r) · B · A` at the target linear layers. Only `A` and
  `B` are trained; `A` is initialized at random and `B` at zero,
  so the model starts identical to the base.
- The compression is of the **update**, not the base. A LoRA
  adapter for a 7B model with `r=16` is ~8–50 MB (vs ~14 GB for
  the base in FP16). Many adapters can share one base at inference
  time.
- There are two deployment modes:
  - **Attached**: load base + adapter; incurs a small per-forward
    overhead but allows hot-swapping adapters per request.
  - **Merged**: fold `ΔW` into `W` with
    `model.merge_and_unload()`; produces a plain `transformers`
    checkpoint with no runtime overhead, but loses the ability to
    swap or stack adapters.
- `peft.LoraConfig.target_modules` selects which linear layers
  receive adapters. For decoder-only LLMs, the common choice is
  the attention projections (`q_proj`, `k_proj`, `v_proj`,
  `o_proj`); adding the MLP projections (`gate_proj`, `up_proj`,
  `down_proj`) increases capacity at the cost of adapter size.

Scope note: this module exercise covers the single-GPU case.
Deeper variants — QLoRA 4-bit on 13B, multi-GPU DDP, and vLLM
adapter hot-swap serving — live in the engineer-track companion
solution and are referenced in Section 6.

## 2. Worked answer or implementation

The script below is the canonical reference. It runs on a single
GPU with ≥ 16 GB VRAM for a 1–3 B base model (TinyLlama,
`SmolLM2-1.7B`, or similar) in FP16 or BF16. For a 7 B base model
the exercise expects QLoRA (4-bit base) — the knob to flip is
`load_in_4bit=True` plus `prepare_model_for_kbit_training`; see
Section 6 for the full QLoRA companion.

```python
# modules/mod-005-model-compression/exercise-04/lora_finetune.py
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


BASE = os.environ.get("BASE_MODEL", "HuggingFaceTB/SmolLM2-1.7B")
OUT  = os.environ.get("OUT_DIR",    "out/lora-smollm2-1.7b")


def build_dataset(tokenizer, max_len: int = 512):
    # Small instruction-tuning slice — any supervised chat dataset works.
    ds = load_dataset("yahma/alpaca-cleaned", split="train[:2000]")

    def fmt(ex):
        prompt = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n{ex['output']}"
        )
        return tokenizer(prompt, truncation=True, max_length=max_len)

    return ds.map(fmt, remove_columns=ds.column_names)


def main() -> None:
    tok = AutoTokenizer.from_pretrained(BASE)
    tok.pad_token = tok.pad_token or tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    # Required for gradient checkpointing to propagate through inputs
    # after the base is frozen.
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # ~0.1–1% of total parameters

    train_ds = build_dataset(tok)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    args = TrainingArguments(
        output_dir=OUT,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,       # effective batch 16
        learning_rate=2e-4,                   # LoRA tolerates higher LR
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_steps=200,
        bf16=True,
        optim="adamw_torch",
        report_to="none",
    )

    Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
    ).train()

    # Save ONLY the adapter (the small artifact).
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)


if __name__ == "__main__":
    main()
```

Inference and merge are separate entry points so the two
deployment modes stay obvious:

```python
# modules/mod-005-model-compression/exercise-04/infer_lora.py
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "HuggingFaceTB/SmolLM2-1.7B"
ADAPTER = "out/lora-smollm2-1.7b"

tok = AutoTokenizer.from_pretrained(BASE)
base = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=torch.bfloat16, device_map="auto")

# --- Mode A: attached (adapter stays separate; swappable per request) ---
attached = PeftModel.from_pretrained(base, ADAPTER)
attached.eval()

# --- Mode B: merged (adapter folded into base weights) ---
merged = PeftModel.from_pretrained(base, ADAPTER).merge_and_unload()
merged.save_pretrained("out/merged-smollm2-1.7b")
```

Decision rationale (why this shape of answer):

- **`r=16`, `alpha=32`.** This is the Hugging Face / `peft`
  reference default for decoder-only LLMs and sits in the "big
  enough to adapt, small enough to stay cheap" sweet spot for 1–7 B
  bases. Lower ranks (`r=8`) under-fit on instruction tuning;
  higher ranks (`r=64`) stop helping and inflate the adapter.
- **Attention-only target modules by default.** `q/k/v/o` captures
  most of the capacity benefit for ~4× less trainable parameters
  than also adapting the MLP block. Learners should know the
  trade-off; the comment in `target_modules` is the place to
  record which experiment they ran.
- **`lora_alpha / r` is the effective scale, not `alpha` alone.**
  The common Hugging Face pattern is `alpha = 2 * r`, giving a
  scale of 2. Raising `r` without also raising `alpha` silently
  attenuates the update.
- **`bias="none"`.** Training the LoRA biases adds trainable
  parameters that rarely pay for themselves for causal LM
  adaptation. Leaving biases frozen keeps the adapter small.
- **`gradient_checkpointing_enable()` + `enable_input_require_grads()`.**
  Gradient checkpointing halves activation memory, which is what
  lets a 7 B base + LoRA fit on a 24 GB consumer card in BF16.
  The `enable_input_require_grads` call is specifically required
  when the base is frozen; without it, autograd sees no gradient
  path into the checkpointed region and training silently
  produces zero loss updates.
- **`bf16=True`.** BF16 is numerically safer than FP16 for
  training (no loss-scaling needed) on Ampere+ hardware. Fall back
  to FP16 only when training on pre-Ampere GPUs.
- **Two separate scripts for the two deployment modes.** Attaching
  vs merging are different operational postures — the reference
  solution makes that explicit rather than hiding it behind a
  single flag.

## 3. Validation steps

1. Confirm trainable-parameter ratio is small:
   ```
   trainable params: X || all params: Y || trainable%: 0.1–1.0
   ```
   Printed by `print_trainable_parameters()`. A ratio above a few
   percent means `target_modules` is too broad or `r` is too high.
2. Train for the configured epochs and check that training loss
   trends down and perplexity on a held-out slice improves relative
   to the base. The bar is "improved on the target task", not "beat
   SOTA".
3. Verify the saved adapter is small:
   ```
   du -sh out/lora-smollm2-1.7b/adapter_model.safetensors
   ```
   For `r=16` on a 1–2 B base this is typically 10–50 MB.
4. Run `infer_lora.py` in attached mode and confirm the outputs
   differ from the base model on prompts covered by the fine-tune
   distribution.
5. Merge with `merge_and_unload()` and confirm numeric equivalence
   between attached and merged outputs on the same prompt:
   ```python
   torch.testing.assert_close(attached(**x).logits,
                              merged(**x).logits,
                              rtol=1e-3, atol=1e-3)
   ```
   bf16 tolerances are intentional; fp32 should match much more
   tightly.
6. Load the saved adapter in a fresh process (no training state in
   memory) and confirm it still produces the expected outputs.
   This catches "I only tested in the training process" mistakes.
7. For QLoRA variants: confirm the base is in 4-bit (`load_in_4bit`
   or `BitsAndBytesConfig(load_in_4bit=True)`) **before** calling
   `get_peft_model`. Quantizing after is a common and silent bug.

## 4. Rubric or review checklist

| Check | Weight | Pass criterion |
|---|---|---|
| Base weights are frozen; only adapter trains | 15 | `print_trainable_parameters()` reports well under 5% trainable; `requires_grad` is False on base params |
| `LoraConfig` is correct for the task | 10 | `task_type=CAUSAL_LM` for decoder LLMs; `target_modules` names real attention / MLP projections |
| Reasonable `(r, alpha)` pair | 10 | `r ∈ [8, 32]`, `alpha ≈ 2r` (or justified); ratio documented |
| Gradient checkpointing works with frozen base | 10 | `enable_input_require_grads()` present when checkpointing is on, or demonstrated non-zero loss gradient without it |
| Adapter saved as a small standalone artifact | 10 | `save_pretrained()` writes `adapter_model.safetensors` + `adapter_config.json`; total size is a small fraction of the base |
| Loads in a fresh process | 10 | `PeftModel.from_pretrained(base, path)` in a new Python process produces the adapter-driven outputs |
| Merge path works and matches attached outputs | 15 | `merge_and_unload()` produces a plain HF model whose logits match the attached model within bf16 tolerance |
| Task metric improves over the base | 10 | A measurable delta (held-out loss, task eval, or domain-specific metric) is reported, not just "it trained" |
| Hardware and dtype context recorded | 5 | GPU model, dtype (bf16/fp16/4-bit), and effective batch size appear next to any reported metric |
| QLoRA variant (if attempted) orders ops correctly | 5 | Base quantized to 4-bit before `get_peft_model`; `prepare_model_for_kbit_training` applied |

## 5. Common mistakes

1. **Saving the full merged model instead of the adapter.** Calling
   `save_pretrained()` on the merged model (several GB) defeats
   the compression-of-update framing. Save the adapter artifact
   first; derive the merged model only when you need one.
2. **Loading the base in FP32 for training.** This doubles the
   memory footprint and typically triggers OOM for 7 B bases on a
   24 GB card. Load in BF16 (or 4-bit for QLoRA) and keep LoRA
   parameters in FP32 — `peft` does this split automatically.
3. **Omitting `enable_input_require_grads()` with gradient
   checkpointing + frozen base.** Autograd detects no grad path
   into the checkpointed region and the loss updates are silently
   zero. Symptom: training loss is flat; `grad_norm` is tiny from
   step 1.
4. **Attaching LoRA to the wrong modules.** `target_modules=
   ["Linear"]` or similar typo-prone broad patterns can match
   every linear in the model, including the LM head, which usually
   hurts generation quality. Name the attention/MLP projections
   explicitly.
5. **Raising `r` without raising `alpha`.** The effective scale is
   `alpha / r`. Doubling `r` while holding `alpha` fixed halves
   the update magnitude and the adapter stops learning.
6. **Not comparing attached vs merged outputs.** Users ship the
   merged model, so differences between the two paths — if they
   exist — become production bugs. Assert numeric equivalence
   before trusting the merge.
7. **QLoRA order-of-operations bug.** Quantizing the base model
   *after* `get_peft_model` wraps it leaves the adapter in FP16
   but destroys the base's precision in a way `peft` cannot
   compensate for. Quantize first, call
   `prepare_model_for_kbit_training`, then wrap.
8. **Evaluating on the training distribution.** If the adapter is
   trained and evaluated on the same split, it will look better
   than it is. Hold out a slice (or an entire eval set) and
   report on that.

## 6. References

- `peft` documentation — LoRA guide:
  <https://huggingface.co/docs/peft/main/en/conceptual_guides/lora>
- `peft` documentation — Working with adapters
  (attach / merge / stack):
  <https://huggingface.co/docs/peft/main/en/tutorial/peft_model_config>
- Hugging Face Transformers — `Trainer` API:
  <https://huggingface.co/docs/transformers/main_classes/trainer>
- `bitsandbytes` — 4-bit quantization for QLoRA:
  <https://huggingface.co/docs/bitsandbytes/main/en/index>
- PyTorch docs — gradient checkpointing:
  <https://pytorch.org/docs/stable/checkpoint.html>
- NIST AI RMF (`map` + `measure` functions): defining a target
  behavior for the adapter and a held-out metric before fine-
  tuning —
  <https://www.nist.gov/itl/ai-risk-management-framework>
- Deeper engineer-track companion (QLoRA 4-bit on 13B, multi-GPU
  DDP, vLLM adapter hot-swap serving) as a practitioner
  implementation example:
  <https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-110-llm-infrastructure/exercise-05-finetuning-at-scale>
- Module-level context: this solution sits beneath the "LoRA as
  the compress-fine-tuning technique" framing in
  `modules/mod-005-model-compression/SOLUTION.md`, which is where
  the attached-vs-merged deployment discussion and the "LoRA is
  compression of the update, not the base" framing come from.
