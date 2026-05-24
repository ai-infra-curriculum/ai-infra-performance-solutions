# FP8 Training — Solution

`fp8_train.py` shows TE's `fp8_autocast` + per-tensor scaling recipe. Typical: 30-50% training speedup on H100 vs bf16, with <0.5pp accuracy loss on transformer benchmarks.

Requires: H100 / B100 + Transformer Engine + Megatron-LM or NeMo for production-grade training loops.
