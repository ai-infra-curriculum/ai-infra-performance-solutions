# Triton LayerNorm — Solution

`layernorm_triton.py` implements a single-kernel fused LayerNorm forward.
Typical: matches `torch.nn.LayerNorm` performance; sometimes faster on
specific shapes.
