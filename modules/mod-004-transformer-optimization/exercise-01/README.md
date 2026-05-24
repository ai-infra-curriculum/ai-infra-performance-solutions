# FlashAttention — Solution

`flash_attn.py` shows the 1-line drop-in via `F.scaled_dot_product_attention`.
At seq=8192, expect 4-8× speedup + dramatic memory reduction vs vanilla.
