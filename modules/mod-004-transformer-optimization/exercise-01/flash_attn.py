"""Drop-in FlashAttention via torch's scaled_dot_product_attention."""
import torch
import torch.nn.functional as F


def vanilla_attention(q, k, v, is_causal=True):
    scale = q.size(-1) ** -0.5
    attn = q @ k.transpose(-2, -1) * scale
    if is_causal:
        mask = torch.triu(torch.ones_like(attn, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(mask, float("-inf"))
    attn = attn.softmax(dim=-1)
    return attn @ v


def flash_attention(q, k, v, is_causal=True):
    """Replaces O(N^2) memory with O(N) via FlashAttention."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)


if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    for seq in (512, 2048, 8192):
        q = torch.randn(1, 32, seq, 64, device="cuda", dtype=torch.bfloat16)
        k = torch.randn(1, 32, seq, 64, device="cuda", dtype=torch.bfloat16)
        v = torch.randn(1, 32, seq, 64, device="cuda", dtype=torch.bfloat16)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(10): out = flash_attention(q, k, v)
        torch.cuda.synchronize()
        print(f"seq={seq:5d}  flash={time.perf_counter()-t0:.3f}s")
