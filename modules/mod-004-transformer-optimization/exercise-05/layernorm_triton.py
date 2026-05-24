"""LayerNorm in Triton — fused mean/variance/normalize/scale-shift."""
import torch
import triton
import triton.language as tl


@triton.jit
def layer_norm_fwd(X, Y, W, B, Mean, Rstd, stride, N, eps,
                    BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    X_ptr = X + row * stride
    Y_ptr = Y + row * stride

    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    x = tl.load(X_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    x_centered = tl.where(mask, x - mean, 0.0)
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    tl.store(Mean + row, mean)
    tl.store(Rstd + row, rstd)

    w = tl.load(W + cols, mask=mask)
    b = tl.load(B + cols, mask=mask)
    y = x_centered * rstd * w + b
    tl.store(Y_ptr + cols, y, mask=mask)


def layer_norm(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                eps: float = 1e-5) -> torch.Tensor:
    rows, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty(rows, device=x.device)
    rstd = torch.empty(rows, device=x.device)
    BLOCK_SIZE = triton.next_power_of_2(N)
    layer_norm_fwd[(rows,)](
        x, y, w, b, mean, rstd, x.stride(0), N, eps, BLOCK_SIZE=BLOCK_SIZE,
    )
    return y
