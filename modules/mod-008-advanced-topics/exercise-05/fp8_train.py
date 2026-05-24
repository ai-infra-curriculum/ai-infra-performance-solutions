"""FP8 training with NVIDIA Transformer Engine."""
import torch
import transformer_engine.pytorch as te
from transformer_engine.common import recipe


fp8_recipe = recipe.DelayedScaling(
    fp8_format=recipe.Format.HYBRID,    # E4M3 forward, E5M2 backward
    margin=0, interval=1, amax_history_len=16, amax_compute_algo="max",
)


class FP8Block(torch.nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear1 = te.Linear(d_model, d_model * 4)
        self.linear2 = te.Linear(d_model * 4, d_model)

    def forward(self, x):
        return self.linear2(torch.nn.functional.gelu(self.linear1(x)))


def train_step(model, optimizer, x):
    with te.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model(x)
        loss = out.sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
