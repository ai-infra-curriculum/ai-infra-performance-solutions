"""PyTorch profiler skeleton."""
import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


def profile_step(model, x):
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        on_trace_ready=tensorboard_trace_handler("./profiler_log"),
        record_shapes=True, with_stack=True,
    ) as prof:
        for _ in range(5):
            out = model(x)
            loss = out.sum()
            loss.backward()
            prof.step()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
