# Stream Overlap — Solution

```python
import torch

s_compute = torch.cuda.Stream()
s_transfer = torch.cuda.Stream()

next_batch = next(dataloader)

for batch in dataloader:
    # Transfer next batch in parallel with current compute
    with torch.cuda.stream(s_transfer):
        next_batch_gpu = next_batch.cuda(non_blocking=True)
    with torch.cuda.stream(s_compute):
        out = model(batch.cuda(non_blocking=True))
        loss = out.sum()
        loss.backward()
    torch.cuda.current_stream().wait_stream(s_compute)
    batch = next_batch_gpu
```

Typical: 10-30% step-time reduction when DataLoader is the bottleneck.
