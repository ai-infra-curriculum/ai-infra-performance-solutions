# Int8 Static Quantization — Solution

```python
import torch
import torch.quantization as tq
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model.train(False)
model.qconfig = tq.get_default_qconfig("fbgemm")
tq.prepare(model, inplace=True)
# Calibrate on representative batches
for batch in calibration_loader:
    model(batch)
tq.convert(model, inplace=True)

torch.jit.save(torch.jit.script(model), "resnet50_int8.pt")
```

Typical: 4× smaller, 2-3× faster on x86 CPU; ~1pp accuracy loss.
