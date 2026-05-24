# 2:4 Sparsity — Solution

```python
import torch
from torch.sparse import to_sparse_semi_structured

# Apply 2:4 sparsity to a linear layer
linear = torch.nn.Linear(4096, 4096).cuda().half()
# ... train with sparsity-aware methods ...
linear.weight = torch.nn.Parameter(to_sparse_semi_structured(linear.weight))
```

Requires A100+ Sparse Tensor Cores. Reference: [PyTorch 2:4 tutorial](https://pytorch.org/tutorials/prototype/semi_structured_sparse.html).
