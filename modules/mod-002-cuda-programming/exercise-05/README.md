# PyTorch Extension — Solution

Full reference: [engineer-solutions/mod-107 ex-02](https://github.com/ai-infra-curriculum/ai-infra-engineer-solutions/tree/main/modules/mod-107-gpu-computing/exercise-02-cuda-kernel) — vector_add.cu + my_ops.cpp + setup.py.

Add autograd by overriding `torch.autograd.Function`:
```python
class MyMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B):
        ctx.save_for_backward(A, B)
        return my_ops.matmul(A, B)

    @staticmethod
    def backward(ctx, grad_out):
        A, B = ctx.saved_tensors
        return my_ops.matmul(grad_out, B.t()), my_ops.matmul(A.t(), grad_out)
```
