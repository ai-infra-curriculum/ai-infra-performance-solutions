# torch.compile — Solution

```python
model = torch.compile(model, mode="reduce-overhead")
# Warm up: first call recompiles
for _ in range(3): model(x)
# Then measure
```

Typical: 1.3-2× speedup on inference paths.
