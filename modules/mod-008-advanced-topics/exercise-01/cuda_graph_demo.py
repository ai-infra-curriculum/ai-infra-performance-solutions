"""CUDA Graph capture + replay for inference."""
import torch


def capture_graph(model, x):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        # Warmup
        for _ in range(3): _ = model(x)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        static_out = model(x)
    return g, static_out


if __name__ == "__main__":
    model = torch.nn.Linear(4096, 4096).cuda()
    static_input = torch.randn(32, 4096, device="cuda")
    g, out = capture_graph(model, static_input)

    import time
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(1000):
        g.replay()
    torch.cuda.synchronize()
    print(f"graph replay: {(time.perf_counter() - t0) * 1000:.2f}ms total")
