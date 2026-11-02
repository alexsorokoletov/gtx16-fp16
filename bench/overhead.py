"""Per-call CPU cost of FastLinear vs nn.Linear at batch 1 (what token-by-token generation sees)."""
import time

import torch
import torch.nn as nn

import gtx16_fp16 as g


def cost(fn, n=1000):
    with torch.no_grad():
        for _ in range(50):
            fn()
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        cpu = (time.perf_counter() - t) / n
        torch.cuda.synchronize()
        wall = (time.perf_counter() - t) / n
    return cpu * 1e6, wall * 1e6


for K, N in [(1024, 1024), (1024, 3072), (3072, 1024), (4096, 4096)]:
    lin = nn.Linear(K, N).cuda()
    fl = g.FastLinear.from_linear(lin)
    x = torch.randn(1, 1, K, device="cuda")
    a, b = cost(lambda: lin(x)), cost(lambda: fl(x))
    print(f"{K}x{N}: nn.Linear cpu {a[0]:5.1f} us wall {a[1]:6.1f} us | FastLinear cpu {b[0]:5.1f} us wall {b[1]:6.1f} us")
