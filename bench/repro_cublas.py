"""Minimal reproduction (PyTorch only): fp16 GEMM is ~9x slower than fp32 GEMM on GTX 16-series (TU116/TU117).
Expected on a GPU with fast fp16: fp16 >= fp32. Run: python repro_cublas.py"""
import time

import torch


def tflops(dtype, n=4096, reps=30):
    a = torch.randn(n, n, device="cuda", dtype=dtype)
    b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5):
        a @ b
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        a @ b
    torch.cuda.synchronize()
    return 2 * n ** 3 * reps / (time.perf_counter() - t) / 1e12


print(torch.cuda.get_device_name(0), "| cc", torch.cuda.get_device_capability(0),
      "| torch", torch.__version__, "| CUDA", torch.version.cuda)
for dt in (torch.float32, torch.float16, torch.bfloat16):
    print(f"{str(dt):15} {tflops(dt):6.2f} TFLOPS")
