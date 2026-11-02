# fp16 matmul is 9× slower than fp32 on GTX 16-series (TU116/TU117): cuBLAS uses emulated tensor-core kernels

**Status:** not filed yet. **Where to file:** github.com/pytorch/pytorch/issues (template: Performance). Suggested labels: `module: cuda`, `module: cublas`, `module: half`, `triaged` is for maintainers.

---

### 🐛 Describe the bug

On GTX 16-series GPUs (TU116 / TU117, compute capability 7.5, **no tensor cores**), `torch.matmul` on fp16 tensors is about 9× slower than on fp32 tensors. bf16 (emulated) is also faster than fp16.

```python
import time, torch

def tflops(dtype, n=4096, reps=30):
    a = torch.randn(n, n, device="cuda", dtype=dtype); b = torch.randn(n, n, device="cuda", dtype=dtype)
    for _ in range(5): a @ b
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(reps): a @ b
    torch.cuda.synchronize()
    return 2 * n**3 * reps / (time.perf_counter() - t) / 1e12

for dt in (torch.float32, torch.float16, torch.bfloat16):
    print(dt, f"{tflops(dt):.2f} TFLOPS")
```

Output on a GeForce GTX 1660 Ti:

```
torch.float32   4.50 TFLOPS
torch.float16   0.50 TFLOPS
torch.bfloat16  2.45 TFLOPS
```

Convolutions show the same pattern: `Conv2d(256, 256, 3)` on 8×256×64×64 takes 9.6 ms in fp32 and 47.8 ms in fp16.

**Cause.** TU116/TU117 have no tensor cores. They have dedicated fp16 units (packed HFMA2 at 2× the fp32 rate). cuBLAS appears to choose HMMA (tensor-core) kernels for all sm_75 devices, and HMMA is emulated on these chips. Instruction microbenchmark: FFMA 4.6 TFLOPS, **HFMA2 9.2 TFLOPS**, **HMMA 0.84 TFLOPS**.

The fp16 hardware is fast. A ~120-line HFMA2-only GEMM (fp16 multiply, fp32 accumulation every 16 K-steps) reaches **6.4 TFLOPS** on 4096³, 12.8× faster than `torch.matmul` fp16 and 1.4× faster than fp32. Relative error is 6e-4. Code and benchmarks: https://github.com/alexsorokoletov/gtx16-fp16.

Both cuBLAS backends (`torch.backends.cuda.preferred_blas_library("cublas")` and `("cublaslt")`) give the same 0.5 TFLOPS. The CUDA 13.0 build gives the same result.

**Impact.** GTX 1650/1660 cards are still common in consumer and low-cost setups. On these GPUs, common advice ("use `.half()` / autocast to speed up inference") makes models about 9× slower. Users usually blame the model.

**Possible actions** (open to maintainer judgment):

1. Report to NVIDIA as a cuBLAS heuristic bug. A report is drafted (not filed yet): `upstream/nvidia-cublas-report.md` in the repo above.
2. In PyTorch: on sm_75 devices without tensor cores, route fp16 GEMM through fp32 compute (cast or `cublasGemmEx` with fp32 compute type and a non-tensor-op algorithm), if cuBLAS exposes a way to avoid HMMA.
3. At minimum, document the behavior, or emit a one-time warning when `autocast(dtype=torch.float16)` is used on TU116/TU117.

### Versions

- PyTorch 2.14.1+cu126 (also 2.14.1+cu130)
- GPU: NVIDIA GeForce GTX 1660 Ti (TU116), driver 580.178.04, Linux (Docker, rootless)
- Python 3.12.3
