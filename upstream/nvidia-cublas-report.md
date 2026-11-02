# cuBLAS selects emulated HMMA kernels for fp16 GEMM on TU116/TU117 (GTX 16-series): fp16 GEMM 9× slower than fp32

**Status:** not filed yet. **Where to file:** NVIDIA Developer Forums → Accelerated Computing → CUDA → CUDA Programming and Performance (or the NVIDIA bug portal, developer.nvidia.com/bugs, if an account is available).

---

**Summary**

On TU116 / TU117 GPUs (GTX 1650, 1660, 1660 Ti, 1660 Super, MX450, T400/T600/T1000), cuBLAS fp16 GEMM runs at about 0.5 TFLOPS. fp32 SGEMM on the same card runs at 4.5 TFLOPS. These chips are sm_75 but have no tensor cores. They have dedicated fp16 units that execute HFMA2 at 2× the fp32 rate.

It appears that cuBLAS selects HMMA (tensor-core) kernels for every sm_75 device. On TU116/TU117 the HMMA instruction is emulated. Our microbenchmark measures it at about 1/11 of the HFMA2 rate.

**Environment**

- GPU: GeForce GTX 1660 Ti (TU116), 6 GB, 24 SMs, compute capability 7.5
- Driver: 580.178.04 (Linux)
- cuBLAS: as shipped with PyTorch 2.14.1+cu126 (CUDA 12.6), and with PyTorch 2.14.1+cu130 (CUDA 13.0). Same result with both.

**Instruction throughput** (custom kernels, NVRTC, all SMs busy)

| Instruction | Measured | Expected (spec) |
|---|---|---|
| FFMA (fp32) | 4.58 TFLOPS | ~5.4 at boost |
| HFMA2 (packed fp16) | 9.15 TFLOPS | 2× fp32 |
| HMMA (`mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32`) | 0.84 TFLOPS | n/a: no tensor cores |

**GEMM throughput, 4096×4096×4096**

| Path | TFLOPS |
|---|---|
| cuBLAS SGEMM (fp32) | 4.54 |
| cuBLAS HGEMM / GemmEx fp16 (default and cuBLASLt backends) | 0.50 |
| cuBLAS fp16 on CUDA 13.0 | 0.49 |
| Simple HFMA2-only GEMM (fp16 multiply, fp32 accumulation every 16 K-steps) | 6.41 |

The HFMA2-only kernel is about 120 lines. It does not use tensor-core instructions. It is 12.8× faster than cuBLAS fp16 on this GPU, with relative error 6e-4 (cuBLAS fp16: 2e-4).

**Reproduction**

1. `bench/repro_cublas.py` (PyTorch only, 20 lines): fp32 vs fp16 vs bf16 GEMM throughput.
2. `bench/peak.py`: FFMA / HFMA2 / HMMA instruction throughput.
3. `bench/bench_gemm.py`: cuBLAS against the HFMA2 kernel across shapes.

Code: https://github.com/alexsorokoletov/gtx16-fp16

**Expected behavior**

On sm_75 devices without tensor cores, cuBLAS should select non-HMMA kernels for fp16 GEMM, either HFMA2-based or fp32-compute kernels. fp16 GEMM should then be at least as fast as SGEMM.

**Questions**

1. Is there a supported way to make cuBLAS / cuBLASLt avoid tensor-core kernels on these devices, for example a math mode or an algorithm restriction?
2. Can the cuBLAS heuristics detect TU116/TU117 (no tensor cores) and choose non-HMMA kernels?
