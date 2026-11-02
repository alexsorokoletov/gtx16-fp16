"""Instruction throughput microbenchmark: FFMA (fp32) vs HFMA2 (packed fp16) vs HMMA (mma.sync, tensor-core
instruction). On TU116/TU117 HMMA is emulated and far slower than HFMA2. Needs only PyTorch."""
import time

import torch

from gtx16_fp16.kernels import _ensure_cuda_home

SRC = r'''
#include <cuda_fp16.h>
extern "C" __global__ void ffma(float* out, float a, float b, int iters) {
  float x0 = threadIdx.x, x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
  for (int i = 0; i < iters; i++) {
    x0 = fmaf(x0, a, b); x1 = fmaf(x1, a, b); x2 = fmaf(x2, a, b); x3 = fmaf(x3, a, b);
    x4 = fmaf(x4, a, b); x5 = fmaf(x5, a, b); x6 = fmaf(x6, a, b); x7 = fmaf(x7, a, b);
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
extern "C" __global__ void hfma2(float* out, float af, float bf, int iters) {
  __half2 a = __float2half2_rn(af), b = __float2half2_rn(bf);
  __half2 x0 = __float2half2_rn(threadIdx.x * 1e-3f), x1 = __hadd2(x0, b), x2 = __hadd2(x1, b), x3 = __hadd2(x2, b);
  __half2 x4 = __hadd2(x3, b), x5 = __hadd2(x4, b), x6 = __hadd2(x5, b), x7 = __hadd2(x6, b);
  for (int i = 0; i < iters; i++) {
    x0 = __hfma2(x0, a, b); x1 = __hfma2(x1, a, b); x2 = __hfma2(x2, a, b); x3 = __hfma2(x3, a, b);
    x4 = __hfma2(x4, a, b); x5 = __hfma2(x5, a, b); x6 = __hfma2(x6, a, b); x7 = __hfma2(x7, a, b);
  }
  __half2 s = __hadd2(__hadd2(__hadd2(x0, x1), __hadd2(x2, x3)), __hadd2(__hadd2(x4, x5), __hadd2(x6, x7)));
  out[blockIdx.x * blockDim.x + threadIdx.x] = __low2float(s) + __high2float(s);
}
extern "C" __global__ void hmma(float* out, int iters) {
  unsigned a0 = 0x3c003c00u ^ threadIdx.x, a1 = 0x3c003c00u, b0 = 0x3c003c00u;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0;
  for (int i = 0; i < iters; i++)
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3) : "r"(a0), "r"(a1), "r"(b0));
  out[blockIdx.x * blockDim.x + threadIdx.x] = c0 + c1 + c2 + c3;
}
'''


def main():
    _ensure_cuda_home()
    p = torch.cuda.get_device_properties(0)
    print(f"{torch.cuda.get_device_name(0)}  cc {p.major}.{p.minor}  SMs {p.multi_processor_count}")
    blocks, threads, iters = p.multi_processor_count * 64, 256, 4096
    out = torch.zeros(blocks * threads, device="cuda")
    tests = [("ffma", [out, 0.999, 0.001, iters], threads * iters * 8 * 2),
             ("hfma2", [out, 0.999, 0.001, iters], threads * iters * 8 * 4),
             ("hmma", [out, iters], threads // 32 * iters * 16 * 8 * 8 * 2)]
    for name, args, flops_per_block in tests:
        k = torch.cuda._compile_kernel(SRC, name, nvcc_options=["--use_fast_math"])
        k(grid=(blocks, 1, 1), block=(threads, 1, 1), args=args)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(5):
            k(grid=(blocks, 1, 1), block=(threads, 1, 1), args=args)
        torch.cuda.synchronize()
        e = (time.perf_counter() - t) / 5
        print(f"{name:6} {blocks * flops_per_block / e / 1e12:6.2f} TFLOPS")


if __name__ == "__main__":
    main()
