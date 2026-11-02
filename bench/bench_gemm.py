"""GEMM / skinny-GEMM benchmark: cuBLAS fp32, cuBLAS fp16, gtx16_fp16 (HFMA2). Prints a markdown table."""
import time

import torch

import gtx16_fp16 as g


def bench(fn, flops, reps=None):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter(); fn(); torch.cuda.synchronize(); one = time.perf_counter() - t
    reps = reps or max(3, min(200, int(0.5 / max(one, 1e-6))))
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    e = (time.perf_counter() - t) / reps
    return flops / e / 1e12, e * 1e6


def main():
    print(torch.cuda.get_device_name(0), "| torch", torch.__version__, "| CUDA", torch.version.cuda)
    print("| M x K x N | cuBLAS fp32 | cuBLAS fp16 | gtx16_fp16 | vs fp32 | vs fp16 | rel. error (fp16 / ours) |")
    print("|---|---|---|---|---|---|---|")
    shapes = [(1, 4096, 4096), (4, 4096, 4096), (1, 1024, 3072), (64, 1024, 1024), (512, 1024, 4096),
              (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 2048, 2048)]
    for M, K, N in shapes:
        torch.manual_seed(0)
        x = (torch.randn(M, K, device="cuda") * 0.5).half(); w = (torch.randn(K, N, device="cuda") * 0.5).half()
        xf, wf = x.float(), w.float()
        ref = xf @ wf
        fl = 2 * M * K * N
        f32, _ = bench(lambda: xf @ wf, fl)
        f16, _ = bench(lambda: x @ w, fl)
        ours, us = bench(lambda: g.mm_t(x, w, None, torch.float32), fl)
        e16 = ((x @ w).float() - ref).norm() / ref.norm()
        eo = (g.mm_t(x, w, None, torch.float32) - ref).norm() / ref.norm()
        unit = "TFLOPS" if M > 8 else "TFLOPS (memory-bound)"
        print(f"| {M}x{K}x{N} | {f32:.2f} | {f16:.2f} | **{ours:.2f}** | {ours / f32:.2f}x | {ours / f16:.1f}x | "
              f"{e16:.1e} / {eo:.1e} |")
    print("\nSkinny shapes (M <= 8) are memory-bound: the gain comes from reading fp16 weights.")


if __name__ == "__main__":
    main()
