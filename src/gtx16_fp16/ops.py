"""Host-side wrappers: shape checks, kernel selection, launch on the current torch stream."""
from __future__ import annotations

import re

import torch

from .kernels import SKINNY_MAX_M, get_kernel, launch, stream_handle

# TU116 / TU117: Turing chips without tensor cores. Matched on the CUDA device name.
_AFFECTED = re.compile(r"GTX 16\d\d|MX ?450|MX ?550|\bT400\b|\bT600\b|\bT1000\b|T1200", re.I)
_sms: dict[int, int] = {}

# Rows whose |max| exceeds this are scaled down by a power of two before the fp16 cast (GEMM path).
# 1024 keeps 16-wide fp16 partial sums below 65504 for weights up to |w| ~ 4.
ROW_SCALE_THRESHOLD = 1024.0


def is_affected(device: torch.device | int | None = None) -> bool:
    """True when the device is a Turing GPU without tensor cores (cc 7.5, TU116/TU117)."""
    if not torch.cuda.is_available():
        return False
    idx = torch.cuda.current_device() if device is None else torch.device(device).index or 0
    if torch.cuda.get_device_capability(idx) != (7, 5):
        return False
    return bool(_AFFECTED.search(torch.cuda.get_device_name(idx)))


def supported(x2d: torch.Tensor, w_t: torch.Tensor) -> bool:
    """x2d: [M, K] fp16/fp32 contiguous CUDA; w_t: [K, N] fp16 contiguous, 16-byte aligned;
    K % 8 == 0 and N % 8 == 0."""
    K, N = w_t.shape
    return (x2d.is_cuda and x2d.dim() == 2 and x2d.dtype in (torch.float16, torch.float32)
            and w_t.dtype == torch.float16 and x2d.is_contiguous() and w_t.is_contiguous()
            and x2d.shape[1] == K and K % 8 == 0 and N % 8 == 0 and w_t.data_ptr() % 16 == 0)


def _sm_count(device: torch.device) -> int:
    i = device.index or 0
    if i not in _sms:
        _sms[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _sms[i]


def skinny_config(N: int, K: int, sms: int) -> tuple[tuple[int, int, int], int]:
    """Grid and K-chunk for the skinny kernel: enough blocks (about 4 per SM) to saturate memory."""
    nbx = (N + 255) // 256
    ksplit = max(1, min((4 * sms + nbx - 1) // nbx, K // 64))
    kchunk = ((K + ksplit - 1) // ksplit + 7) // 8 * 8
    ksplit = (K + kchunk - 1) // kchunk
    return (nbx, ksplit, 1), kchunk


def _prefilled(bias: torch.Tensor | None, M: int, N: int, device: torch.device) -> torch.Tensor:
    """New fp32 [M, N] tensor holding bias in every row (or zeros). Always a fresh allocation:
    kernels accumulate into it with atomics, so it must never alias the bias buffer."""
    out = torch.empty(M, N, device=device, dtype=torch.float32)
    if bias is None:
        out.zero_()
    else:
        out.copy_(bias.float().expand(M, N))
    return out


def mm_t(x2d: torch.Tensor, w_t: torch.Tensor, bias: torch.Tensor | None = None,
         out_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """out[M,N] = x2d[M,K] @ w_t[K,N] + bias[N].
    x2d fp16 or fp32 (fp32 preferred: no overflow on the small-M path, and large rows are rescaled
    on the GEMM path); w_t fp16; bias fp32 or None; out fp32 or fp16. Check supported() first."""
    if out_dtype not in (torch.float32, torch.float16):
        raise ValueError(f"out_dtype must be float32 or float16, got {out_dtype}")
    M, K = x2d.shape
    N = w_t.shape[1]
    dev = x2d.device
    stream = stream_handle(dev.index or 0)
    sms = _sm_count(dev)

    if M <= SKINNY_MAX_M:  # memory-bound path, fp32 math
        out = torch.zeros(M, N, device=dev, dtype=torch.float32)
        grid, kchunk = skinny_config(N, K, sms)
        name = "skinny_f32in" if x2d.dtype == torch.float32 else "skinny_f16in"
        b = None if bias is None else bias.float().contiguous()
        launch(get_kernel(name, dev), grid, (256, 1, 1), [x2d, w_t, out, b, M, N, K, kchunk], stream)
        return out if out_dtype == torch.float32 else out.to(out_dtype)

    # GEMM path: fp16 inputs, with per-row power-of-two scaling for rows that would overflow
    x16 = torch.empty(M, K, device=dev, dtype=torch.float16)
    rscale = torch.empty(M, device=dev, dtype=torch.float32)
    launch(get_kernel("rowscale_f32" if x2d.dtype == torch.float32 else "rowscale_f16", dev),
           (M, 1, 1), (256, 1, 1), [x2d, x16, rscale, K, ROW_SCALE_THRESHOLD], stream)

    tiles = ((N + 127) // 128) * ((M + 127) // 128)
    if tiles < 2 * sms and K >= 256:
        # too few output tiles to fill the GPU: split K across blocks, accumulate with fp32 atomics
        ksplit = min((2 * sms + tiles - 1) // tiles, K // 128)
        kchunk = ((K + ksplit - 1) // ksplit + 15) // 16 * 16
        ksplit = (K + kchunk - 1) // kchunk
        out = _prefilled(bias, M, N, dev)
        launch(get_kernel("hgemm_splitk", dev), ((N + 127) // 128, (M + 127) // 128, ksplit), (256, 1, 1),
               [x16, w_t, out, rscale, M, N, K, kchunk], stream)
        return out if out_dtype == torch.float32 else out.to(out_dtype)

    b = None if bias is None else bias.float().contiguous()
    out = torch.empty(M, N, device=dev, dtype=out_dtype)
    name = "hgemm_f32out" if out_dtype == torch.float32 else "hgemm_f16out"
    launch(get_kernel(name, dev), ((N + 127) // 128, (M + 127) // 128, 1), (256, 1, 1),
           [x16, w_t, out, b, rscale, M, N, K], stream)
    return out


def matmul(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype | None = None) -> torch.Tensor:
    """Drop-in for a @ b with 2-D fp16 inputs. Falls back to torch.matmul when the shape is not
    supported. Default output dtype: fp16 (same as torch)."""
    out_dtype = out_dtype or torch.float16
    if a.dim() == 2 and b.dim() == 2 and a.dtype == b.dtype == torch.float16 and a.is_cuda:
        a_c, b_c = a.contiguous(), b.contiguous()
        if supported(a_c, b_c):
            return mm_t(a_c, b_c, None, out_dtype)
    return torch.matmul(a, b).to(out_dtype)
