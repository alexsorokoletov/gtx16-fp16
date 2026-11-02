"""CUDA kernels for fp16 matmul on GPUs whose fp16 units are fast but whose tensor-core (HMMA)
instruction is emulated: TU116 / TU117 (GTX 1650, 1660, 1660 Ti, 1660 Super, MX450, T400/T600/T1000).

cuBLAS selects HMMA (tensor-core) kernels for every sm_75 device. On TU116/TU117 these run ~11x slower
than the chip's packed-fp16 FMA (HFMA2) units, so fp16 GEMM ends up ~9x slower than fp32.
These kernels use HFMA2 only and never emit HMMA.

Compiled at runtime with NVRTC through torch.cuda._compile_kernel (no nvcc, no extra packages).
"""
from __future__ import annotations

import ctypes
import glob
import os
import threading

import torch

_SRC = r"""
#include <cuda_fp16.h>

// ---------------------------------------------------------------------------------------------
// GEMM: C[M,N] = A[M,K] @ B[K,N] (+ bias[N]); A, B fp16 row-major; C fp32 or fp16.
// Tile 128x128x16, 256 threads, 8x8 outputs per thread. Products are accumulated with packed
// fp16 FMA (HFMA2) over each 16-wide K slice, then folded into fp32 registers ("hybrid"
// accumulation): about 2x faster than fp32 FFMA, error close to fp32 accumulation.
// Requirements (checked on the host): K % 8 == 0, N % 8 == 0, 16-byte aligned pointers.
// ---------------------------------------------------------------------------------------------
#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

template <bool OUT_F32, bool SPLITK>
__device__ __forceinline__ void hgemm_body(const __half* __restrict__ A, const __half* __restrict__ B,
                                           void* __restrict__ Cv, const float* __restrict__ bias,
                                           const float* __restrict__ rscale, int M, int N, int K, int kchunk) {
  // split-K: this block covers K range [kb, ke); A and B pointers are offset so the loop runs 0..klen
  const int kb = SPLITK ? blockIdx.z * kchunk : 0;
  const int klen = SPLITK ? min(K - kb, kchunk) : K;
  const int ldA = K;
  A += kb; B += (size_t)kb * N;
  __shared__ __align__(16) __half As[2][BK][BM + 8];
  __shared__ __align__(16) __half Bs[2][BK][BN + 8];
  const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
  const int bm = blockIdx.y * BM, bn = blockIdx.x * BN;

  __half2 acc[TM][TN / 2];
  float accf[TM][TN];
#pragma unroll
  for (int i = 0; i < TM; i++) {
#pragma unroll
    for (int j = 0; j < TN / 2; j++) acc[i][j] = __float2half2_rn(0.f);
#pragma unroll
    for (int j = 0; j < TN; j++) accf[i][j] = 0.f;
  }

  const int ar = tid / 2, ac = (tid % 2) * 8;    // A tile 128x16: 8 halves along K per thread
  const int br = tid / 16, bc = (tid % 16) * 8;  // B tile 16x128: 8 halves along N per thread
  const uint4 zero4 = make_uint4(0, 0, 0, 0);

  auto load = [&](int buf, int k0) {
    const int arow = bm + ar, acol = k0 + ac;
    uint4 av = (arow < M && acol < klen) ? *reinterpret_cast<const uint4*>(A + (size_t)arow * ldA + acol) : zero4;
    const __half* ah = reinterpret_cast<const __half*>(&av);
#pragma unroll
    for (int i = 0; i < 8; i++) As[buf][ac + i][ar] = ah[i];
    const int brow = k0 + br, bcol = bn + bc;
    uint4 bv = (brow < klen && bcol < N) ? *reinterpret_cast<const uint4*>(B + (size_t)brow * N + bcol) : zero4;
    *reinterpret_cast<uint4*>(&Bs[buf][br][bc]) = bv;
  };

  load(0, 0);
  __syncthreads();
  int buf = 0;
  for (int k0 = 0; k0 < klen; k0 += BK) {
    if (k0 + BK < klen) load(buf ^ 1, k0 + BK);
#pragma unroll
    for (int k = 0; k < BK; k++) {
      uint4 a4 = *reinterpret_cast<const uint4*>(&As[buf][k][ty * TM]);
      uint4 b4 = *reinterpret_cast<const uint4*>(&Bs[buf][k][tx * TN]);
      const __half* a = reinterpret_cast<const __half*>(&a4);
      const __half2* b = reinterpret_cast<const __half2*>(&b4);
#pragma unroll
      for (int i = 0; i < TM; i++) {
        const __half2 a2 = __half2half2(a[i]);
#pragma unroll
        for (int j = 0; j < TN / 2; j++) acc[i][j] = __hfma2(a2, b[j], acc[i][j]);
      }
    }
#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
      for (int j = 0; j < TN / 2; j++) {
        const float2 f = __half22float2(acc[i][j]);
        accf[i][2 * j] += f.x;
        accf[i][2 * j + 1] += f.y;
        acc[i][j] = __float2half2_rn(0.f);
      }
    }
    __syncthreads();
    buf ^= 1;
  }

  const int col = bn + tx * TN;
  if (col >= N) return;
  float bv[TN];
#pragma unroll
  for (int j = 0; j < TN; j++) bv[j] = (!SPLITK && bias) ? bias[col + j] : 0.f;
#pragma unroll
  for (int i = 0; i < TM; i++) {
    const int row = bm + ty * TM + i;
    if (row >= M) break;
    // undo the per-row power-of-two input scaling applied by rowscale_cast (exact in fp32)
    const float rs = rscale ? rscale[row] : 1.f;
#pragma unroll
    for (int j = 0; j < TN; j++) accf[i][j] *= rs;
    if (SPLITK) {
      float* C = reinterpret_cast<float*>(Cv) + (size_t)row * N + col;
#pragma unroll
      for (int j = 0; j < TN; j++) atomicAdd(C + j, accf[i][j]);
    } else if (OUT_F32) {
      float* C = reinterpret_cast<float*>(Cv) + (size_t)row * N + col;
      float4 o0 = make_float4(accf[i][0] + bv[0], accf[i][1] + bv[1], accf[i][2] + bv[2], accf[i][3] + bv[3]);
      float4 o1 = make_float4(accf[i][4] + bv[4], accf[i][5] + bv[5], accf[i][6] + bv[6], accf[i][7] + bv[7]);
      reinterpret_cast<float4*>(C)[0] = o0;
      reinterpret_cast<float4*>(C)[1] = o1;
    } else {
      __half* C = reinterpret_cast<__half*>(Cv) + (size_t)row * N + col;
      __half2 o[TN / 2];
#pragma unroll
      for (int j = 0; j < TN / 2; j++) o[j] = __floats2half2_rn(accf[i][2 * j] + bv[2 * j], accf[i][2 * j + 1] + bv[2 * j + 1]);
      *reinterpret_cast<uint4*>(C) = *reinterpret_cast<uint4*>(o);
    }
  }
}

extern "C" __global__ void __launch_bounds__(256)
hgemm_f32out(const __half* A, const __half* B, float* C, const float* bias, const float* rscale, int M, int N, int K) {
  hgemm_body<true, false>(A, B, C, bias, rscale, M, N, K, K);
}
// split-K: C (fp32) must be pre-filled with bias or zeros; gridDim.z = number of K splits
extern "C" __global__ void __launch_bounds__(256)
hgemm_splitk(const __half* A, const __half* B, float* C, const float* rscale, int M, int N, int K, int kchunk) {
  hgemm_body<true, true>(A, B, C, nullptr, rscale, M, N, K, kchunk);
}
extern "C" __global__ void __launch_bounds__(256)
hgemm_f16out(const __half* A, const __half* B, __half* C, const float* bias, const float* rscale, int M, int N, int K) {
  hgemm_body<false, false>(A, B, C, bias, rscale, M, N, K, K);
}

// ---------------------------------------------------------------------------------------------
// Skinny GEMM for small M (token-by-token generation): C[M,N] += A[M,K] @ B[K,N], fp32 accumulate.
// Memory-bound: the win comes from reading fp16 weights (half the bytes of fp32).
// C must be zeroed; bias (or null) is added once by the blocks with blockIdx.y == 0. Split-K
// across gridDim.y, fp32 atomics.
// Block: 256 threads = 32 column groups (8 cols each, 256 cols) x 8 K-lanes.
// ---------------------------------------------------------------------------------------------
#define SK_MMAX 8
__device__ __forceinline__ float to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ float to_f(float x) { return x; }
template <typename TA>
__device__ __forceinline__ void skinny_body(const TA* __restrict__ A, const __half* __restrict__ B,
                                            float* __restrict__ C, const float* __restrict__ bias,
                                            int M, int N, int K, int kchunk) {
  __shared__ float red[8][32 * 8];
  const int tx = threadIdx.x % 32, ty = threadIdx.x / 32;
  const int col = blockIdx.x * 256 + tx * 8;
  const int kb = blockIdx.y * kchunk, ke = min(K, kb + kchunk);
  float acc[SK_MMAX][8];
#pragma unroll
  for (int m = 0; m < SK_MMAX; m++)
#pragma unroll
    for (int j = 0; j < 8; j++) acc[m][j] = 0.f;
  if (col < N) {
    for (int k = kb + ty; k < ke; k += 8) {
      const uint4 b4 = *reinterpret_cast<const uint4*>(B + (size_t)k * N + col);
      const __half2* b = reinterpret_cast<const __half2*>(&b4);
      float2 bf[4];
#pragma unroll
      for (int j = 0; j < 4; j++) bf[j] = __half22float2(b[j]);
#pragma unroll
      for (int m = 0; m < SK_MMAX; m++) {
        if (m < M) {
          const float a = to_f(A[(size_t)m * K + k]);
#pragma unroll
          for (int j = 0; j < 4; j++) { acc[m][2 * j] += a * bf[j].x; acc[m][2 * j + 1] += a * bf[j].y; }
        }
      }
    }
  }
#pragma unroll
  for (int m = 0; m < SK_MMAX; m++) {
    if (m >= M) break;
#pragma unroll
    for (int j = 0; j < 8; j++) red[ty][tx * 8 + j] = acc[m][j];
    __syncthreads();
    if (ty == 0 && col < N) {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float s = 0.f;
#pragma unroll
        for (int t = 0; t < 8; t++) s += red[t][tx * 8 + j];
        if (bias && blockIdx.y == 0) s += bias[col + j];
        atomicAdd(C + (size_t)m * N + col + j, s);
      }
    }
    __syncthreads();
  }
}
extern "C" __global__ void __launch_bounds__(256)
skinny_f16in(const __half* A, const __half* B, float* C, const float* bias, int M, int N, int K, int kchunk) {
  skinny_body<__half>(A, B, C, bias, M, N, K, kchunk);
}
extern "C" __global__ void __launch_bounds__(256)
skinny_f32in(const float* A, const __half* B, float* C, const float* bias, int M, int N, int K, int kchunk) {
  skinny_body<float>(A, B, C, bias, M, N, K, kchunk);
}

// ---------------------------------------------------------------------------------------------
// rowscale_cast: Y[M,K] (fp16) = X / S[row], S[row] = 2^ceil(log2(amax/thresh)) if a row's |max|
// exceeds thresh, else 1. Keeps fp16 inputs and the 16-wide fp16 partial sums of the GEMM inside
// fp16 range for layers with "massive activations" (e.g. LLM MLP down_proj inputs > 65504).
// Power-of-two scales make the scaling exact. One block (256 threads) per row.
// ---------------------------------------------------------------------------------------------
template <typename TX>
__device__ __forceinline__ void rowscale_body(const TX* __restrict__ X, __half* __restrict__ Y,
                                              float* __restrict__ S, int K, float thresh) {
  __shared__ float red[256];
  const int row = blockIdx.x;
  const TX* x = X + (size_t)row * K;
  float m = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x) m = fmaxf(m, fabsf(to_f(x[k])));
  red[threadIdx.x] = m;
  __syncthreads();
  for (int st = 128; st > 0; st >>= 1) {
    if (threadIdx.x < st) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + st]);
    __syncthreads();
  }
  const float amax = red[0];
  const float sc = amax > thresh ? exp2f(ceilf(log2f(amax / thresh))) : 1.f;
  const float inv = 1.f / sc;
  for (int k = threadIdx.x; k < K; k += blockDim.x) Y[(size_t)row * K + k] = __float2half_rn(to_f(x[k]) * inv);
  if (threadIdx.x == 0) S[row] = sc;
}
extern "C" __global__ void __launch_bounds__(256)
rowscale_f32(const float* X, __half* Y, float* S, int K, float thresh) { rowscale_body<float>(X, Y, S, K, thresh); }
extern "C" __global__ void __launch_bounds__(256)
rowscale_f16(const __half* X, __half* Y, float* S, int K, float thresh) { rowscale_body<__half>(X, Y, S, K, thresh); }
"""

SKINNY_MAX_M = 8
_lock = threading.Lock()
_cache: dict[tuple[int, str], object] = {}


def _ensure_cuda_home() -> None:
    """torch's NVRTC wrapper needs CUDA_HOME/include with cuda_fp16.h. pip wheels ship it under
    site-packages/nvidia/cuda_runtime; point CUDA_HOME there when no toolkit is installed."""
    import site

    def has_header(d: str) -> bool:
        return os.path.exists(os.path.join(d, "include", "cuda_fp16.h"))

    if has_header(os.environ.get("CUDA_HOME", "")):
        return
    candidates = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        candidates += glob.glob(os.path.join(root, "nvidia", "cuda_runtime"))
        candidates += glob.glob(os.path.join(root, "nvidia", "cu*"))
    candidates.append("/usr/local/cuda")
    for d in candidates:
        if has_header(d):
            os.environ["CUDA_HOME"] = d
            return


_libcuda = None


def _cuLaunchKernel():
    """The driver's cuLaunchKernel with ctypes signatures set (looked up once)."""
    global _libcuda
    if _libcuda is None:
        f = torch.cuda._utils._get_gpu_runtime_library().cuLaunchKernel
        f.argtypes = [ctypes.c_void_p] + [ctypes.c_uint] * 6 + [ctypes.c_uint, ctypes.c_void_p,
                                                                 ctypes.c_void_p, ctypes.c_void_p]
        f.restype = ctypes.c_int
        _libcuda = f
    return _libcuda


def stream_handle(device_index: int) -> int:
    """Raw handle of torch's current CUDA stream (much cheaper than torch.cuda.current_stream())."""
    return torch._C._cuda_getCurrentRawStream(device_index)


def _launch_raw(func, grid: tuple[int, int, int], block: tuple[int, int, int], stream: int, ptrs) -> None:
    err = _cuLaunchKernel()(func, grid[0], grid[1], grid[2], block[0], block[1], block[2], 0, stream, ptrs, None)
    if err != 0:
        raise RuntimeError(f"cuLaunchKernel failed with CUDA error {err}")


def _c_arg(a):
    if isinstance(a, torch.Tensor):
        return ctypes.c_void_p(a.data_ptr())
    if a is None:
        return ctypes.c_void_p(None)
    if isinstance(a, float):
        return ctypes.c_float(a)
    return ctypes.c_int(a)


def launch(kernel, grid: tuple[int, int, int], block: tuple[int, int, int], args: list, stream: int) -> None:
    """Lean replacement for torch's _CudaKernel.__call__ (which costs ~50 us of Python per launch).
    args: CUDA tensors (pointers), None (null pointer), Python ints (int32) and floats (float32).
    stream: raw stream handle, see stream_handle()."""
    vals = [_c_arg(a) for a in args]
    ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.addressof(v) for v in vals])
    _launch_raw(kernel.func, grid, block, stream, ptrs)


class PreparedLaunch:
    """A kernel launch whose argument block is built once. Only pointer/int values change per call,
    which avoids most per-launch Python cost (matters for token-by-token generation)."""

    def __init__(self, kernel, grid, block, kinds: str):
        # kinds: one char per argument, 'p' = pointer, 'i' = int32, 'f' = float32
        self.func = kernel.func
        self.grid, self.block = grid, block
        self.vals = [ctypes.c_void_p() if k == "p" else ctypes.c_int() if k == "i" else ctypes.c_float() for k in kinds]
        self.ptrs = (ctypes.c_void_p * len(self.vals))(*[ctypes.addressof(v) for v in self.vals])

    def __call__(self, stream: int, *args) -> None:
        for v, a in zip(self.vals, args):
            v.value = a
        _launch_raw(self.func, self.grid, self.block, stream, self.ptrs)


def get_kernel(name: str, device: torch.device | None = None):
    dev = torch.cuda.current_device() if device is None else torch.device(device).index or 0
    key = (dev, name)
    k = _cache.get(key)
    if k is None:
        with _lock:
            k = _cache.get(key)
            if k is None:
                _ensure_cuda_home()
                with torch.cuda.device(dev):
                    major, minor = torch.cuda.get_device_capability(dev)
                    k = torch.cuda._compile_kernel(_SRC, name, compute_capability=f"{major}{minor}",
                                                   nvcc_options=["--use_fast_math", "-std=c++17"])
                _cache[key] = k
    return k
