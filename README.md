# gtx16-fp16

Fast fp16 matrix multiply for **GTX 16-series GPUs** (TU116 / TU117: GTX 1650, 1660, 1660 Ti, 1660 Super, MX450, T400/T600/T1000).

On these cards, PyTorch fp16 matmul is **9× slower than fp32**. This package uses the chip's fp16 units directly and makes fp16 matmul **~13× faster than PyTorch fp16 and ~1.4–1.9× faster than fp32**.

## The problem

TU116 and TU117 are Turing chips (compute capability 7.5) **without tensor cores**. NVIDIA replaced the tensor cores with dedicated fp16 units that do packed fp16 FMA (`HFMA2`) at 2× the fp32 rate.

cuBLAS treats every sm_75 device as having tensor cores and uses `HMMA` (tensor-core) kernels for fp16 GEMM. On TU116/TU117 the `HMMA` instruction is emulated, about 11× slower than `HFMA2`. Measured on a GTX 1660 Ti (`bench/peak.py`):

| Instruction | Throughput |
|---|---|
| fp32 FMA (`FFMA`) | 4.6 TFLOPS |
| packed fp16 FMA (`HFMA2`) | **9.2 TFLOPS** |
| tensor-core MMA (`HMMA`, `mma.sync.m16n8k8`) | **0.84 TFLOPS** |

Result: `torch.matmul` on fp16 tensors runs at 0.5 TFLOPS, against 4.5 TFLOPS for fp32 (`bench/repro_cublas.py`). Newer CUDA does not help (CUDA 13.0 build: same 0.5 TFLOPS fp16). The driver is not the cause.

## The fix

Kernels that use only `HFMA2`, compiled at runtime with NVRTC through PyTorch (no nvcc, no extra packages):

- **GEMM** (128×128 tiles): products accumulate in fp16 over each 16-wide K slice, then fold into fp32. Error stays close to fp32 accumulation.
- **Split-K GEMM** for shapes with too few output tiles to fill the GPU.
- **Skinny GEMM** for M ≤ 8 (token-by-token generation): fp32 accumulation, reads fp16 weights (half the bytes of fp32).

GTX 1660 Ti, PyTorch 2.14.1 + CUDA 12.6 (`bench/bench_gemm.py`), TFLOPS:

| M × K × N | cuBLAS fp32 | cuBLAS fp16 | **gtx16-fp16** | vs fp32 | rel. error (cuBLAS fp16 / ours) |
|---|---|---|---|---|---|
| 1×4096×4096 | 0.13 | 0.18 | **0.25** | 1.9× | 2.1e-4 / 3.2e-7 |
| 4×4096×4096 | 0.53 | 0.03 | **0.95** | 1.8× | 2.9e-4 / 2.6e-7 |
| 1×1024×3072 | 0.13 | 0.10 | **0.14** | 1.15× | 2.1e-4 / 2.2e-7 |
| 64×1024×1024 | 2.32 | 0.43 | **1.50** | 0.65× | 3.0e-4 / 6.1e-4 |
| 1024×1024×1024 | 2.76 | 0.45 | **5.23** | 1.9× | 2.1e-4 / 6.0e-4 |
| 4096×4096×4096 | 4.54 | 0.50 | **6.41** | 1.4× | 2.1e-4 / 6.0e-4 |

### Real models (`bench/e2e_*.py`)

| Model | Workload | fp32 | `patch(model)` | Result quality |
|---|---|---|---|---|
| bge-base-en-v1.5 (BERT, 110M) | 256 texts × 256 tokens | 51 texts/s | **104 texts/s (2.0×)** | cosine vs fp32 ≥ 0.999999 |
| Qwen3-TTS 0.6B | token-by-token speech generation | 0.65× realtime | 0.63× realtime (same) | valid audio; VRAM 4.96 → **3.84 GB** |

Batched, GPU-bound inference gets the full speedup. Token-by-token generation of a small model is limited by Python overhead in the model code, not by matmul, so speed stays the same while memory drops by the fp32 → fp16 weight savings.

Qwen3-TTS also showed why overflow protection matters. One MLP layer (`code_predictor.layers.2.mlp.down_proj`) receives inputs up to **179,940**, far above fp16's 65,504. The GEMM path scales such rows down by a power of two before the fp16 cast and scales the result back in fp32. The small-batch path computes in fp32 from fp32 inputs and cannot overflow.

## Use

```bash
pip install git+https://github.com/alexsorokoletov/gtx16-fp16
```

```python
import gtx16_fp16

# Inference: replace nn.Linear layers with FastLinear (fp16 weights, fp32 output).
# Does nothing on GPUs that are not affected, unless force=True.
n = gtx16_fp16.patch(model)

# Or call the matmul directly.
y = gtx16_fp16.matmul(a_fp16, b_fp16)                       # fp16 out, like torch
y = gtx16_fp16.mm_t(x_fp16, w_t_fp16, bias_fp32, torch.float32)  # x @ w_t + bias, fp32 out
```

`FastLinear` stores weights in fp16 (half the memory of fp32) and returns the input's dtype, so an fp32 model keeps fp32 activations between layers.

## Limits

- **Inference only.** When autograd needs a gradient, `FastLinear` falls back to `F.linear`.
- **Shapes:** K and N must be multiples of 8, with 16-byte aligned contiguous tensors. Other shapes fall back to PyTorch.
- **fp16 range:** on the GEMM path (more than 8 rows), rows with |max| > 1024 are rescaled automatically, which covers inputs above 65504. Weights are stored in fp16, so a weight above 65504 (very rare) would overflow; partial sums of 16 products can overflow only if |input| × |weight| products stay near 4096 across a whole slice after rescaling. Use `patch(..., skip=(...))` to keep a layer in fp32.
- **Token-by-token generation** gains memory, not speed, unless the model is large enough that its matmuls dominate (see the Qwen3-TTS row above). CUDA graphs would remove the Python overhead; not done yet.
- **Medium-M shapes** (about 16–128 rows with small K and N) are slower than cuBLAS fp32, though still faster than cuBLAS fp16.
- **Not bit-identical** to cuBLAS. Split-K and skinny paths use fp32 atomics, so results can differ in the last bits between runs.
- Tested on a GTX 1660 Ti. Other TU116/TU117 cards should behave the same; reports welcome.

## Test

```bash
pip install -e .[test] && pytest tests
python bench/peak.py          # instruction throughput
python bench/repro_cublas.py  # the cuBLAS problem, PyTorch only
python bench/bench_gemm.py    # this package vs cuBLAS
```

## Upstream

The real fix belongs in cuBLAS: it should not pick emulated tensor-core kernels on chips without tensor cores. Drafts, not filed yet:

- [`upstream/nvidia-cublas-report.md`](upstream/nvidia-cublas-report.md): what we would report to NVIDIA. It covers instruction throughput (FFMA / HFMA2 / HMMA), cuBLAS vs an HFMA2-only GEMM, the repro scripts, and two questions: is there a supported way to keep cuBLAS off tensor-core kernels on these devices, and can its heuristics detect TU116/TU117?
- [`upstream/pytorch-issue.md`](upstream/pytorch-issue.md): the PyTorch-side issue, a 20-line repro and possible mitigations.

## License

MIT
