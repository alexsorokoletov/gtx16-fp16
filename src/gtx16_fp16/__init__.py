"""Fast fp16 matmul for GTX 16-series (TU116/TU117) GPUs, which have fast fp16 units but no tensor cores.

    import gtx16_fp16
    n = gtx16_fp16.patch(model)          # swap nn.Linear -> FastLinear (only on affected GPUs)
    y = gtx16_fp16.matmul(a16, b16)      # fp16 matmul without cuBLAS's emulated tensor-core path
"""
from .linear import FastLinear, patch
from .ops import is_affected, matmul, mm_t, supported

__all__ = ["FastLinear", "patch", "is_affected", "matmul", "mm_t", "supported"]
__version__ = "0.1.0"
