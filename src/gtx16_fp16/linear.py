"""nn.Linear replacement that stores fp16 weights and computes with the HFMA2 kernels."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernels import SKINNY_MAX_M, PreparedLaunch, get_kernel, stream_handle
from .ops import _sm_count, is_affected, mm_t, skinny_config, supported


class FastLinear(nn.Module):
    """Inference-only linear layer for TU116/TU117 GPUs.

    - Weight stored once as fp16, transposed to [in, out] (half the memory of fp32).
    - Small batches (<= 8 rows, e.g. token-by-token generation) compute in fp32 from fp32 inputs.
    - Larger batches cast inputs to fp16 (rows with |max| > 1024 are first scaled down by a power
      of two, then scaled back in fp32), multiply with packed fp16 FMA, accumulate in fp32.
    - Output dtype follows the input dtype (fp32 in -> fp32 out).
    - When autograd is needed or the shape is unsupported, falls back to F.linear.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self._skinny: dict = {}  # (M, device index) -> (PreparedLaunch, kchunk): token-by-token fast path
        self.register_buffer("weight_t", torch.empty(in_features, out_features, dtype=torch.float16, device=device))
        self.register_buffer("bias", torch.empty(out_features, dtype=torch.float32, device=device) if bias else None)

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "FastLinear":
        m = cls(lin.in_features, lin.out_features, lin.bias is not None, device=lin.weight.device)
        with torch.no_grad():
            m.weight_t.copy_(lin.weight.detach().t().to(torch.float16))
            if lin.bias is not None:
                m.bias.copy_(lin.bias.detach().float())
        return m

    @property
    def weight(self) -> torch.Tensor:  # some model code reads .weight (e.g. for dtype/device)
        return self.weight_t.t()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_dtype = x.dtype if x.dtype in (torch.float32, torch.float16) else torch.float32
        if torch.is_grad_enabled() and x.requires_grad:
            return F.linear(x, self.weight_t.t().to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))
        shape = x.shape
        # Fast path for small batches (generation): minimal Python per call.
        if x.is_cuda and x.dtype == torch.float32 and x.is_contiguous():
            K, N = self.in_features, self.out_features
            M = x.numel() // K if shape[-1] == K else -1
            if 0 < M <= SKINNY_MAX_M:
                dev = x.device
                key = (M, dev.index)
                entry = self._skinny.get(key)
                if entry is None:
                    grid, kchunk = skinny_config(N, K, _sm_count(dev))
                    entry = self._skinny[key] = (
                        PreparedLaunch(get_kernel("skinny_f32in", dev), grid, (256, 1, 1), "ppppiiii"), kchunk)
                pl, kchunk = entry
                out = torch.zeros(*shape[:-1], N, device=dev, dtype=torch.float32)
                pl(stream_handle(dev.index or 0), x.data_ptr(), self.weight_t.data_ptr(), out.data_ptr(),
                   None if self.bias is None else self.bias.data_ptr(), M, N, K, kchunk)
                return out
        x2 = x.reshape(-1, self.in_features)
        if x2.dtype not in (torch.float16, torch.float32):
            x2 = x2.float()
        x2 = x2.contiguous()  # fp32 stays fp32: the kernels cast (and rescale large rows) themselves
        if x2.shape[0] == 0 or not supported(x2, self.weight_t):
            y = F.linear(x2.float(), self.weight_t.t().float(), self.bias)
        else:
            y = mm_t(x2, self.weight_t, self.bias, torch.float32 if out_dtype == torch.float32 else torch.float16)
        return y.reshape(*shape[:-1], self.out_features).to(out_dtype)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


def patch(model: nn.Module, min_features: int = 256, force: bool = False,
          skip: tuple[str, ...] = ()) -> int:
    """Replace nn.Linear layers in-place with FastLinear. Returns the number replaced.

    Only runs on affected GPUs (GTX 16xx / TU116 / TU117) unless force=True.
    Layers with in/out features below min_features, or whose qualified name contains any string
    in `skip`, are left alone (e.g. skip=("lm_head",) to keep an output head in fp32).
    """
    if not force and not any(is_affected(p.device) for p in model.parameters() if p.is_cuda):
        return 0
    n = 0
    for name, mod in list(model.named_modules()):
        for child_name, child in list(mod.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if (isinstance(child, nn.Linear) and child.weight.is_cuda
                    and min(child.in_features, child.out_features) >= min_features
                    and child.in_features % 8 == 0 and child.out_features % 8 == 0
                    and not any(s in full for s in skip)):
                setattr(mod, child_name, FastLinear.from_linear(child))
                n += 1
    return n
