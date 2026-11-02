import pytest
import torch
import torch.nn as nn

import gtx16_fp16 as g

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def rel_err(x, ref):
    return ((x.float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()


SHAPES = [  # (M, K, N): tile-aligned, ragged edges, skinny (M <= 8), tiny
    (128, 128, 128), (256, 512, 384), (1000, 1032, 776), (129, 136, 8), (3, 4096, 4096),
    (1, 1024, 3072), (8, 2048, 520), (9, 64, 64), (4096, 4096, 1024), (17, 8, 4104),
    (64, 1024, 1024), (33, 4096, 768), (200, 1032, 136),  # split-K path
]


@pytest.mark.parametrize("M,K,N", SHAPES)
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_mm_t_matches_fp32(M, K, N, out_dtype, with_bias):
    torch.manual_seed(M * 7 + K * 3 + N)
    x = (torch.randn(M, K, device="cuda") * 0.5).half()
    w = (torch.randn(K, N, device="cuda") * 0.5).half()
    b = torch.randn(N, device="cuda") if with_bias else None
    ref = x.float() @ w.float() + (b if with_bias else 0)
    assert g.supported(x, w)
    y = g.mm_t(x, w, b, out_dtype)
    assert y.dtype == out_dtype and y.shape == (M, N)
    assert torch.isfinite(y).all()
    # fp16-output rounding alone is ~3e-4 relative; hybrid accumulation adds a little more
    assert rel_err(y, ref) < (2e-3 if out_dtype == torch.float16 else 1.5e-3)


@pytest.mark.parametrize("M", [1, 2, 8, 64, 1000])
def test_bias_and_inputs_are_not_modified(M):
    # regression: the skinny path once accumulated into the caller's bias tensor (expand().contiguous()
    # returned the bias itself for M == 1)
    x = torch.randn(M, 512, device="cuda").half()
    w = torch.randn(512, 768, device="cuda").half()
    b = torch.randn(768, device="cuda")
    x0, w0, b0 = x.clone(), w.clone(), b.clone()
    for _ in range(3):
        g.mm_t(x, w, b, torch.float32)
    torch.cuda.synchronize()
    assert torch.equal(x, x0) and torch.equal(w, w0) and torch.equal(b, b0)


def test_fast_linear_is_deterministic_across_calls():
    lin = nn.Linear(1024, 768).cuda()
    fl = g.FastLinear.from_linear(lin)
    x = torch.randn(1, 1024, device="cuda")
    with torch.no_grad():
        outs = [fl(x) for _ in range(4)]
    for o in outs[1:]:
        assert rel_err(o, outs[0]) < 1e-6


def test_matmul_fallback_for_odd_shapes():
    a = torch.randn(33, 13, device="cuda").half()  # K % 8 != 0 -> torch fallback
    b = torch.randn(13, 7, device="cuda").half()
    assert not g.supported(a, b)
    torch.testing.assert_close(g.matmul(a, b), (a @ b), rtol=1e-2, atol=1e-2)


def test_fast_linear_matches_nn_linear():
    torch.manual_seed(0)
    lin = nn.Linear(1024, 768).cuda()
    fl = g.FastLinear.from_linear(lin)
    for shape in [(4, 50, 1024), (1, 1, 1024), (2, 1024), (7, 1024)]:
        x = torch.randn(*shape, device="cuda")
        with torch.no_grad():
            ref, y = lin(x), fl(x)
        assert y.dtype == torch.float32 and y.shape == ref.shape
        assert rel_err(y, ref) < 3e-3


def test_fast_linear_falls_back_when_grad_needed():
    lin = nn.Linear(256, 256).cuda()
    fl = g.FastLinear.from_linear(lin)
    x = torch.randn(4, 256, device="cuda", requires_grad=True)
    fl(x).sum().backward()
    assert x.grad is not None


def test_patch_replaces_linear_layers():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(512, 1024), nn.GELU(), nn.Linear(1024, 512), nn.Linear(512, 10)).cuda().eval()
    x = torch.randn(64, 512, device="cuda")
    with torch.no_grad():
        ref = model(x)
    n = g.patch(model, min_features=256, force=True)
    assert n == 2  # the 512->10 head is below min_features
    assert isinstance(model[0], g.FastLinear) and isinstance(model[3], nn.Linear)
    with torch.no_grad():
        y = model(x)
    assert rel_err(y, ref) < 3e-3


def test_large_outputs_do_not_overflow_with_fp32_out():
    # products sum to ~1e6, far above fp16 max (65504); fp32 output must stay finite and exact-ish
    x = torch.full((128, 4096), 16.0, device="cuda").half()
    w = torch.full((4096, 128), 16.0, device="cuda").half()
    y = g.mm_t(x, w, None, torch.float32)
    assert torch.isfinite(y).all()
    assert abs(y[0, 0].item() - 16 * 16 * 4096) / (16 * 16 * 4096) < 1e-3


@pytest.mark.parametrize("M", [1, 5, 64, 300])
@pytest.mark.parametrize("in_dtype", [torch.float32, torch.float16])
def test_massive_activations_do_not_overflow(M, in_dtype):
    # LLM MLP down_proj inputs can exceed fp16 max (seen: 179,940 in Qwen3-TTS). Rows must be rescaled.
    torch.manual_seed(1)
    K, N = 3072, 1024
    x = torch.randn(M, K, device="cuda")
    x[:, 7] = 179940.0 if in_dtype == torch.float32 else 60000.0
    x[0, 100] = -120000.0 if in_dtype == torch.float32 else -50000.0
    x = x.to(in_dtype)
    w = (torch.randn(K, N, device="cuda") * 0.3).half()
    ref = x.float() @ w.float()
    y = g.mm_t(x, w, None, torch.float32)
    assert torch.isfinite(y).all()
    assert rel_err(y, ref) < 2e-3


def test_is_affected_reports_bool():
    assert isinstance(g.is_affected(), bool)
