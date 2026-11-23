"""Inkling-2 RMS norm and QK L2 clip XPU kernels against their PyTorch references.

FP32 statistics; BF16 outputs may differ by one BF16 rounding (rel-L2 <= 2e-3).
Shapes are the served ones: width 2304, 24 query / 8 KV heads of 128.
"""

import pytest
import torch

from sglang.kernels.ops.layernorm.inkling2_qk_l2_clip_xpu import qk_l2_clip
from sglang.kernels.ops.layernorm.inkling2_rms_norm_xpu import rms_norm
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kernels import inkling2_xpu as ref

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")

DEV = "xpu"
W = 2304


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("tokens", [1, 9, 37])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_rms_norm(tokens, dtype):
    x = (torch.randn(tokens, W, device=DEV) * 3).to(dtype)
    weight = torch.rand(W, device=DEV) + 0.5
    got = rms_norm(x, weight, 1e-6, output_dtype=torch.bfloat16)
    want = ref.rms_norm_reference(x, weight, 1e-6, output_dtype=torch.bfloat16)
    assert got.dtype == torch.bfloat16 and rel_l2(got, want) <= 2e-3
    out = torch.empty(3, tokens, W, device=DEV, dtype=torch.bfloat16)
    rms_norm(x, None, 1e-6, out=out[1])
    assert (
        rel_l2(
            out[1], ref.rms_norm_reference(x, None, 1e-6, output_dtype=torch.bfloat16)
        )
        <= 2e-3
    )


@pytest.mark.parametrize("tokens", [1, 9, 37])
@pytest.mark.parametrize(
    "heads,width,scale_mode,tau",
    [
        (24, 128, 2, True),
        (8, 128, 0, False),
        (24, 16, 2, True),
        (8, 128, 1, False),
    ],
)
def test_qk_l2_clip(tokens, heads, width, scale_mode, tau):
    torch.manual_seed(heads + width + tokens)
    projected = (torch.randn(tokens, heads * width + 77, device=DEV) * 4).to(
        torch.bfloat16
    )
    x = projected[:, 5 : 5 + heads * width].reshape(tokens, heads, width)
    scale = {
        0: None,
        1: torch.rand(width, device=DEV) + 0.5,
        2: torch.rand(heads, width, device=DEV) + 0.5,
    }[scale_mode]
    out_scale = torch.rand(tokens, device=DEV) + 0.5 if tau else None
    got = qk_l2_clip(x, 22.6, 1e-6, scale=scale, output_scale=out_scale)
    want = ref.qk_l2_clip_reference(x, 22.6, 1e-6, scale=scale, output_scale=out_scale)
    assert got.shape == want.shape and rel_l2(got, want) <= 2e-3
