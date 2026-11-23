"""Inkling-2 block attention residual XPU kernel against its PyTorch reference.

FP32 outputs match to accumulation order (rel-L2 <= 1e-5): up to nine sources,
three queries, with and without the partial addend and value normalization.
"""

import pytest
import torch

from sglang.kernels.ops.attention.inkling2_attn_res_xpu import attn_res
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
@pytest.mark.parametrize(
    "num_sources,queries,addend,normalize",
    [
        (1, 3, False, False),
        (5, 1, True, False),
        (9, 3, True, False),
        (8, 3, False, True),
    ],
)
def test_attn_res(tokens, num_sources, queries, addend, normalize):
    torch.manual_seed(tokens * 10 + num_sources)
    blocks = [
        torch.randn(tokens, W, device=DEV).to(torch.bfloat16) * (1 + i)
        for i in range(num_sources)
    ]
    extra = torch.randn(tokens, W, device=DEV).to(torch.bfloat16) if addend else None
    q = torch.randn(queries, W, device=DEV) * W**-0.5
    got = attn_res(blocks, extra, q, normalize_values=normalize)
    want = ref.attn_res_reference(blocks, extra, q, normalize_values=normalize)
    assert len(got) == len(want) == queries
    for g, w in zip(got, want):
        assert g.dtype == torch.float32 and rel_l2(g, w) <= 1e-5
