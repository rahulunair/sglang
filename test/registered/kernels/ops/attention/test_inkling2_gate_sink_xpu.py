"""Inkling-2 gate-sink epilogue on Intel XPU against an FP32 reference.

On XPU the kernel replaces the PTX sigmoid with tl.exp2, so the CUDA tests
do not cover its math. Gate and sinks are strided slices of one projection,
as the model passes them.
"""

import pytest
import torch

from sglang.kernels.ops.attention.inkling2_attn import gate_sink
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")


@pytest.mark.parametrize("tokens", [1, 37])
@pytest.mark.parametrize("head_groups", [None, (3, 4)])
def test_gate_sink_matches_fp32_reference(tokens, head_groups):
    """Padded head groups read only the real heads of each group."""
    torch.manual_seed(tokens)
    heads, dim = 12, 128
    group, padded_group = head_groups or (heads, heads)
    stored = heads // group * padded_group
    output = torch.randn(tokens, stored * dim, device="xpu")
    lse = torch.randn(tokens, stored, device="xpu") * 4
    projection = torch.randn(tokens, heads * dim + heads + 5, device="xpu") * 3
    projection = projection.bfloat16()
    gate = projection[:, : heads * dim]
    sinks = projection[:, heads * dim + 3 : heads * dim + 3 + heads]

    got = gate_sink(
        output, gate, lse, sinks, output_dtype=torch.bfloat16, head_groups=head_groups
    )

    real = torch.arange(heads, device="xpu")
    real = real + real // group * (padded_group - group)
    x = output.view(tokens, stored, dim)[:, real]
    want = (
        x
        * torch.sigmoid(gate.float().view(tokens, heads, dim))
        * torch.sigmoid(lse[:, real] - sinks.float()).unsqueeze(-1)
    )
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(
        got.float().view(tokens, heads, dim), want, atol=2e-2, rtol=2e-2
    )
