"""Inkling-2 NVFP4 and MXFP8 activation quantizers on XPU.

Bit exact with the PyTorch formulation and with an independent scalar codebook,
over zero rows, ties and a wide dynamic range.
"""

import pytest
import torch

from sglang.kernels.ops.moe.inkling2_experts_xpu import quantize_mx, quantize_nv
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kernels import inkling2_xpu as ref

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")


@pytest.mark.parametrize(
    "quant,scalar,width",
    [(quantize_nv, ref.scalar_nv, 16), (quantize_mx, ref.scalar_mx, 32)],
)
def test_quantizers_match_scalar_codebook(quant, scalar, width):
    x = torch.stack(
        [
            torch.zeros(width * 2),
            torch.linspace(-3.0, 5.0, width * 2),
            torch.tensor([0.0, 0.5, -1.0, 2.0] * ((width * 2) // 4)),
        ]
    ).to(torch.bfloat16)
    actual, expected = quant(x.xpu()), scalar(x)
    if isinstance(actual, tuple):
        torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=0, atol=0)
        actual, expected = actual[0], expected[0]
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [1152, 2304])
@pytest.mark.parametrize("rows", [1, 8, 67, 2048])
def test_triton_quantizers_bit_exact(width, rows):
    torch.manual_seed(width + rows)
    x = torch.randn(rows, width) * torch.logspace(-3, 3, rows)[:, None]
    x[0, : width // 2] = 0.0
    if rows > 2:
        x[1] = 0.0
        x[2, ::7] = torch.tensor([0.25, -0.75, 1.25, -1.75, 2.5, -3.5, 5.0]).repeat(
            width
        )[: x[2, ::7].numel()]
    x = x.bfloat16().xpu()
    q_ref, a_ref = ref.nv_reference(x)
    q, a = quantize_nv(x)
    torch.testing.assert_close(a.cpu(), a_ref.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(q.cpu(), q_ref.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(
        quantize_mx(x).cpu(), ref.mx_reference(x).cpu(), rtol=0, atol=0
    )
