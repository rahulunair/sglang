"""Inkling-2 loop message controllers and residual add on XPU against references.

Decode (1-9 tokens) and prefill tile configurations (37-2048 tokens), one and
three controllers; BF16 rounding of the hidden projection bounds rel-L2 at 2e-3.
"""

import pytest
import torch

from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers
from sglang.kernels.ops.elementwise.inkling2_residual_add import residual_add
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kernels import inkling2_xpu as ref

register_xpu_ci(est_time=20, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")

DEV = "xpu"
W = 2304


def rel_l2(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("tokens", [1, 9])
def test_residual_add_matches_reference(tokens):
    xs = [torch.randn(tokens, W, device=DEV).to(torch.bfloat16) for _ in range(3)]
    assert torch.equal(residual_add(*xs), ref.residual_add_reference(*xs))


@pytest.mark.parametrize("tokens", [1, 9, 37, 512, 2048])
@pytest.mark.parametrize("num", [1, 3])
def test_message_controllers_triton_on_xpu(tokens, num):
    torch.manual_seed(tokens + num)
    rank = 64
    dests = tuple(torch.randn(tokens, W, device=DEV) for _ in range(num))
    stacked = torch.randn(num, tokens, W, device=DEV)
    messages = tuple(stacked.unbind(0))
    down = (torch.randn(num, rank, 2 * W, device=DEV) * 0.02).to(torch.bfloat16)
    gain = (torch.randn(num, W, rank, device=DEV) * 0.1).to(torch.bfloat16)
    forget = (torch.randn(num, W, rank, device=DEV) * 0.1).to(torch.bfloat16)
    got = message_controllers(dests, messages, down, gain, forget)
    want = ref.message_controllers_reference(dests, messages, down, gain, forget)
    assert got.shape == want.shape and rel_l2(got, want) <= 2e-3
