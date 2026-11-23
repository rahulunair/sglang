"""Inkling-2 route top-k on Intel XPU.

XPU selects the top-k experts by iterative argmax instead of the 64-bit key
sort, which Xe emulates. Both must pick the same experts in the same order
(value descending, ties to the lower id) and produce the same weights.
"""

import pytest
import torch
import triton

from sglang.kernels.ops.moe import inkling2_route
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")


def route(partials, bias, by_argmax):
    groups, m, n = partials.shape
    logits = torch.empty((m, n), device="xpu", dtype=torch.float32)
    ids = torch.empty((m, 8), device="xpu", dtype=torch.int32)
    weights = torch.empty((m, 8), device="xpu", dtype=torch.float32)
    scale = torch.ones(1, device="xpu")
    inkling2_route._route_topk[(m,)](
        partials,
        bias,
        logits,
        ids,
        weights,
        scale,
        2.5,
        m,
        n,
        groups,
        bias is not None,
        8,
        8,
        triton.next_power_of_2(n),
        True,
        by_argmax,
        enable_fp_fusion=False,
    )
    return ids, weights


@pytest.mark.parametrize("rows", [1, 9, 37])
@pytest.mark.parametrize("groups", [1, 2])
@pytest.mark.parametrize("ties,with_bias", [(False, True), (True, True), (True, False)])
def test_route_topk_argmax_matches_key_sort(rows, groups, ties, with_bias):
    torch.manual_seed(rows * 7 + groups)
    partials = torch.randn(groups, rows, 128, device="xpu")
    if ties:
        partials = (partials * 4).round() / 4
    bias = (torch.randn(128, device="xpu") * 0.1) if with_bias else None
    if ties and with_bias:
        bias = (bias * 8).round() / 8
    want_ids, want_w = route(partials, bias, False)
    got_ids, got_w = route(partials, bias, True)
    assert torch.equal(got_ids, want_ids)
    assert torch.equal(got_w, want_w)
