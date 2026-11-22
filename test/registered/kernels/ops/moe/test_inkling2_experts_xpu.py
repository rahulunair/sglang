"""Inkling-2 routed experts on XPU (sgl-kernel-xpu Xe20 grouped GEMMs).

The NVFP4 and MXFP8 activation quantizers are bit-exact with the PyTorch
formulation and with an independent scalar codebook. The full expert layer
matches an FP32 CPU reference route by route (rel-L2), EP rank partials sum to the full
layer, and graph replay follows new routes.
"""

import types

import pytest
import torch

from sglang.kernels.ops.moe import inkling2_experts_xpu
from sglang.kernels.ops.moe.inkling2_experts_xpu import (
    check_native_experts,
    experts_forward,
    quantize_mx,
    quantize_nv,
    sort_routes,
)
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.kernels import inkling2_xpu as ref

register_xpu_ci(est_time=25, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")


def fixture(fp4, experts=8, hidden=256, inter=128, seed=71):
    torch.manual_seed(seed)
    m = types.SimpleNamespace(
        tp_size=1, ep_size=1, ep_rank=0, gate_cap=4.0, up_cap=25.0
    )
    for name, (n, k) in (("w13", (2 * inter, hidden)), ("w2", (hidden, inter))):
        if fp4:
            w = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8)
            s = (torch.randint(1, 8, (experts, n, k // 16)).float() / 4).to(
                torch.float8_e4m3fn
            )
            setattr(m, name + "_amax", torch.full((experts,), 8.0))
            setattr(m, name + "_outer_scale_denominator", torch.tensor([8.0]))
        else:
            w = (torch.randn(experts, n, k) * 0.25).to(torch.float8_e4m3fn)
            s = torch.randint(118, 124, (experts, n, k // 32), dtype=torch.uint8)
        setattr(m, name + "_weight", w)
        setattr(m, name + "_scale", s)
    return m


def to_device(m, rank=None, ep=1):
    d = types.SimpleNamespace(**vars(m))
    for key, value in vars(m).items():
        if isinstance(value, torch.Tensor):
            if rank is not None and key.endswith(("_weight", "_scale", "_amax")):
                per = value.shape[0] // ep
                value = value[rank * per : (rank + 1) * per].contiguous()
            setattr(d, key, value.to("xpu"))
    if rank is not None:
        d.ep_size, d.ep_rank = ep, rank
    return d


def rel_l2(actual, expected):
    a, e = actual.float(), expected.float()
    return ((a - e).norm() / e.norm().clamp_min(1e-12)).item()


def inputs(tokens, experts, hidden, top=4, seed=5):
    torch.manual_seed(seed)
    x = (torch.randn(tokens, hidden) * 0.5).bfloat16()
    ids = torch.stack([torch.randperm(experts)[:top] for _ in range(tokens)]).int()
    weights = torch.rand(tokens, top) * 0.5
    return x, weights, ids


def supported(fp4):
    try:
        check_native_experts(fixture(fp4, experts=8, hidden=32, inter=32))
    except RuntimeError:
        return False
    return True


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("tokens", [1, 8, 37])
def test_experts_match_cpu_reference(fp4, tokens):
    if not supported(fp4):
        pytest.skip("installed sgl-kernel-xpu lacks this format")
    m = fixture(fp4)
    x, w, ids = inputs(tokens, 8, 256)
    expected = ref.experts_reference(m, x, w, ids, fp4)
    actual = experts_forward(x.xpu(), w.xpu(), ids.xpu(), to_device(m)).cpu()
    assert actual.dtype == torch.bfloat16
    # NVFP4 re-quantizes the W2 input, where one-ULP accumulation differences flip
    # FP4 codes relative to the scalar reference (measured 0.75-2.85% here).
    assert rel_l2(actual, expected) <= (0.03 if fp4 else 2e-3)


@pytest.mark.parametrize("fp4", [False, True])
def test_expert_parallel_partials_sum_to_full_layer(fp4):
    if not supported(fp4):
        pytest.skip("installed sgl-kernel-xpu lacks this format")
    m = fixture(fp4, experts=16)
    x, w, ids = inputs(19, 16, 256)
    full = experts_forward(
        x.xpu(), w.xpu(), ids.xpu(), to_device(m), output_dtype=torch.float32
    )
    parts = [
        experts_forward(
            x.xpu(), w.xpu(), ids.xpu(), to_device(m, r, 2), output_dtype=torch.float32
        )
        for r in (0, 1)
    ]
    assert rel_l2(parts[0] + parts[1], full) <= 2e-3


@pytest.mark.parametrize("fp4", [False, True])
def test_native_graph_replay_tracks_new_routes(fp4):
    if not supported(fp4):
        pytest.skip("installed sgl-kernel-xpu lacks this format")
    m = to_device(fixture(fp4))
    x, w, ids = (t.xpu() for t in inputs(8, 8, 256))
    run = lambda: experts_forward(x, w, ids, m)
    run()
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        captured = run()
    for seed in (1, 2, 3):
        nx, nw, nids = inputs(8, 8, 256, seed=seed)
        x.copy_(nx.xpu())
        w.copy_(nw.xpu())
        ids.copy_(nids.xpu())
        graph.replay()
        torch.xpu.synchronize()
        torch.testing.assert_close(captured.cpu(), run().cpu(), rtol=0, atol=0)


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


@pytest.mark.parametrize("tokens,top", [(1, 8), (9, 8), (72, 8), (37, 4)])
@pytest.mark.parametrize("rank", [0, 1])
def test_route_sort_matches_native_prepare(tokens, top, rank):
    torch.manual_seed(tokens * 10 + rank)
    experts, local = 128, 64
    ids = (
        torch.stack([torch.randperm(experts)[:top] for _ in range(tokens)]).int().xpu()
    )
    weights = torch.rand(tokens, top).xpu()
    small = sort_routes(ids, weights, rank * local, local, 2304)
    native = sort_routes(ids, weights, rank * local, local, 2304, force_native=True)
    # Same expert blocks and the same per-route metadata; the order of rows inside
    # an expert may differ, which the per-row GEMMs and the combine do not see.
    perm_small, perm_native = small[0].long(), native[0].long()
    assert torch.equal(small[1], native[1])
    assert torch.equal(small[3], native[3])
    for column in (2, 3, 4):
        assert torch.equal(small[column][perm_small], native[column][perm_native])
    assert torch.equal(
        torch.sort(perm_small)[0], torch.arange(perm_small.numel(), device="xpu")
    )


@pytest.mark.parametrize("fp4", [False, True])
@pytest.mark.parametrize("ep_rank", [None, 1])
def test_fused_route_path_is_bitwise_identical(fp4, ep_rank, monkeypatch):
    if not supported(fp4):
        pytest.skip("installed sgl-kernel-xpu lacks this format")
    m = fixture(fp4, experts=16)
    x, w, ids = (t.xpu() for t in inputs(9, 16, 256))
    d = to_device(m) if ep_rank is None else to_device(m, ep_rank, 2)
    fused = experts_forward(x, w, ids, d, output_dtype=torch.float32)
    monkeypatch.setattr(inkling2_experts_xpu, "SORT_MAX_ROUTES", 0)
    native = experts_forward(x, w, ids, d, output_dtype=torch.float32)
    assert torch.equal(fused, native)
