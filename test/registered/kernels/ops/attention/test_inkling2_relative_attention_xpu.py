"""Inkling-2 paged relative-bias attention on XPU against a PyTorch reference.

Split-K decode/verify, packed GQA, native Xe20 prefill (with LSE), sliding
windows, anchored bidirectional draft blocks, and binary reuse across token
counts.
"""

import pytest
import torch

from sglang.kernels.ops.attention.inkling2_relative_attention_xpu import (
    relative_paged_attention,
)
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=30, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")


@pytest.mark.parametrize("window", [-1, 3])
@pytest.mark.parametrize("splits", [1, 4])
def test_paged_relative_attention_prefix_window_and_empty_splits(window, splits):
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(913)
    page, hkv, dim, hq, extent = 16, 2, 32, 6, 32
    lengths = [143, 70]
    queries = [3, 1]
    q = torch.randn(sum(queries), hq, dim).to(torch.bfloat16)
    key = torch.randn(18, page, hkv, dim).to(torch.bfloat16)
    value = torch.randn_like(key)
    table = torch.randperm(18).reshape(2, 9).to(torch.int32)
    rel = torch.randn(sum(queries), hq, extent).to(torch.bfloat16) * 0.1
    expected = torch.empty(q.shape, dtype=torch.float32)
    expected_lse = torch.empty(q.shape[:2], dtype=torch.float32)
    start = 0
    for req, (length, qlen) in enumerate(zip(lengths, queries)):
        packed_key = key[table[req].long()].reshape(-1, hkv, dim)[:length].float()
        packed_value = value[table[req].long()].reshape(-1, hkv, dim)[:length].float()
        for qi in range(qlen):
            pos = length - qlen + qi
            distance = pos - torch.arange(length)
            visible = (distance >= 0) & ((distance <= window) if window >= 0 else True)
            for h in range(hq):
                scores = packed_key[:, h // 3] @ q[start + qi, h].float() / (dim**0.5)
                within = (distance >= 0) & (distance < extent)
                scores[within] += rel[start + qi, h, distance[within]].float()
                scores[~visible] = -torch.inf
                expected_lse[start + qi, h] = torch.logsumexp(scores, 0)
                expected[start + qi, h] = scores.softmax(0) @ packed_value[:, h // 3]
        start += qlen
    out, lse = relative_paged_attention(
        q.to("xpu"),
        key.to("xpu"),
        value.to("xpu"),
        rel.to("xpu"),
        table.to("xpu"),
        torch.tensor(lengths, device="xpu", dtype=torch.int32),
        torch.tensor([0, 3, 4], device="xpu", dtype=torch.int32),
        max_query_length=3,
        scale=dim**-0.5,
        window_left=window,
        splits=splits,
    )
    torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-5, rtol=2e-5)


def test_checkpoint_shape_multitile_prefill_and_relative_extent_boundary():
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(417)
    torch.set_num_threads(4)
    page, length, qlen, hq, hkv, dim, extent = 64, 193, 19, 24, 8, 128, 128
    q = torch.randn(qlen, hq, dim).bfloat16()
    k = torch.randn(4, page, hkv, dim).bfloat16()
    v = torch.randn_like(k)
    rel = (torch.randn(qlen, hq, extent) * 0.3).bfloat16()
    table = torch.tensor([[2, 0, 3, 1]], dtype=torch.int32)
    kc = (
        k[table[0].long()]
        .reshape(-1, hkv, dim)[:length]
        .repeat_interleave(3, dim=1)
        .float()
    )
    vc = (
        v[table[0].long()]
        .reshape(-1, hkv, dim)[:length]
        .repeat_interleave(3, dim=1)
        .float()
    )
    scores = torch.einsum("qhd,khd->qhk", q.float(), kc) * (dim**-0.5)
    distance = (length - qlen + torch.arange(qlen))[:, None] - torch.arange(length)[
        None
    ]
    bias = torch.gather(
        rel.float(), -1, distance.clamp(0, extent - 1)[:, None].expand(-1, hq, -1)
    )
    scores += torch.where(((distance >= 0) & (distance < extent))[:, None], bias, 0)
    scores.masked_fill_((distance < 0)[:, None], -torch.inf)
    expected_lse = scores.logsumexp(-1)
    expected = torch.einsum("qhk,khd->qhd", scores.softmax(-1), vc)
    out, lse = relative_paged_attention(
        q.to("xpu"),
        k.to("xpu"),
        v.to("xpu"),
        rel.to("xpu"),
        table.to("xpu"),
        torch.tensor([length], dtype=torch.int32, device="xpu"),
        torch.tensor([0, qlen], dtype=torch.int32, device="xpu"),
        max_query_length=qlen,
        scale=dim**-0.5,
    )
    torch.testing.assert_close(out.cpu(), expected, atol=2e-3, rtol=1e-2)
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("length", [17, 143])
@pytest.mark.parametrize("window", [-1, 31])
def test_mtp_bidirectional_block_anchor_keeps_all_draft_keys(length, window):
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(421)
    torch.set_num_threads(4)
    qlen, hq, hkv, dim, page = 8, 24, 8, 128, 16
    pages = (length + page - 1) // page
    q = torch.randn(qlen, hq, dim).bfloat16()
    k = torch.randn(pages, page, hkv, dim).bfloat16()
    v = torch.randn_like(k)
    table = torch.randperm(pages).to(torch.int32)[None]
    kc = (
        k[table[0].long()]
        .reshape(-1, hkv, dim)[:length]
        .repeat_interleave(3, dim=1)
        .float()
    )
    vc = (
        v[table[0].long()]
        .reshape(-1, hkv, dim)[:length]
        .repeat_interleave(3, dim=1)
        .float()
    )
    scores = torch.einsum("qhd,khd->qhk", q.float(), kc) * (dim**-0.5)
    # All queries share the same context boundary, and see every draft key.
    anchor = max(length - qlen - (window + 1), 0) if window >= 0 else 0
    scores[:, :, :anchor] = -torch.inf
    gold_lse = scores.logsumexp(-1)
    gold = torch.einsum("qhk,khd->qhd", scores.softmax(-1), vc)
    output, lse = relative_paged_attention(
        q.to("xpu"),
        k.to("xpu"),
        v.to("xpu"),
        None,
        table.to("xpu"),
        torch.tensor([length], device="xpu", dtype=torch.int32),
        torch.tensor([0, qlen], device="xpu", dtype=torch.int32),
        max_query_length=qlen,
        scale=dim**-0.5,
        window_left=window,
        causal=False,
        window_anchor_at_start=window >= 0,
    )
    torch.testing.assert_close(output.cpu(), gold, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(lse.cpu(), gold_lse, rtol=2e-5, atol=2e-5)


# Causal attention runs the packed kernel; counts within one M bucket share a binary.
@pytest.mark.parametrize(
    "kernel,heads,counts",
    [
        ("_relative_paged_packed", 4, (17, 40)),
        ("_relative_paged_packed", 4, (17, 24)),
        ("_relative_paged_packed", 2, (5, 8)),
    ],
)
def test_variable_token_counts_reuse_the_same_attention_binary(
    monkeypatch, kernel, heads, counts
):
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    from sglang.kernels.ops.attention import (
        inkling2_relative_attention_xpu as attention,
    )

    target = getattr(attention, kernel)
    compiled = []
    original = target.run

    def record(*args, **kwargs):
        compiled_kernel = original(*args, **kwargs)
        compiled.append(compiled_kernel.hash)
        return compiled_kernel

    monkeypatch.setattr(target, "run", record)
    torch.manual_seed(755)
    dim, length = 128, 48
    q = (torch.randn(max(counts), heads, dim) * 0.1).bfloat16()
    k = (torch.randn(3, 16, 1, dim) * 0.1).bfloat16()
    v = (torch.randn_like(k.float()) * 0.1).bfloat16()
    kd, vd = k.to("xpu"), v.to("xpu")
    table = torch.arange(3, dtype=torch.int32, device="xpu")[None]
    lens = torch.tensor([length], dtype=torch.int32, device="xpu")
    for count in counts:
        query = q[:count].contiguous()
        rel = torch.zeros(count, heads, 32, dtype=torch.bfloat16)
        out, lse = relative_paged_attention(
            query.to("xpu"),
            kd,
            vd,
            rel.to("xpu"),
            table,
            lens,
            torch.tensor([0, count], dtype=torch.int32, device="xpu"),
            max_query_length=count,
            scale=dim**-0.5,
        )
        keys = k[:, :, 0].reshape(-1, dim).double()
        values = v[:, :, 0].reshape(-1, dim).double()
        scores = torch.einsum("qhd,kd->qhk", query.double(), keys) * dim**-0.5
        visible = (
            torch.arange(length)[None, :]
            <= (length - count + torch.arange(count))[:, None]
        )
        scores.masked_fill_(~visible[:, None, :], -torch.inf)
        expected = torch.einsum("qhk,kd->qhd", scores.softmax(-1), values).float()
        torch.testing.assert_close(out.cpu(), expected, rtol=1e-2, atol=2e-3)
        torch.testing.assert_close(
            lse.cpu(), scores.logsumexp(-1).float(), rtol=2e-5, atol=2e-5
        )
    assert len(compiled) == 2 and compiled[0] == compiled[1], (
        "Token count must not force a new device binary"
    )


@pytest.mark.parametrize("max_cache_length", [None, 73])
@pytest.mark.parametrize("window", [-1, 31])
def test_three_request_heterogeneous_prefill(window, max_cache_length):
    """Cover the serving failure's 29+73+73 token layout at TP2 head sizes."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(175)
    torch.set_num_threads(4)
    queries, lengths = [29, 73, 73], [29, 73, 73]
    page, hq, hkv, dim, extent = 64, 12, 4, 128, 128
    q = torch.randn(sum(queries), hq, dim).bfloat16()
    k = torch.randn(6, page, hkv, dim).bfloat16()
    v = torch.randn_like(k)
    rel = (torch.randn(sum(queries), hq, extent) * 0.1).bfloat16()
    table = torch.randperm(6).reshape(3, 2).int()
    expected, expected_lse = [], []
    start = 0
    for i, (qlen, length) in enumerate(zip(queries, lengths)):
        kc = (
            k[table[i].long()]
            .reshape(-1, hkv, dim)[:length]
            .repeat_interleave(3, 1)
            .float()
        )
        vc = (
            v[table[i].long()]
            .reshape(-1, hkv, dim)[:length]
            .repeat_interleave(3, 1)
            .float()
        )
        scores = (
            torch.einsum("qhd,khd->qhk", q[start : start + qlen].float(), kc)
            * dim**-0.5
        )
        distance = (length - qlen + torch.arange(qlen))[:, None] - torch.arange(length)[
            None
        ]
        bias = torch.gather(
            rel[start : start + qlen].float(),
            -1,
            distance.clamp(0, extent - 1)[:, None].expand(-1, hq, -1),
        )
        scores += torch.where(((distance >= 0) & (distance < extent))[:, None], bias, 0)
        visible = distance >= 0
        if window >= 0:
            visible &= distance <= window
        scores.masked_fill_(~visible[:, None], -torch.inf)
        expected.append(torch.einsum("qhk,khd->qhd", scores.softmax(-1), vc))
        expected_lse.append(scores.logsumexp(-1))
        start += qlen
    out, lse = relative_paged_attention(
        q.to("xpu"),
        k.to("xpu"),
        v.to("xpu"),
        rel.to("xpu"),
        table.to("xpu"),
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        torch.tensor([0, 29, 102, 175], dtype=torch.int32, device="xpu"),
        max_query_length=73,
        scale=dim**-0.5,
        window_left=window,
        max_cache_length=max_cache_length,
    )
    torch.testing.assert_close(
        out.float().cpu(), torch.cat(expected), atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(lse.cpu(), torch.cat(expected_lse), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("window", [-1, 1023])
@pytest.mark.parametrize("splits", [1, 8, 16])
def test_packed_verify_checkpoint_heads_long_context_splits(window, splits):
    """Nine verify rows of 24/8 heads over 3,000 cached tokens, packed GQA tile."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(1291)
    qlen, length, hq, hkv, dim, extent, page = 9, 3000, 24, 8, 128, 128, 64
    pages = (length + page - 1) // page
    q = torch.randn(qlen, hq, dim, device="xpu").bfloat16()
    k = torch.randn(pages, page, hkv, dim, device="xpu").bfloat16()
    v = torch.randn_like(k)
    rel = (torch.randn(qlen, hq, extent, device="xpu") * 0.3).bfloat16()
    table = torch.randperm(pages, device="xpu").to(torch.int32)[None]
    kc = (
        k[table[0].long()]
        .reshape(-1, hkv, dim)[:length]
        .repeat_interleave(3, dim=1)
        .float()
    )
    vc = (
        v[table[0].long()]
        .reshape(-1, hkv, dim)[:length]
        .repeat_interleave(3, dim=1)
        .float()
    )
    scores = torch.einsum("qhd,khd->qhk", q.float(), kc) * dim**-0.5
    distance = (length - qlen + torch.arange(qlen, device="xpu"))[
        :, None
    ] - torch.arange(length, device="xpu")[None]
    bias = torch.gather(
        rel.float(), -1, distance.clamp(0, extent - 1)[:, None].expand(-1, hq, -1)
    )
    scores += torch.where(((distance >= 0) & (distance < extent))[:, None], bias, 0)
    hidden = (distance < 0) | ((distance > window) if window >= 0 else False)
    scores.masked_fill_(hidden[:, None], -torch.inf)
    out, lse = relative_paged_attention(
        q,
        k,
        v,
        rel,
        table,
        torch.tensor([length], dtype=torch.int32, device="xpu"),
        torch.tensor([0, qlen], dtype=torch.int32, device="xpu"),
        max_query_length=qlen,
        scale=dim**-0.5,
        window_left=window,
        splits=splits,
    )
    torch.testing.assert_close(
        out, torch.einsum("qhk,khd->qhd", scores.softmax(-1), vc), atol=2e-3, rtol=1e-2
    )
    torch.testing.assert_close(lse, scores.logsumexp(-1), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("window", [-1, 1023])
@pytest.mark.parametrize(
    "lengths,queries", [([1000], [300]), ([1200, 333], [1200, 90])]
)
def test_prefill_native_relative_attention_with_lse(window, lengths, queries):
    """Causal prefill on the native relative-attention kernel, with LSE."""
    if not torch.xpu.is_available():
        pytest.skip("XPU required")
    torch.manual_seed(2207)
    hq, hkv, dim, extent, page = 24, 8, 128, 128, 64
    pages_per = [(n + page - 1) // page for n in lengths]
    width = max(pages_per)
    k = torch.randn(sum(pages_per), page, hkv, dim, device="xpu").bfloat16()
    v = torch.randn_like(k)
    table = torch.zeros(len(lengths), width, dtype=torch.int32, device="xpu")
    perm = torch.randperm(sum(pages_per), device="xpu").to(torch.int32)
    start = 0
    for i, n in enumerate(pages_per):
        table[i, :n] = perm[start : start + n]
        start += n
    q = torch.randn(sum(queries), hq, dim, device="xpu").bfloat16()
    rel = (torch.randn(sum(queries), hq, extent, device="xpu") * 0.3).bfloat16()
    cu = torch.tensor(
        [0, *torch.tensor(queries).cumsum(0).tolist()], dtype=torch.int32, device="xpu"
    )
    out, lse = relative_paged_attention(
        q,
        k,
        v,
        rel,
        table,
        torch.tensor(lengths, dtype=torch.int32, device="xpu"),
        cu,
        max_query_length=max(queries),
        scale=dim**-0.5,
        window_left=window,
        max_cache_length=max(lengths),
    )
    row = 0
    for i, (length, qlen) in enumerate(zip(lengths, queries)):
        kc = (
            k[table[i, : pages_per[i]].long()]
            .reshape(-1, hkv, dim)[:length]
            .repeat_interleave(3, 1)
            .float()
        )
        vc = (
            v[table[i, : pages_per[i]].long()]
            .reshape(-1, hkv, dim)[:length]
            .repeat_interleave(3, 1)
            .float()
        )
        qs, rs = q[row : row + qlen].float(), rel[row : row + qlen].float()
        scores = torch.einsum("qhd,khd->qhk", qs, kc) * dim**-0.5
        distance = (length - qlen + torch.arange(qlen, device="xpu"))[
            :, None
        ] - torch.arange(length, device="xpu")[None]
        bias = torch.gather(
            rs, -1, distance.clamp(0, extent - 1)[:, None].expand(-1, hq, -1)
        )
        scores += torch.where(((distance >= 0) & (distance < extent))[:, None], bias, 0)
        hidden = (distance < 0) | ((distance > window) if window >= 0 else False)
        scores.masked_fill_(hidden[:, None], -torch.inf)
        torch.testing.assert_close(
            out[row : row + qlen].float(),
            torch.einsum("qhk,khd->qhd", scores.softmax(-1), vc),
            atol=1e-2,
            rtol=2e-2,
        )
        torch.testing.assert_close(
            lse[row : row + qlen], scores.logsumexp(-1), atol=2e-5, rtol=2e-5
        )
        row += qlen
