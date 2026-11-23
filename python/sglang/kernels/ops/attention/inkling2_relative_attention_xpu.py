"""Paged relative-logit attention, ported from Inkling's online-softmax path."""

import torch
import triton
import triton.language as tl


# T affects output offsets only; alignment must not create another binary.
@triton.jit(do_not_specialize=["T"])
def _relative_paged(
    Q,
    K,
    V,
    Rel,
    Pages,
    Lengths,
    CuQ,
    O,
    L,
    SQ0: tl.constexpr,
    SQ1: tl.constexpr,
    SK0: tl.constexpr,
    SK1: tl.constexpr,
    SK2: tl.constexpr,
    SV0: tl.constexpr,
    SV1: tl.constexpr,
    SV2: tl.constexpr,
    SR0: tl.constexpr,
    SR1: tl.constexpr,
    SP,
    T,
    H: tl.constexpr,
    HK: tl.constexpr,
    D: tl.constexpr,
    E: tl.constexpr,
    PAGE: tl.constexpr,
    LEFT: tl.constexpr,
    SCALE: tl.constexpr,
    SPLITS: tl.constexpr,
    HAS_REL: tl.constexpr,
    CAUSAL: tl.constexpr,
    ANCHORED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
):
    qt = tl.program_id(0)
    head_split = tl.program_id(1)
    head, split = head_split // SPLITS, head_split % SPLITS
    req = tl.program_id(2)
    start = tl.load(CuQ + req)
    end = tl.load(CuQ + req + 1)
    qlen = end - start
    if qt * BM >= qlen:
        return
    length = tl.load(Lengths + req)
    row = qt * BM + tl.arange(0, BM)
    token = start + row
    pos = length - qlen + row
    dim = tl.arange(0, D)
    query = tl.load(
        Q + token[:, None] * SQ0 + head * SQ1 + dim[None, :],
        row[:, None] < qlen,
        other=0.0,
    )
    kvhead = head // (H // HK)
    maximum = tl.full((BM,), -float("inf"), tl.float32)
    denom = tl.zeros((BM,), tl.float32)
    result = tl.zeros((BM, D), tl.float32)
    low = 0
    anchor_start = 0
    if LEFT >= 0:
        if ANCHORED:
            anchor_start = tl.maximum(length - qlen - (LEFT + 1), 0)
            low = anchor_start // BN
        else:
            low = tl.maximum(length - qlen + qt * BM - LEFT, 0) // BN
    if CAUSAL:
        high = tl.cdiv(tl.minimum(length - qlen + (qt + 1) * BM, length), BN)
    else:
        high = tl.cdiv(length, BN)
    for block in range(low + split, high, SPLITS):
        keypos = block * BN + tl.arange(0, BN)
        validkeys = keypos < length
        page = tl.load(Pages + req * SP + keypos // PAGE, validkeys, other=0).to(
            tl.int64
        )
        keyptr = (
            K
            + page[:, None] * SK0
            + (keypos % PAGE)[:, None] * SK1
            + kvhead * SK2
            + dim[None, :]
        )
        valptr = (
            V
            + page[:, None] * SV0
            + (keypos % PAGE)[:, None] * SV1
            + kvhead * SV2
            + dim[None, :]
        )
        keys = tl.load(keyptr, validkeys[:, None], other=0.0)
        values = tl.load(valptr, validkeys[:, None], other=0.0)
        score = tl.dot(query, tl.trans(keys)) * SCALE
        distance = pos[:, None] - keypos[None, :]
        visible = (row[:, None] < qlen) & validkeys[None, :]
        if CAUSAL:
            visible = visible & (distance >= 0)
        if LEFT >= 0:
            if ANCHORED:
                visible = visible & (keypos[None, :] >= anchor_start)
            else:
                visible = visible & (distance <= LEFT)
        if HAS_REL:
            bias = tl.load(
                Rel + token[:, None] * SR0 + head * SR1 + distance,
                visible & (distance >= 0) & (distance < E),
                other=0.0,
            ).to(tl.float32)
            score = score + bias
        score = tl.where(visible, score, -float("inf"))
        newmax = tl.maximum(maximum, tl.max(score, 1))
        alpha = tl.where(maximum > -float("inf"), tl.exp(maximum - newmax), 0.0)
        probs = tl.where(visible, tl.exp(score - newmax[:, None]), 0.0)
        denom = denom * alpha + tl.sum(probs, 1)
        result = result * alpha[:, None] + tl.dot(probs.to(values.dtype), values)
        maximum = newmax
    normal = tl.where(denom > 0.0, denom, 1.0)
    result = result / normal[:, None]
    lse = tl.where(denom > 0.0, maximum + tl.log(normal), -float("inf"))
    outbase = split * T * H * D + token[:, None] * H * D + head * D + dim[None, :]
    tl.store(O + outbase, result, row[:, None] < qlen)
    tl.store(L + split * T * H + token * H + head, lse, row < qlen)


@triton.jit
def _merge_splits(
    O,
    L,
    Out,
    OutL,
    TH,
    SPLITS: tl.constexpr,
    SPLITS_POW2: tl.constexpr,
    D: tl.constexpr,
):
    """Out[t, h] = sum_s exp(L[s] - lse) O[s] with lse = logsumexp_s L[s]."""
    row = tl.program_id(0).to(tl.int64)
    split = tl.arange(0, SPLITS_POW2)
    lse = tl.load(L + split * TH + row, split < SPLITS, other=-float("inf"))
    maximum = tl.max(lse, 0)
    stable = tl.where(maximum > -float("inf"), maximum, 0.0)
    weight = tl.where(lse > -float("inf"), tl.exp(lse - stable), 0.0)
    denom = tl.sum(weight, 0)
    dim = tl.arange(0, D)
    parts = tl.load(
        O + (split[:, None] * TH + row) * D + dim[None, :],
        (split < SPLITS)[:, None],
        other=0.0,
    )
    merged = tl.sum(parts * weight[:, None], 0) / tl.maximum(
        denom, 1.1754943508222875e-38
    )
    tl.store(Out + row * D + dim, merged)
    tl.store(
        OutL + row,
        tl.where(maximum > -float("inf"), stable + tl.log(denom), -float("inf")),
    )


def relative_paged_attention(
    q,
    key_cache,
    value_cache,
    rel_bias,
    page_table,
    cache_lengths,
    cu_q,
    *,
    max_query_length,
    scale,
    window_left=-1,
    splits=1,
    causal=True,
    window_anchor_at_start=False,
):
    """Returns (out, lse): out (tokens, heads, dim) and lse (tokens, heads), FP32."""
    if q.device.type != "xpu" or q.dtype != torch.bfloat16:
        raise ValueError("Inkling XPU relative attention requires BF16 XPU queries")
    if key_cache.dtype != q.dtype or value_cache.dtype != q.dtype:
        raise NotImplementedError(
            "Native Inkling KV formats need a separate validated reader"
        )
    t, heads, dim = q.shape
    pages, page_size, kvheads, kd = key_cache.shape
    if dim != kd or value_cache.shape != key_cache.shape or heads % kvheads:
        raise ValueError("Incompatible GQA/cache shape")
    if window_anchor_at_start and (causal or window_left < 0):
        raise ValueError("An anchored window requires noncausal local attention")
    if not causal and window_left >= 0 and not window_anchor_at_start:
        raise NotImplementedError(
            "Bidirectional local attention requires an explicit anchor"
        )
    if rel_bias is not None and (
        rel_bias.shape[:2] != (t, heads) or not rel_bias.is_contiguous()
    ):
        rel_bias = rel_bias.contiguous()
        if rel_bias.shape[:2] != (t, heads):
            raise ValueError("Relative logits must be [tokens, heads, extent]")
    splits = max(1, int(splits))
    out = torch.empty((splits, t, heads, dim), device=q.device, dtype=torch.float32)
    lse = torch.empty((splits, t, heads), device=q.device, dtype=torch.float32)
    common = (
        q.stride(0),
        q.stride(1),
        *key_cache.stride()[:3],
        *value_cache.stride()[:3],
        *(rel_bias.stride()[:2] if rel_bias is not None else (0, 0)),
        page_table.stride(0),
    )
    extent = rel_bias.shape[2] if rel_bias is not None else 0
    if t:
        _relative_paged[
            (triton.cdiv(max_query_length, 16), heads * splits, cu_q.numel() - 1)
        ](
            q,
            key_cache,
            value_cache,
            rel_bias,
            page_table,
            cache_lengths,
            cu_q,
            out,
            lse,
            *common,
            t,
            heads,
            kvheads,
            dim,
            extent,
            page_size,
            window_left,
            scale,
            splits,
            rel_bias is not None,
            causal,
            window_anchor_at_start,
            16,
            64,
            num_warps=4,
            enable_fp_fusion=False,
        )
    if splits == 1:
        return out[0], lse[0]
    merged = torch.empty((t, heads, dim), device=q.device, dtype=torch.float32)
    merged_lse = torch.empty((t, heads), device=q.device, dtype=torch.float32)
    if t:
        _merge_splits[(t * heads,)](
            out,
            lse,
            merged,
            merged_lse,
            t * heads,
            splits,
            triton.next_power_of_2(splits),
            dim,
            num_warps=4,
        )
    return merged, merged_lse
