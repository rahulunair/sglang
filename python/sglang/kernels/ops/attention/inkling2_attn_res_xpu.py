"""Inkling-2 block attention residual on Intel XPU (Triton).

The CUDA implementation is the CUTE DSL kernel in inkling2_attn_res.py; this is the
XPU path with the same contract: a softmax over up to nine residual sources per
token for up to three queries, FP32 outputs.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_res_stats_kernel(
    Sources,
    Addend,
    Queries,
    Stats,
    NUM_SOURCES: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Stats[token, source] = (sum of squares, dot with each query)."""
    token = tl.program_id(0)
    source = tl.program_id(1)
    row = token.to(tl.int64) * WIDTH
    for s in tl.static_range(NUM_SOURCES):
        if s == source:
            sumsq = tl.zeros((BLOCK,), tl.float32)
            dot0 = tl.zeros((BLOCK,), tl.float32)
            dot1 = tl.zeros((BLOCK,), tl.float32)
            dot2 = tl.zeros((BLOCK,), tl.float32)
            for start in range(0, WIDTH, BLOCK):
                offsets = start + tl.arange(0, BLOCK)
                mask = offsets < WIDTH
                value = tl.load(Sources[s] + row + offsets, mask, other=0.0).to(
                    tl.float32
                )
                if HAS_ADDEND:
                    if s == NUM_SOURCES - 1:
                        value += tl.load(Addend + row + offsets, mask, other=0.0).to(
                            tl.float32
                        )
                sumsq += value * value
                dot0 += value * tl.load(Queries + offsets, mask, other=0.0)
                if NUM_QUERIES > 1:
                    dot1 += value * tl.load(Queries + WIDTH + offsets, mask, other=0.0)
                if NUM_QUERIES > 2:
                    dot2 += value * tl.load(
                        Queries + 2 * WIDTH + offsets, mask, other=0.0
                    )
            out = Stats + (token * NUM_SOURCES + s) * 4
            tl.store(out, tl.sum(sumsq))
            tl.store(out + 1, tl.sum(dot0))
            if NUM_QUERIES > 1:
                tl.store(out + 2, tl.sum(dot1))
            if NUM_QUERIES > 2:
                tl.store(out + 3, tl.sum(dot2))


@triton.jit
def _source_softmax(logits, valid):
    logits = tl.where(valid, logits, -float("inf"))
    weights = tl.where(valid, tl.exp(logits - tl.max(logits)), 0.0)
    return weights / tl.sum(weights)


@triton.jit
def _attn_res_mix_kernel(
    Sources,
    Addend,
    Stats,
    Out,
    tokens,
    NUM_SOURCES: tl.constexpr,
    SOURCES_POW2: tl.constexpr,
    NUM_QUERIES: tl.constexpr,
    HAS_ADDEND: tl.constexpr,
    NORMALIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Out[q, token, chunk] = sum over sources of softmax(logits)[q, s] * source."""
    token = tl.program_id(0)
    row = token.to(tl.int64) * WIDTH
    source = tl.arange(0, SOURCES_POW2)
    valid = source < NUM_SOURCES
    stats = Stats + (token * NUM_SOURCES + source) * 4
    rstd = tl.rsqrt(tl.load(stats, valid, other=0.0) / WIDTH + 1e-6)
    p0 = _source_softmax(tl.load(stats + 1, valid, other=0.0) * rstd, valid)
    p1 = p0
    p2 = p0
    if NUM_QUERIES > 1:
        p1 = _source_softmax(tl.load(stats + 2, valid, other=0.0) * rstd, valid)
    if NUM_QUERIES > 2:
        p2 = _source_softmax(tl.load(stats + 3, valid, other=0.0) * rstd, valid)
    if NORMALIZE:
        p0 = p0 * rstd
        p1 = p1 * rstd
        p2 = p2 * rstd
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < WIDTH
    acc0 = tl.zeros((BLOCK,), tl.float32)
    acc1 = tl.zeros((BLOCK,), tl.float32)
    acc2 = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(NUM_SOURCES):
        value = tl.load(Sources[s] + row + offsets, mask, other=0.0).to(tl.float32)
        if HAS_ADDEND:
            if s == NUM_SOURCES - 1:
                value += tl.load(Addend + row + offsets, mask, other=0.0).to(tl.float32)
        hit = source == s
        acc0 += tl.sum(tl.where(hit, p0, 0.0)) * value
        if NUM_QUERIES > 1:
            acc1 += tl.sum(tl.where(hit, p1, 0.0)) * value
        if NUM_QUERIES > 2:
            acc2 += tl.sum(tl.where(hit, p2, 0.0)) * value
    tl.store(Out + row + offsets, acc0, mask)
    if NUM_QUERIES > 1:
        tl.store(Out + tokens * WIDTH + row + offsets, acc1, mask)
    if NUM_QUERIES > 2:
        tl.store(Out + 2 * tokens * WIDTH + row + offsets, acc2, mask)


def attn_res(blocks, partial_addend, queries, *, normalize_values=False):
    """FP32 (queries, tokens, width) softmax mix over the sources of each token,
    returned unbound; the addend joins the last source in FP32."""
    assert blocks and queries.ndim == 2 and queries.dtype == torch.float32
    num_queries = queries.shape[0]
    assert num_queries <= 3 and queries.is_contiguous()
    tokens, width = blocks[0].shape
    assert all(b.shape == (tokens, width) and b.is_contiguous() for b in blocks)
    device = blocks[0].device
    out = torch.empty((num_queries, tokens, width), dtype=torch.float32, device=device)
    if tokens:
        sources = tuple(blocks)
        stats = torch.empty(
            (tokens, len(blocks), 4), dtype=torch.float32, device=device
        )
        _attn_res_stats_kernel[(tokens, len(blocks))](
            sources,
            partial_addend,
            queries,
            stats,
            len(blocks),
            num_queries,
            partial_addend is not None,
            width,
            256,
            num_warps=4,
        )
        block = 256
        _attn_res_mix_kernel[(tokens, triton.cdiv(width, block))](
            sources,
            partial_addend,
            stats,
            out,
            tokens,
            len(blocks),
            max(2, triton.next_power_of_2(len(blocks))),
            num_queries,
            partial_addend is not None,
            normalize_values,
            width,
            block,
            num_warps=4,
        )
    return tuple(out.unbind(0))
