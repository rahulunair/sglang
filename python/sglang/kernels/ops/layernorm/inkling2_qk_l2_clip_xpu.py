"""Inkling-2 per-head QK L2 clip on Intel XPU (Triton).

The CUDA implementation is the CUTE DSL kernel in inkling2_qk_l2_clip.py; this is
the XPU path with the same contract.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_l2_clip_kernel(
    X,
    Scale,
    OutScale,
    Out,
    x_stride,
    heads,
    total,
    rows_per_scale,
    threshold,
    eps,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    SCALE_MODE: tl.constexpr,
    HAS_OUT_SCALE: tl.constexpr,
    ROWS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    offsets = tl.arange(0, BLOCK)
    mask = (row < total)[:, None] & (offsets[None, :] < WIDTH)
    token, head = row // heads, row % heads
    x = (
        X
        + token.to(tl.int64)[:, None] * x_stride
        + head[:, None] * WIDTH
        + offsets[None, :]
    )
    value = tl.load(x, mask, other=0.0).to(tl.float32)
    if SCALE_MODE == 2:
        value = value * tl.load(
            Scale + head[:, None] * WIDTH + offsets[None, :], mask, other=0.0
        ).to(tl.float32)
    elif SCALE_MODE == 1:
        value = value * tl.load(Scale + offsets[None, :], mask, other=0.0).to(
            tl.float32
        )
    norm = tl.sqrt(tl.sum(value * value, axis=1))
    value = value * tl.minimum(threshold / (norm + eps), 1.0)[:, None]
    value = value.to(Out.dtype.element_ty)
    if HAS_OUT_SCALE:
        factor = tl.load(OutScale + row // rows_per_scale, row < total, other=0.0).to(
            tl.float32
        )
        value = (value.to(tl.float32) * factor[:, None]).to(Out.dtype.element_ty)
    out = Out + row.to(tl.int64)[:, None] * WIDTH + offsets[None, :]
    tl.store(out, value, mask)


def qk_l2_clip(
    x, threshold, eps=1e-6, *, output_dtype=None, scale=None, output_scale=None
):
    """x is (tokens, heads, width) with contiguous heads; returns a contiguous clip."""
    assert x.ndim == 3 and x.stride(2) == 1 and x.stride(1) == x.shape[2]
    tokens, heads, width = x.shape
    out = torch.empty(x.shape, dtype=output_dtype or x.dtype, device=x.device)
    total = tokens * heads
    if not total:
        return out
    if scale is not None:
        assert scale.is_contiguous() and scale.shape[-1] == width
        if scale.ndim == 2:
            # The reference cycles 2-D scales over flat rows; heads must match.
            assert scale.shape[0] == heads
    rows_per_scale = 1
    if output_scale is not None:
        assert total % output_scale.numel() == 0
        rows_per_scale = total // output_scale.numel()
    block_rows = 8
    _qk_l2_clip_kernel[(triton.cdiv(total, block_rows),)](
        x,
        scale,
        output_scale,
        out,
        x.stride(0),
        heads,
        total,
        rows_per_scale,
        threshold,
        eps,
        width,
        triton.next_power_of_2(width),
        0 if scale is None else scale.ndim,
        output_scale is not None,
        block_rows,
        num_warps=4,
    )
    return out
