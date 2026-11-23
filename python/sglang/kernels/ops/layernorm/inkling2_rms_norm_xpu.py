"""Inkling-2 RMS norm on Intel XPU (Triton).

The CUDA implementation is the CUTE DSL kernel in inkling2_rms_norm.py; this is
the XPU path with the same contract (FP32 statistics, optional pre/post weights).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_kernel(
    X,
    PreWeight,
    Weight,
    Out,
    x_stride,
    out_stride,
    eps,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PRE: tl.constexpr,
    HAS_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < WIDTH
    value = tl.load(X + row * x_stride + offsets, mask, other=0.0).to(tl.float32)
    if HAS_PRE:
        value = value * tl.load(PreWeight + offsets, mask, other=0.0).to(tl.float32)
    value = value * tl.rsqrt(tl.sum(value * value) / WIDTH + eps)
    if HAS_WEIGHT:
        value = value * tl.load(Weight + offsets, mask, other=0.0).to(tl.float32)
    tl.store(Out + row * out_stride + offsets, value, mask)


def rms_norm(x, weight, eps, *, pre_weight=None, output_dtype=None, out=None):
    width = x.shape[-1]
    rows = x.reshape(-1, width)
    if rows.stride(-1) != 1:
        rows = rows.contiguous()
    result = (
        out
        if out is not None
        else torch.empty(x.shape, dtype=output_dtype or x.dtype, device=x.device)
    )
    flat = result.view(-1, width)
    if rows.shape[0]:
        _rms_norm_kernel[(rows.shape[0],)](
            rows,
            pre_weight,
            weight,
            flat,
            rows.stride(0),
            flat.stride(0),
            eps,
            width,
            triton.next_power_of_2(width),
            pre_weight is not None,
            weight is not None,
            num_warps=8,
        )
    return result
