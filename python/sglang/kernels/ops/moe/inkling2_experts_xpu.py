"""Inkling-2 NVFP4 and MXFP8 activation fake quantization on Intel XPU.

The routed experts quantize their activations before the Xe20 grouped GEMMs, as
the checkpoint's formats define: NVFP4 (E2M1 values, E4M3 scale per 16, FP32 row
amax) and MXFP8 (E4M3 values, E8M0 scale per 32). Outputs are bit exact with the
PyTorch formulation.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _fp4_round(x):
    """Nearest E2M1 magnitude; ties at 0.25, 1.25, 2.5 and 5.0 round down."""
    a = tl.abs(x)
    v = tl.where(
        a <= 0.25,
        0.0,
        tl.where(
            a < 0.75,
            0.5,
            tl.where(
                a <= 1.25,
                1.0,
                tl.where(
                    a < 1.75,
                    1.5,
                    tl.where(
                        a <= 2.5,
                        2.0,
                        tl.where(a < 3.5, 3.0, tl.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return tl.where(x < 0, -v, v)


@triton.jit
def _e4m3_rne(x):
    """Round non-negative fp32 x <= 448 to the nearest-even float8_e4m3fn value."""
    subnormal = libdevice.rint(x * 512.0) * 0.001953125
    bits = x.to(tl.int32, bitcast=True)
    bits = (bits + 0x7FFFF + ((bits >> 20) & 1)) & -1048576
    return tl.where(x < 0.015625, subnormal, bits.to(tl.float32, bitcast=True))


# Both quantizers match the PyTorch reference formulation bit for bit, which
# computes tensor / scalar as a multiply by the fp32 reciprocal of the scalar and
# scalar / tensor as reciprocal(tensor) * scalar; the kernels use the same forms.
@triton.jit
def _nv_candidate(z, group_max, enc, dec, amax_scale, den: tl.constexpr):
    s = _e4m3_rne(tl.minimum(group_max * (enc * (1.0 / den)), 448.0))
    inv = tl.minimum(tl.div_rn(1.0, s * dec), 3.4028234663852886e38)
    v = _fp4_round(z * inv[:, None]) * s[:, None]
    return v, tl.sum(tl.abs(z - v * amax_scale), 1)


@triton.jit
def _nv_quant_kernel(X, Q, A, GROUPS: tl.constexpr, GROUPS_POW2: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, GROUPS_POW2)
    mask = (group < GROUPS)[:, None]
    offsets = row * GROUPS * 16 + group[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
    group_max = tl.max(tl.abs(z), 1)
    a = tl.max(group_max, 0)
    enc = tl.minimum(tl.div_rn(1.0, a) * 1536.0, 3.4028234663852886e38)
    dec = tl.div_rn(1.0, enc)
    amax_scale = a * (1.0 / 1536.0)
    v6, err6 = _nv_candidate(z, group_max, enc, dec, amax_scale, 6.0)
    v4, err4 = _nv_candidate(z, group_max, enc, dec, amax_scale, 4.0)
    tl.store(
        Q + offsets, tl.where((err4 < err6)[:, None], v4, v6).to(tl.bfloat16), mask
    )
    tl.store(A + row, a)


@triton.jit
def _mx_quant_kernel(X, Q, BLOCKS: tl.constexpr, BLOCKS_POW2: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, BLOCKS_POW2)
    mask = (block < BLOCKS)[:, None]
    offsets = row * BLOCKS * 32 + block[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets, mask, other=0.0).to(tl.float32)
    raw = tl.max(tl.abs(x), 1) * (1.0 / 448.0)
    p = ((raw.to(tl.int32, bitcast=True) + 0x007FFFFF) & 0x7F800000) >> 23
    p = tl.where(raw <= 5.877471754111438e-39, 0, tl.minimum(p, 254))
    # The 2**-127 scale of an all-tiny block is subnormal; use its exact inverse.
    scale = (p << 23).to(tl.float32, bitcast=True)
    inverse = ((254 - p) << 23).to(tl.float32, bitcast=True)
    tiny = (p == 0)[:, None]
    scaled = tl.where(tiny, x * 1.7014118346046923e38, x * inverse[:, None])
    scaled = tl.minimum(tl.maximum(scaled, -448.0), 448.0)
    magnitude = _e4m3_rne(tl.abs(scaled))
    value = tl.where(scaled < 0, -magnitude, magnitude)
    out = tl.where(tiny, value * 5.877471754111438e-39, value * scale[:, None])
    tl.store(Q + offsets, out.to(tl.bfloat16), mask)


def quantize_nv(x):
    """NVFP4 activation fake quantization of [rows, width] (width % 16 == 0).

    Returns the dequantized BF16 values and the FP32 per-row amax. Each K16 group
    takes the E4M3 scale of its 6- or 4-level candidate with the lower error.
    """
    rows, width = x.shape
    assert width % 16 == 0 and x.is_contiguous()
    q = torch.empty((rows, width), device=x.device, dtype=torch.bfloat16)
    amax = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        _nv_quant_kernel[(rows,)](
            x,
            q,
            amax,
            width // 16,
            triton.next_power_of_2(width // 16),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, amax


def quantize_mx(x):
    """MXFP8 activation fake quantization of [rows, width] (width % 32 == 0):
    E4M3 values with one UE8M0 scale per K32 block, dequantized to BF16."""
    rows, width = x.shape
    assert width % 32 == 0 and x.is_contiguous()
    q = torch.empty((rows, width), device=x.device, dtype=torch.bfloat16)
    if rows:
        _mx_quant_kernel[(rows,)](
            x,
            q,
            width // 32,
            triton.next_power_of_2(width // 32),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q
