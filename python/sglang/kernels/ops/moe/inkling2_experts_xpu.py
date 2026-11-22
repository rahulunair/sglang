"""Inkling packed experts on Intel XPU through sgl-kernel-xpu Xe20 grouped GEMMs.

The GEMMs are the native W4A16 (NVFP4) and W8A16 (MXFP8) grouped kernels, used as
for GPT-OSS. The Triton kernels here cover what is specific to Inkling: activation
fake quantization, route sorting metadata, NVFP4 restore with SiTU, and the
source-order combine.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _row_scale(XAmax, WAmax, WDenominator, expert, amax_row):
    denominator = 1536.0 * tl.load(WDenominator)
    return (tl.load(XAmax + amax_row) * tl.load(WAmax + expert)) * tl.div_rn(
        1.0, denominator
    )


@triton.jit
def _restore_situ_rows(
    Acc,
    Out,
    Token,
    Expert,
    Gamma,
    XAmax,
    WAmax,
    Den,
    num_local,
    F: tl.constexpr,
    FP4: tl.constexpr,
    GATE_CAP: tl.constexpr,
    UP_CAP: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One sorted row: optional NVFP4 restore (BF16), SiTU on interleaved
    gate/up, route gamma, BF16 store. Rows routed to the skip expert get 0."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < F
    expert = tl.load(Expert + row)
    gamma = tl.load(Gamma + row)
    g = tl.load(Acc + row * 2 * F + 2 * col, mask, other=0.0).to(tl.float32)
    u = tl.load(Acc + row * 2 * F + 2 * col + 1, mask, other=0.0).to(tl.float32)
    if FP4:
        scale = _row_scale(XAmax, WAmax, Den, expert, tl.load(Token + row))
        g = (g * scale).to(tl.bfloat16).to(tl.float32)
        u = (u * scale).to(tl.bfloat16).to(tl.float32)
    tg = 2.0 * tl.sigmoid(2.0 * g / GATE_CAP) - 1.0
    tu = 2.0 * tl.sigmoid(2.0 * u / UP_CAP) - 1.0
    y = (GATE_CAP * tg * tl.sigmoid(g)) * (UP_CAP * tu) * gamma
    # Skip-expert rows were never written by the GEMM; never read them as data.
    y = tl.where(expert < num_local, y, 0.0)
    tl.store(Out + row * F + col, y.to(tl.bfloat16), mask)


@triton.jit
def _combine_sorted(
    Down,
    Perm,
    Ids,
    HAmax,
    WAmax,
    Den,
    Y,
    start,
    num_local,
    TOP_K: tl.constexpr,
    N: tl.constexpr,
    FP4: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Source-order combine read through the sort permutation: each route's
    (optionally restored) row is rounded to BF16, then summed in FP32."""
    token = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    total = tl.zeros((BLOCK,), tl.float32)
    for k in tl.static_range(TOP_K):
        src = token * TOP_K + k
        expert = tl.load(Ids + src) - start
        if (expert >= 0) & (expert < num_local):
            dst = tl.load(Perm + src).to(tl.int64)
            x = tl.load(Down + dst * N + cols, mask, other=0).to(tl.float32)
            if FP4:
                x = (
                    (x * _row_scale(HAmax, WAmax, Den, expert, dst))
                    .to(tl.bfloat16)
                    .to(tl.float32)
                )
            total += x
    tl.store(Y + token * N + cols, total, mask)


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
def _store_rows(
    Q, Perm, row, inner, value, mask, WIDTH: tl.constexpr, TOP_K: tl.constexpr
):
    """Store one quantized row at row, or at each of its TOP_K sorted positions."""
    if TOP_K == 0:
        tl.store(Q + row * WIDTH + inner, value, mask)
    else:
        for k in tl.static_range(TOP_K):
            dst = tl.load(Perm + row * TOP_K + k).to(tl.int64)
            tl.store(Q + dst * WIDTH + inner, value, mask)


@triton.jit
def _nv_quant_kernel(
    X, Q, A, Perm, GROUPS: tl.constexpr, GROUPS_POW2: tl.constexpr, TOP_K: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    group = tl.arange(0, GROUPS_POW2)
    mask = (group < GROUPS)[:, None]
    inner = group[:, None] * 16 + tl.arange(0, 16)[None, :]
    z = tl.load(X + row * GROUPS * 16 + inner, mask, other=0.0).to(tl.float32)
    group_max = tl.max(tl.abs(z), 1)
    a = tl.max(group_max, 0)
    enc = tl.minimum(tl.div_rn(1.0, a) * 1536.0, 3.4028234663852886e38)
    dec = tl.div_rn(1.0, enc)
    amax_scale = a * (1.0 / 1536.0)
    v6, err6 = _nv_candidate(z, group_max, enc, dec, amax_scale, 6.0)
    v4, err4 = _nv_candidate(z, group_max, enc, dec, amax_scale, 4.0)
    value = tl.where((err4 < err6)[:, None], v4, v6).to(tl.bfloat16)
    _store_rows(Q, Perm, row, inner, value, mask, GROUPS * 16, TOP_K)
    tl.store(A + row, a)


@triton.jit
def _mx_quant_kernel(
    X, Q, Perm, BLOCKS: tl.constexpr, BLOCKS_POW2: tl.constexpr, TOP_K: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    block = tl.arange(0, BLOCKS_POW2)
    mask = (block < BLOCKS)[:, None]
    inner = block[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + row * BLOCKS * 32 + inner, mask, other=0.0).to(tl.float32)
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
    _store_rows(Q, Perm, row, inner, out.to(tl.bfloat16), mask, BLOCKS * 32, TOP_K)


def quantize_nv(x, perm=None, top_k=0):
    """NVFP4 activation fake quantization of [rows, width] (width % 16 == 0).

    Returns the dequantized BF16 values and the FP32 per-row amax. Each K16 group
    takes the E4M3 scale of its 6- or 4-level candidate with the lower error.
    With perm ([rows * top_k] destination rows), row r is written to rows
    perm[r * top_k + k] of a [rows * top_k, width] output instead.
    """
    rows, width = x.shape
    assert width % 16 == 0 and x.is_contiguous()
    out_rows = rows * top_k if perm is not None else rows
    q = torch.empty((out_rows, width), device=x.device, dtype=torch.bfloat16)
    amax = torch.empty(rows, device=x.device, dtype=torch.float32)
    if rows:
        _nv_quant_kernel[(rows,)](
            x,
            q,
            amax,
            perm if perm is not None else amax,
            width // 16,
            triton.next_power_of_2(width // 16),
            top_k if perm is not None else 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, amax


def quantize_mx(x, perm=None, top_k=0):
    """MXFP8 activation fake quantization of [rows, width] (width % 32 == 0):
    E4M3 values with one UE8M0 scale per K32 block, dequantized to BF16.
    perm and top_k scatter rows as in quantize_nv."""
    rows, width = x.shape
    assert width % 32 == 0 and x.is_contiguous()
    out_rows = rows * top_k if perm is not None else rows
    q = torch.empty((out_rows, width), device=x.device, dtype=torch.bfloat16)
    if rows:
        _mx_quant_kernel[(rows,)](
            x,
            q,
            perm if perm is not None else q,
            width // 32,
            triton.next_power_of_2(width // 32),
            top_k if perm is not None else 0,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q


def _xe20_gemm(out, act, weight, scale, counts, n_experts, fp4):
    if fp4:
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w4a16(
            out, act, weight, scale, None, None, counts, n_experts, False, 16
        )
    else:
        torch.ops.sgl_kernel.moe_grouped_mm_nt_xe20_w8a16(
            out, act, weight, scale, None, counts, n_experts
        )


def check_native_experts(m):
    """Raise unless the installed sgl-kernel-xpu serves this expert layer."""
    import sgl_kernel.moe as sk_moe

    if m.tp_size != 1:
        raise ValueError(
            "Inkling experts on XPU require expert parallelism (moe_tp_size 1)"
        )
    if m.w13_weight.dtype == torch.uint8 and not getattr(
        sk_moe, "W4A16_NVFP4_SUPPORTED", False
    ):
        raise RuntimeError(
            "Inkling NVFP4 experts need sgl-kernel-xpu with the NVFP4 W4A16 GEMM"
        )


@triton.jit
def _local_expert_ids(Ids, Out, rows, start, local, BLOCK: tl.constexpr):
    """Rank-local expert of each route; other ranks' routes go to skip expert local."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = r < rows
    expert = tl.load(Ids + r, mask, other=0).to(tl.int32) - start
    expert = tl.where((expert >= 0) & (expert < local), expert, local)
    tl.store(Out + r, expert, mask)


@triton.jit
def _sorted_route_meta(
    Perm,
    SortIds,
    Weights,
    TokenOf,
    ExpertOf,
    Gamma,
    rows,
    local,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Per sorted row: its source token, local expert, and gamma (0 for skip rows)."""
    r = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = r < rows
    dst = tl.load(Perm + r, mask, other=0)
    expert = tl.load(SortIds + r, mask, other=0)
    weight = tl.load(Weights + r, mask, other=0.0).to(tl.float32)
    tl.store(TokenOf + dst, r // TOP_K, mask)
    tl.store(ExpertOf + dst, expert, mask)
    tl.store(Gamma + dst, tl.where(expert < local, weight, 0.0), mask)


# Decode, verify and small prefill batches sort their routes in one program; above
# this the native prepare_moe_input path runs. Arbitrary bound on the in-register
# sort (8 requests x 9 verify tokens x top-8 is 576).
SORT_MAX_ROUTES = 1024


@triton.jit
def _route_sort(
    Ids,
    Weights,
    Perm,
    Counts,
    TokenOf,
    ExpertOf,
    Gamma,
    rows,
    start,
    local,
    TOP_K: tl.constexpr,
    BLOCK: tl.constexpr,
    EXPERTS_POW2: tl.constexpr,
):
    """Sort routes by rank-local expert (other ranks' routes last, as skip expert
    local), stable in route order. Writes each route's sorted row (Perm), the
    per-expert row counts, and each sorted row's token, expert and gamma."""
    r = tl.arange(0, BLOCK)
    valid = r < rows
    expert = tl.load(Ids + r, valid, other=0).to(tl.int32) - start
    expert = tl.where((expert >= 0) & (expert < local), expert, local)
    key = tl.sort(tl.where(valid, expert * BLOCK + r, 2147483647))
    ok = r < rows
    src = tl.where(ok, key % BLOCK, 0)
    sorted_expert = tl.where(ok, key // BLOCK, local)
    tl.store(Perm + src, r, ok)
    tl.store(TokenOf + r, src // TOP_K, ok)
    tl.store(ExpertOf + r, sorted_expert, ok)
    weight = tl.load(Weights + src, ok, other=0.0).to(tl.float32)
    tl.store(Gamma + r, tl.where(sorted_expert < local, weight, 0.0), ok)
    bins = tl.arange(0, EXPERTS_POW2)
    counts = tl.sum((sorted_expert[None, :] == bins[:, None]).to(tl.int32), 1)
    tl.store(Counts + bins, counts, bins < local)


def _sort_routes_native(flat, weights, rows, start, local, hidden, top, dev):
    """prepare_moe_input path for large route counts; same outputs as _route_sort."""
    import sgl_kernel.moe  # noqa: F401  registers the Xe20 MoE ops

    sort_ids = torch.empty(rows, dtype=torch.int32, device=dev)
    _local_expert_ids[(triton.cdiv(rows, 1024),)](
        flat, sort_ids, rows, start, local, 1024
    )
    counts = torch.empty(local + 1, dtype=torch.int32, device=dev)
    sizes1 = torch.empty((local + 1, 3), dtype=torch.int32, device=dev)
    sizes2 = torch.empty((local + 1, 3), dtype=torch.int32, device=dev)
    a_map = torch.empty(rows, dtype=torch.int32, device=dev)
    perm = torch.empty(rows, dtype=torch.int32, device=dev)
    torch.ops.sgl_kernel.prepare_moe_input.default(
        sort_ids.view(-1, top),
        counts,
        None,
        sizes1,
        sizes2,
        a_map,
        perm,
        local + 1,
        hidden,
        top,
    )
    token_of = torch.empty(rows, dtype=torch.int32, device=dev)
    expert_of = torch.empty(rows, dtype=torch.int32, device=dev)
    gamma = torch.empty(rows, dtype=torch.float32, device=dev)
    _sorted_route_meta[(triton.cdiv(rows, 1024),)](
        perm, sort_ids, weights, token_of, expert_of, gamma, rows, local, top, 1024
    )
    return perm, counts[:local], token_of, expert_of, gamma


def sort_routes(ids, weights, start, local, hidden, *, force_native=False):
    """Route sorting metadata for experts_forward: (perm, counts, token_of,
    expert_of, gamma). perm[r] is route r's sorted row; counts covers this rank's
    experts; sorted rows past their sum belong to other ranks."""
    flat = ids.reshape(-1)
    rows, top, dev = flat.numel(), ids.shape[1], ids.device
    flat_weights = weights.reshape(-1)
    if force_native or rows > SORT_MAX_ROUTES:
        return _sort_routes_native(
            flat, flat_weights, rows, start, local, hidden, top, dev
        )
    perm = torch.empty(rows, dtype=torch.int32, device=dev)
    counts = torch.empty(local, dtype=torch.int32, device=dev)
    token_of = torch.empty(rows, dtype=torch.int32, device=dev)
    expert_of = torch.empty(rows, dtype=torch.int32, device=dev)
    gamma = torch.empty(rows, dtype=torch.float32, device=dev)
    block = max(16, triton.next_power_of_2(rows))
    _route_sort[(1,)](
        flat,
        flat_weights,
        perm,
        counts,
        token_of,
        expert_of,
        gamma,
        rows,
        start,
        local,
        top,
        block,
        triton.next_power_of_2(local),
        num_warps=8,
    )
    return perm, counts, token_of, expert_of, gamma


def experts_forward(x, weights, ids, m, *, output_dtype=None):
    """Routed experts of one Inkling layer on sgl-kernel-xpu Xe20 grouped GEMMs.

    x [tokens, hidden] BF16, weights/ids [tokens, top_k]. NVFP4 (uint8 E2M1 with
    E4M3 K16 scales) and MXFP8 (E4M3 with UE8M0 K32 scales) expert weights; this
    rank's experts only. Activations are fake-quantized to the weight format at
    both GEMM inputs, NVFP4 accumulators are restored with the per-row and
    per-expert amax before SiTU, and each route's row is rounded to BF16 before
    the FP32 sum in source order. Under EP, routes to other ranks' experts sort
    into a trailing skip expert that the GEMMs never visit and the combine skips.
    """
    import sgl_kernel.moe  # noqa: F401  registers the Xe20 MoE ops

    tokens, hidden = x.shape
    top = ids.shape[1]
    out_dtype = output_dtype or x.dtype
    if tokens == 0:
        return torch.empty((0, hidden), dtype=out_dtype, device=x.device)
    fp4 = m.w13_weight.dtype == torch.uint8
    local = m.w13_weight.shape[0]
    start = m.ep_rank * local
    f = m.w2_weight.shape[2] * (2 if fp4 else 1)
    flat = ids.reshape(-1)
    rows = flat.numel()
    dev = x.device
    perm, local_counts, token_of, expert_of, gamma = sort_routes(
        ids, weights, start, local, hidden
    )
    # Quantize each token once and write it to its top_k sorted rows.
    x = x.contiguous()
    if fp4:
        q_sorted, xa = quantize_nv(x, perm, top)
    else:
        q_sorted, xa = quantize_mx(x, perm, top), None

    gate_up = torch.empty((rows, 2 * f), dtype=torch.bfloat16, device=dev)
    _xe20_gemm(gate_up, q_sorted, m.w13_weight, m.w13_scale, local_counts, local, fp4)
    hidden_act = torch.empty((rows, f), dtype=torch.bfloat16, device=dev)
    _restore_situ_rows[(rows, triton.cdiv(f, 256))](
        gate_up,
        hidden_act,
        token_of,
        expert_of,
        gamma,
        xa if fp4 else gamma,
        m.w13_amax if fp4 else gamma,
        m.w13_outer_scale_denominator if fp4 else gamma,
        local,
        f,
        fp4,
        float(m.gate_cap),
        float(m.up_cap),
        256,
        enable_fp_fusion=False,
    )
    q2, ha = quantize_nv(hidden_act) if fp4 else (quantize_mx(hidden_act), None)
    down = torch.empty((rows, hidden), dtype=torch.bfloat16, device=dev)
    _xe20_gemm(down, q2, m.w2_weight, m.w2_scale, local_counts, local, fp4)

    result = torch.empty((tokens, hidden), dtype=out_dtype, device=dev)
    _combine_sorted[(tokens, triton.cdiv(hidden, 1024))](
        down,
        perm,
        flat,
        ha if fp4 else gamma,
        m.w2_amax if fp4 else gamma,
        m.w2_outer_scale_denominator if fp4 else gamma,
        result,
        start,
        local,
        top,
        hidden,
        fp4,
        1024,
        enable_fp_fusion=False,
    )
    return result
