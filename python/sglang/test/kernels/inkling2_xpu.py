"""PyTorch references for the Inkling-2 XPU kernels (test oracles).

Each function is the eager formulation an XPU kernel must match: the activation
quantizers bit for bit, the reductions to FP32 accumulation order.
"""

import math

import torch


def rms_norm_reference(x, weight, eps, *, pre_weight=None, output_dtype=None, out=None):
    value = x.float()
    if pre_weight is not None:
        value = value * pre_weight.float()
    value = value * torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        value = value * weight.float()
    value = value.to(output_dtype or x.dtype)
    if out is not None:
        out.copy_(value)
        return out
    return value


def qk_l2_clip_reference(
    x, threshold, eps=1e-6, *, output_dtype=None, scale=None, output_scale=None
):
    shape = x.shape
    rows = x.reshape(-1, shape[-1]).float()
    if scale is not None:
        if scale.ndim == 2:
            indices = torch.arange(rows.shape[0], device=x.device) % scale.shape[0]
            rows = rows * scale.float()[indices]
        else:
            rows = rows * scale.float()
    # Keep the same reduction tree for a decode row and a verification block.
    # Native XPU sum can change rounding with the number of rows, enough to
    # cross a BF16 rounding boundary before attention. Reuse the fixed-row
    # reduction already used by deterministic inference.
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import mean_dim

    norm = (mean_dim(rows.square(), -1, keepdim=True) * shape[-1]).sqrt()
    value = rows * (threshold / (norm + eps)).clamp(max=1.0)
    dtype = output_dtype or x.dtype
    value = value.to(dtype)
    if output_scale is not None and rows.shape[0]:
        assert rows.shape[0] % output_scale.numel() == 0
        value = (
            value.float()
            * output_scale.repeat_interleave(rows.shape[0] // output_scale.numel())[
                :, None
            ]
        ).to(dtype)
    return value.reshape(shape)


def attn_res_reference(blocks, partial_addend, queries, *, normalize_values=False):
    assert blocks and queries.ndim == 2 and queries.dtype == torch.float32
    # The addend is summed in FP32, without an intermediate source-dtype rounding.
    sources = [x.float() for x in blocks]
    if partial_addend is not None:
        sources[-1] = sources[-1] + partial_addend.float()
    values = torch.stack(sources, dim=0)
    rstd = torch.rsqrt(values.square().mean(dim=-1) + 1e-6)
    logits = torch.einsum("std,qd->qst", values, queries) * rstd.unsqueeze(0)
    # Keep each source reduction contiguous for decode/verify batch invariance.
    # XPU non-last-axis softmax changes rounding with the token-axis stride.
    probs = torch.softmax(logits.permute(0, 2, 1).contiguous(), dim=-1).permute(0, 2, 1)
    if normalize_values:
        probs = probs * rstd.unsqueeze(0)
    result = torch.einsum("qst,std->qtd", probs, values)
    return tuple(result.unbind(0))


def residual_add_reference(*inputs):
    value = inputs[0].float()
    for item in inputs[1:]:
        value = value + item.float()
    return value.to(inputs[0].dtype)


def message_controllers_reference(dests, messages, down, gain, forget, eps=1e-6):
    assert len(dests) == len(messages) == down.shape[0]
    outputs = []
    for i, (dest, message) in enumerate(zip(dests, messages)):
        dest, message = dest.float(), message.float()
        pair = torch.cat(
            (
                rms_norm_reference(dest, None, eps),
                rms_norm_reference(message, None, eps),
            ),
            dim=-1,
        ).to(torch.bfloat16)
        # Source sums K1024 FP32 partials in order before the BF16 cast.
        hidden = None
        for start in range(0, pair.shape[-1], 1024):
            part = (
                pair[..., start : start + 1024].float()
                @ down[i, :, start : start + 1024].float().T
            )
            hidden = part if hidden is None else hidden + part
        hidden = hidden.to(torch.bfloat16).float()
        gain_value = hidden @ gain[i].float().T
        forget_value = hidden @ forget[i].float().T
        outputs.append(
            (1.0 - torch.tanh(forget_value)) * dest + torch.tanh(gain_value) * message
        )
    return torch.stack(outputs)


# One program per row keeps decode and verify rows on the same reduction tree.


def fp4round(x):
    a = x.abs()
    v = torch.where(
        a <= 0.25,
        0.0,
        torch.where(
            a < 0.75,
            0.5,
            torch.where(
                a <= 1.25,
                1.0,
                torch.where(
                    a < 1.75,
                    1.5,
                    torch.where(
                        a <= 2.5,
                        2.0,
                        torch.where(a < 3.5, 3.0, torch.where(a <= 5.0, 4.0, 6.0)),
                    ),
                ),
            ),
        ),
    )
    return torch.where(x < 0, -v, v)


def nv_reference(x):
    """Vectorized NVFP4 activation fake quantization (values, per-row amax)."""
    x = x.float()
    z = x.reshape(x.shape[0], -1, 16)
    a = x.abs().amax(-1)
    enc = (1536.0 / a).clamp(max=torch.finfo(torch.float32).max)
    dec = 1.0 / enc
    mx = z.abs().amax(-1)
    candidates = []
    for den in (6.0, 4.0):
        s = (mx * (enc[:, None] / den)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
        inv = (1.0 / (s * dec[:, None])).clamp(max=torch.finfo(torch.float32).max)
        v = fp4round(z * inv[..., None]) * s[..., None]
        err = (z - v * (a[:, None, None] / 1536.0)).abs().sum(-1)
        candidates.append((v, err))
    return torch.where(
        (candidates[1][1] < candidates[0][1])[..., None],
        candidates[1][0],
        candidates[0][0],
    ).reshape_as(x).to(torch.bfloat16), a


def mx_reference(x):
    """Vectorized MXFP8 activation fake quantization."""
    x = x.float().reshape(x.shape[0], -1, 32)
    raw = x.abs().amax(-1) / 448.0
    p = ((raw.contiguous().view(torch.int32) + 0x007FFFFF) & 0x7F800000) >> 23
    p = torch.where(raw <= 2.0**-127, 0, p.clamp(max=254))
    s = torch.where(p == 0, 2.0**-127, (p << 23).contiguous().view(torch.float32))
    return (
        (
            (x / s[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()
            * s[..., None]
        )
        .reshape(x.shape[0], -1)
        .to(torch.bfloat16)
    )


def fp8_round(value):
    return (
        torch.tensor(value).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float().item()
    )


def scalar_nv(x):
    # Scalar, codebook-based reference independent of vectorized implementation.
    codebook = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
    rows, width = x.shape
    out = torch.empty((rows, width), dtype=torch.bfloat16)
    maxima = torch.empty(rows, dtype=torch.float32)
    for row in range(rows):
        values = x[row].float().tolist()
        amax = max(map(abs, values))
        maxima[row] = amax
        if amax == 0:
            out[row].zero_()
            continue
        encoder = min(1536.0 / amax, torch.finfo(torch.float32).max)
        for start in range(0, width, 16):
            group = values[start : start + 16]
            maximum = max(map(abs, group))
            candidates = []
            for denominator in (6.0, 4.0):
                scale = fp8_round(min(maximum * encoder / denominator, 448.0))
                decoded = []
                for value in group:
                    normalized = value * encoder / scale if scale else 0.0
                    magnitude = abs(normalized)
                    best = min(
                        range(8), key=lambda i: (abs(codebook[i] - magnitude), i % 2)
                    )
                    decoded.append(math.copysign(codebook[best] * scale, value))
                error = sum(
                    abs(value - q * amax / 1536.0) for value, q in zip(group, decoded)
                )
                candidates.append((error, decoded))
            chosen = (
                candidates[1] if candidates[1][0] < candidates[0][0] else candidates[0]
            )
            out[row, start : start + 16] = torch.tensor(chosen[1])
    return out, maxima


def scalar_mx(x):
    rows, width = x.shape
    out = torch.empty((rows, width), dtype=torch.bfloat16)
    for row in range(rows):
        for start in range(0, width, 32):
            group = x[row, start : start + 32].float().tolist()
            maximum = max(map(abs, group))
            exponent = (
                max(-127, min(127, math.ceil(math.log2(maximum / 448.0))))
                if maximum
                else -127
            )
            scale = 2.0**exponent
            out[row, start : start + 32] = torch.tensor(
                [fp8_round(v / scale) * scale for v in group]
            )
    return out
