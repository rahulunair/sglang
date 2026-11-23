"""PyTorch references for the Inkling-2 XPU kernels (test oracles).

Each function is the eager formulation an XPU kernel must match: the activation
quantizers bit for bit, the reductions to FP32 accumulation order.
"""

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
