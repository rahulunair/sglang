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
