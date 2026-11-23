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
