from __future__ import annotations

import torch


def inputs_on_cuda(*args, **kwargs) -> bool:
    """Route kernel dispatch by input placement: the first tensor argument
    decides. CUDA and XPU inputs take the fused triton kernel; CPU inputs take
    the torch reference implementation (CPU-side callers such as unit tests
    exercise the reference path)."""
    for value in (*args, *kwargs.values()):
        if isinstance(value, torch.Tensor):
            return value.is_cuda or value.is_xpu
    raise AssertionError("kernel dispatch requires at least one tensor argument")
