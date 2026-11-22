# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/xpu_communicator.py

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_xpu


class XpuCommunicator:
    def __init__(self, group: ProcessGroup):
        if not is_xpu():
            self.disabled = True
            return
        self.disabled = False
        self.group = group
        self.world_size = dist.get_world_size(self.group)

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        # Eager XCCL may start before work already queued on this stream finishes,
        # which hangs large TP2 prefill reductions; captured graphs order it.
        if not torch.xpu.is_current_stream_capturing():
            torch.xpu.current_stream(x.device).synchronize()
        dist.all_reduce(x, group=self.group)
        return x
