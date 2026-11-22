"""Tiled prefix-valid KV commit on XPU against the eager masked write (milliseconds).

The DSpark draft commits each request's accepted prefix of the verify block into
the KV pool. On XPU this must use the tiled Triton kernel with no host sync; the
result must equal writing exactly the first commit_lens[i] rows of request i.
"""

import pytest
import torch

from sglang.srt.mem_cache.memory_pool import _set_kv_buffer_prefix_valid_impl
from sglang.test.ci.ci_register import register_xpu_ci

register_xpu_ci(est_time=5, suite="extra-test-1-gpu-xpu")

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU only")


@pytest.mark.parametrize(
    "batch,block,heads,dim",
    [(1, 8, 2, 128), (4, 8, 2, 128), (8, 8, 8, 128), (3, 16, 1, 64)],
)
def test_prefix_valid_commit_matches_masked_write(batch, block, heads, dim):
    torch.manual_seed(batch * 100 + block)
    slots = 4096
    row = heads * dim
    k_cache = torch.randn(slots, heads, dim, device="xpu").bfloat16()
    v_cache = torch.randn(slots, heads, dim, device="xpu").bfloat16()
    want_k, want_v = k_cache.clone(), v_cache.clone()
    loc_2d = torch.randperm(slots, device="xpu")[: batch * block].view(batch, block)
    commit = torch.randint(0, block + 1, (batch,), device="xpu", dtype=torch.int32)
    commit[0] = block
    if batch > 1:
        commit[1] = 0
    k = torch.randn(batch * block, heads, dim, device="xpu").bfloat16()
    v = torch.randn(batch * block, heads, dim, device="xpu").bfloat16()
    valid = (torch.arange(block, device="xpu")[None, :] < commit[:, None]).reshape(-1)
    want_k[loc_2d.reshape(-1)[valid]] = k[valid]
    want_v[loc_2d.reshape(-1)[valid]] = v[valid]
    _set_kv_buffer_prefix_valid_impl(
        k, v, k_cache, v_cache, loc_2d, commit, row_dim=row, store_dtype=torch.bfloat16
    )
    assert torch.equal(k_cache, want_k) and torch.equal(v_cache, want_v)
