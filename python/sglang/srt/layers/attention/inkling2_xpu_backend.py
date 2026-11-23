"""Inkling relative attention on the existing XPU cache/metadata infrastructure."""

from sglang.kernels.ops.attention.inkling2_relative_attention_xpu import (
    relative_paged_attention,
)
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.layers.radix_attention import AttentionType
from sglang.srt.runtime_context import get_exec

# Split-K programs per decode/verify call (requests x KV heads x splits), measured on
# B70 at 1, 2, 4 and 8 requests with 2, 4 and 8 KV heads per rank, 1.5K-128K contexts.
DECODE_PROGRAMS_WINDOW = 128
DECODE_PROGRAMS_GLOBAL = 256
DECODE_MAX_SPLITS = 128


def _decode_splits(
    *, requests: int, kv_heads: int, is_swa: bool, batch_invariant: bool
) -> int:
    # Batch-invariant mode sizes for one request: the split count sets the LSE merge
    # order, so it must not depend on the other requests in the batch.
    programs = DECODE_PROGRAMS_WINDOW if is_swa else DECODE_PROGRAMS_GLOBAL
    rows = kv_heads if batch_invariant else requests * kv_heads
    splits = max(1, programs // max(1, rows))
    return min(DECODE_MAX_SPLITS, 1 << (splits.bit_length() - 1))


class Inkling2XPUAttentionBackend(XPUAttentionBackend):
    def __init__(self, model_runner, **kwargs):
        super().__init__(model_runner, **kwargs)
        self.batch_invariant = get_exec().deterministic.enable_deterministic_inference

    def _forward_relative(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        save_kv_cache=True,
        *,
        rel_bias=None,
        return_lse=False,
        **kwargs,
    ):
        bidirectional = layer.attn_type in (
            AttentionType.ENCODER_ONLY,
            AttentionType.DECODER_BIDIRECTIONAL,
        )
        if rel_bias is None and not bidirectional:
            raise ValueError("Inkling target relative logits were not supplied")
        if self.use_mla or self.dcp_size != 1:
            raise NotImplementedError("Inkling relative attention uses GQA without DCP")
        if forward_batch.spec_info is not None and self.topk > 1:
            raise NotImplementedError(
                "Tree ancestry masking is not yet qualified on the XPU port"
            )
        if k is not None and save_kv_cache:
            self._set_kv_buffer_mha(
                layer, forward_batch, forward_batch.out_cache_loc, k, v
            )
        meta = self.forward_metadata
        is_swa = (
            layer.sliding_window_size is not None and layer.sliding_window_size >= 0
        )
        pages = meta.page_table
        if is_swa and self.use_sliding_window_kv_pool:
            if meta.swa_page_table is None:
                raise RuntimeError("SWA cache translation metadata is missing")
            pages = meta.swa_page_table
        key, value = self.kv_layers.read(layer.layer_id).kv()
        key = key.view(-1, self.page_size, layer.tp_k_head_num, layer.head_dim)
        value = value.view(-1, self.page_size, layer.tp_v_head_num, layer.v_head_dim)
        query = q.reshape(-1, layer.tp_q_head_num, layer.head_dim)
        mode = forward_batch.forward_mode
        splits = (
            _decode_splits(
                requests=forward_batch.batch_size,
                kv_heads=layer.tp_k_head_num,
                is_swa=is_swa,
                batch_invariant=self.batch_invariant,
            )
            if mode.is_decode() or mode.is_target_verify()
            else 1
        )
        out, lse = relative_paged_attention(
            query,
            key,
            value,
            rel_bias,
            pages,
            meta.cache_seqlens_int32,
            meta.cu_seqlens_q,
            max_query_length=meta.max_seq_len_q,
            scale=layer.scaling,
            window_left=layer.sliding_window_size if is_swa else -1,
            splits=splits,
            causal=not bidirectional,
            window_anchor_at_start=layer.window_anchor_at_start,
            max_cache_length=meta.max_seq_len_k,
        )
        return (out, lse) if return_lse else out.to(q.dtype).reshape(query.shape[0], -1)

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self._forward_relative(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_decode(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        return self._forward_relative(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )
