"""Inkling relative attention on the existing XPU cache/metadata infrastructure."""

from sglang.kernels.ops.attention.inkling2_relative_attention_xpu import (
    relative_paged_attention,
)
from sglang.srt.layers.attention.xpu_backend import XPUAttentionBackend
from sglang.srt.layers.radix_attention import AttentionType

# Split-K for decode/verify, measured on B70 verify shapes (24/8 heads, 9 rows):
# a 1023-key window and 64K-128K global contexts.
DECODE_SPLITS_WINDOW = 16
DECODE_SPLITS_GLOBAL = 32


class Inkling2XPUAttentionBackend(XPUAttentionBackend):
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
            (DECODE_SPLITS_WINDOW if is_swa else DECODE_SPLITS_GLOBAL)
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
