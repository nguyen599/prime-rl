# Copyright 2026 proof-pilot. Apache-2.0.
"""Transformers attention-interface adapter for MagiAttention sink kernels.

Registers explicit FA2/FA3/FA4 implementations. The adapter converts transformers'
[B, H, S, D] layout to varlen [total, H, D] + cu_seqlens. Deliberately not
registered in the mask interface, so
`create_causal_mask` returns None (no [S,S] mask is built) and we rely on cu_seqlens.
"""
from __future__ import annotations

import torch
from transformers import AttentionInterface  # public top-level API
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    prepare_fa_kwargs_from_position_ids,
)

from .magi_sink import MAGI_SINK_ATTN_IMPLS, magi_varlen_attention_with_sink


def magi_sink_attention_forward(
    module,
    query: torch.Tensor,   # [B, Hq, S, D]
    key: torch.Tensor,     # [B, Hkv, S, D]
    value: torch.Tensor,   # [B, Hkv, S, D]
    attention_mask=None,   # ignored: varlen uses cu_seqlens
    scaling: float | None = None,
    dropout: float = 0.0,
    sliding_window: int | None = None,
    s_aux: torch.Tensor | None = None,
    **kwargs,
):
    B, Hq, S, D = query.shape
    Hkv = key.shape[1]
    sink = s_aux if s_aux is not None else module.sinks

    # [B, H, S, D] -> varlen [B*S, H, D]
    q = query.transpose(1, 2).reshape(B * S, Hq, D)
    k = key.transpose(1, 2).reshape(B * S, Hkv, D)
    v = value.transpose(1, 2).reshape(B * S, Hkv, D)

    # Preferred: varlen metadata computed once in Olmo3SinkModel.forward (reuse, no
    # per-layer recompute). max_* are already ints there.
    cu_q = kwargs.get("cu_seq_lens_q")
    cu_k = kwargs.get("cu_seq_lens_k")
    max_q = kwargs.get("max_length_q")
    max_k = kwargs.get("max_length_k")
    if cu_q is None:
        # Fallback (e.g. reuse_packing_metadata disabled, or used outside Olmo3SinkModel):
        # derive packed boundaries from position_ids so doc isolation stays CORRECT.
        position_ids = kwargs.get("position_ids")
        if position_ids is not None and _is_packed_sequence(position_ids, B):
            (cu_q, cu_k), (mq, mk) = prepare_fa_kwargs_from_position_ids(position_ids)
            max_q, max_k = int(mq), int(mk)
        else:
            # truly unpacked: B rows of full length S
            cu_q = torch.arange(0, (B + 1) * S, S, device=q.device, dtype=torch.int32)
            cu_k = cu_q
            max_q = max_k = S

    window = (sliding_window - 1, 0) if sliding_window is not None else (-1, -1)
    attn_impl = module.config._attn_implementation
    out = magi_varlen_attention_with_sink(
        q, k, v, sink, cu_q, cu_k, max_q, max_k,
        attn_impl=attn_impl,
        softmax_scale=scaling,
        causal=True,
        window_size=window,
        dropout_p=dropout,
    )  # [B*S, Hq, D]
    return out.reshape(B, S, Hq, D), None


def register_magi_sink_attentions() -> None:
    for attn_name in MAGI_SINK_ATTN_IMPLS:
        AttentionInterface.register(attn_name, magi_sink_attention_forward)


# Compatibility aliases for callers written against the first FA3-only adapter.
fa3_sink_attention_forward = magi_sink_attention_forward
register_fa3_sink_attention = register_magi_sink_attentions
