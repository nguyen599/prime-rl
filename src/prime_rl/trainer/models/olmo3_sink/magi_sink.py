# Copyright 2026 proof-pilot. Apache-2.0.
"""Lazy adapters for MagiAttention's FlashAttention sink extensions."""

from __future__ import annotations

from importlib import import_module
from typing import Callable

import torch

MAGI_SINK_ATTN_IMPLS = frozenset(
    {
        "olmo3_sink_fa2",
        "olmo3_sink_fa3",
        "olmo3_sink_fa4",
    }
)

_BACKEND_MODULES = {
    "olmo3_sink_fa2": "magi_attn_extensions.fa2_interface_with_sink",
    "olmo3_sink_fa3": "magi_attn_extensions.fa3_interface_with_sink",
    "olmo3_sink_fa4": "magi_attn_extensions.fa4_interface_with_sink",
}
_BACKEND_FUNCTIONS = {
    "olmo3_sink_fa2": "fa2_varlen_func_with_sink",
    "olmo3_sink_fa3": "fa3_varlen_func_with_sink",
    "olmo3_sink_fa4": "fa4_varlen_func_with_sink",
}


def is_magi_sink_backend(attn_impl: str | None) -> bool:
    return attn_impl in MAGI_SINK_ATTN_IMPLS


def get_magi_sink_varlen_func(attn_impl: str) -> Callable:
    if attn_impl not in MAGI_SINK_ATTN_IMPLS:
        raise ValueError(f"Unknown MagiAttention sink backend: {attn_impl!r}")

    try:
        module = import_module(_BACKEND_MODULES[attn_impl])
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"{attn_impl} requires MagiAttention and magi_attn_extensions. "
            "Install the matching FlashAttention package plus both Magi packages."
        ) from exc

    if attn_impl == "olmo3_sink_fa4" and not getattr(module, "is_fa4_installed", False):
        raise RuntimeError(
            "olmo3_sink_fa4 requires flash-attn-4 with the flash_attn.cute interface"
        )
    return getattr(module, _BACKEND_FUNCTIONS[attn_impl])


def validate_magi_sink_backend(attn_impl: str) -> None:
    """Fail before model construction when the selected kernel cannot run."""
    get_magi_sink_varlen_func(attn_impl)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{attn_impl} requires a CUDA device")

    major, minor = torch.cuda.get_device_capability()
    if attn_impl == "olmo3_sink_fa3" and major != 9:
        raise RuntimeError(
            f"olmo3_sink_fa3 is Hopper-only, but the active device is SM{major}{minor}. "
            "Use olmo3_sink_fa2 or olmo3_sink_fa4 on supported hardware."
        )
    if attn_impl == "olmo3_sink_fa4" and major < 10:
        raise RuntimeError(
            f"olmo3_sink_fa4 requires Blackwell (SM100+), but the active device is SM{major}{minor}"
        )


def magi_varlen_attention_with_sink(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    *,
    attn_impl: str,
    softmax_scale: float | None,
    causal: bool,
    window_size: tuple[int, int] = (-1, -1),
    dropout_p: float = 0.0,
) -> torch.Tensor:
    """Run a Magi sink kernel with Prime-RL's common packed layout."""
    if sink.ndim == 1:
        sink = sink.unsqueeze(0)
    if sink.ndim != 2 or sink.shape[1] != q.shape[1]:
        raise ValueError(
            "OLMo3 sink must have shape [num_sink_tokens, num_query_heads]; "
            f"got {tuple(sink.shape)} for {q.shape[1]} query heads"
        )

    flash_fn = get_magi_sink_varlen_func(attn_impl)
    common_kwargs = {
        "sink": sink,
        "sink_layout": "sh",
        "softmax_scale": softmax_scale,
        "causal": causal,
        "return_attn_probs": False,
    }

    if attn_impl == "olmo3_sink_fa2":
        return flash_fn(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            dropout_p=dropout_p,
            window_size=window_size,
            **common_kwargs,
        )

    if dropout_p:
        raise ValueError(f"{attn_impl} does not support attention dropout; got {dropout_p}")
    if attn_impl == "olmo3_sink_fa3":
        return flash_fn(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            int(max_seqlen_q),
            int(max_seqlen_k),
            window_size=window_size,
            **common_kwargs,
        )

    if window_size != (-1, -1):
        raise ValueError(
            "MagiAttention's FA4 sink interface does not support sliding-window attention. "
            "Use olmo3_sink_fa2 for standard OLMo3 mixed full/sliding layers."
        )
    return flash_fn(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        int(max_seqlen_k),
        **common_kwargs,
    )
