# Copyright 2026 proof-pilot. Apache-2.0.
"""Adapter for Prime-RL's original in-kernel FlashAttention-3 sink path."""

from __future__ import annotations

from importlib import import_module

import torch

NATIVE_FA3_SINK_ATTN_IMPL = "olmo3_sink_fa3_native"


def validate_native_fa3_sink_backend() -> None:
    """Fail before model construction when the original FA3 path is unavailable."""
    if not torch.cuda.is_available():
        raise RuntimeError(f"{NATIVE_FA3_SINK_ATTN_IMPL} requires a CUDA device")

    major, minor = torch.cuda.get_device_capability()
    if major != 9:
        raise RuntimeError(
            f"{NATIVE_FA3_SINK_ATTN_IMPL} is Hopper-only, but the active device is SM{major}{minor}"
        )

    try:
        import_module("prime_rl.trainer.models.olmo3_sink.fa3_sink_kernel")
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            f"{NATIVE_FA3_SINK_ATTN_IMPL} requires Prime-RL's original FA3 sink kernel "
            "and flash_attn_interface"
        ) from exc


def native_fa3_varlen_attention_with_sink(*args, **kwargs) -> torch.Tensor:
    """Call the original custom-op-backed FA3 sink implementation lazily."""
    module = import_module("prime_rl.trainer.models.olmo3_sink.fa3_sink_kernel")
    return module.fa3_varlen_attn_with_sink_kernel(*args, **kwargs)
