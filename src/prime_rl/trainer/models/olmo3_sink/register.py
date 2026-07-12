# Copyright 2026 proof-pilot. Apache-2.0.
"""Approach A: in-process dynamic registration.

Call `register_olmo3_sink()` once at the top of a training / eval script so that
`AutoConfig`/`AutoModel*` recognize `model_type="olmo3_sink"`. This only affects
the current Python process; it does NOT make a checkpoint portable on its own
(for that, see `convert.py` / trust_remote_code).
"""

from __future__ import annotations

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
)

from .configuration_olmo3_sink import Olmo3SinkConfig
from .modeling_olmo3_sink import Olmo3SinkForCausalLM, Olmo3SinkModel

_REGISTERED = False


def register_olmo3_sink(exist_ok: bool = True) -> None:
    """Register Olmo3Sink config/model and its MagiAttention sink backends."""
    global _REGISTERED
    if _REGISTERED:
        return
    AutoConfig.register("olmo3_sink", Olmo3SinkConfig, exist_ok=exist_ok)
    AutoModel.register(Olmo3SinkConfig, Olmo3SinkModel, exist_ok=exist_ok)
    AutoModelForCausalLM.register(Olmo3SinkConfig, Olmo3SinkForCausalLM, exist_ok=exist_ok)
    # Registration itself is dependency-free. The selected adapter imports Magi and
    # its matching FA package lazily, then fails with an actionable error if absent.
    from .attention import register_magi_sink_attentions

    register_magi_sink_attentions()
    _REGISTERED = True
