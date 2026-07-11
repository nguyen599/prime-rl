## a bit of context here, this basically copy AutoModelForCausalLM from transformers, but use our own model instead

import logging
from collections import OrderedDict
from importlib import import_module

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.models.auto.auto_factory import _BaseAutoModelClass, _LazyAutoMapping, auto_class_update
from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from prime_rl.trainer.models.base import PreTrainedModelPrimeRL
from prime_rl.trainer.models.layers.lm_head import PrimeLmOutput, cast_float_and_contiguous
from prime_rl.trainer.models.olmo3_sink import (
    Olmo3SinkConfig,
    Olmo3SinkForCausalLM,
    register_olmo3_sink,
)

logger = logging.getLogger(__name__)


def _optional_import(module_name: str, *symbols: str):
    try:
        module = import_module(module_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Skipping optional Prime-RL model module %s during registry init: %s: %s",
            module_name,
            type(exc).__name__,
            exc,
        )
        return tuple(None for _ in symbols)
    return tuple(getattr(module, symbol) for symbol in symbols)


(LlamaForCausalLM,) = _optional_import("prime_rl.trainer.models.llama", "LlamaForCausalLM")
(Qwen3ForCausalLM,) = _optional_import("prime_rl.trainer.models.qwen3", "Qwen3ForCausalLM")
(Qwen3_5ForCausalLM,) = _optional_import("prime_rl.trainer.models.qwen3_5", "Qwen3_5ForCausalLM")
AfmoeConfig, AfmoeForCausalLM = _optional_import("prime_rl.trainer.models.afmoe", "AfmoeConfig", "AfmoeForCausalLM")
Glm4MoeConfig, Glm4MoeForCausalLM = _optional_import(
    "prime_rl.trainer.models.glm4_moe", "Glm4MoeConfig", "Glm4MoeForCausalLM"
)
GlmMoeDsaConfig, GlmMoeDsaForCausalLM = _optional_import(
    "prime_rl.trainer.models.glm_moe_dsa", "GlmMoeDsaConfig", "GlmMoeDsaForCausalLM"
)
GptOssConfig, GptOssForCausalLM = _optional_import(
    "prime_rl.trainer.models.gpt_oss", "GptOssConfig", "GptOssForCausalLM"
)
LagunaConfig, LagunaForCausalLM = _optional_import(
    "prime_rl.trainer.models.laguna", "LagunaConfig", "LagunaForCausalLM"
)
MiniMaxM2Config, MiniMaxM2ForCausalLM = _optional_import(
    "prime_rl.trainer.models.minimax_m2", "MiniMaxM2Config", "MiniMaxM2ForCausalLM"
)
NemotronHConfig, NemotronHForCausalLM = _optional_import(
    "prime_rl.trainer.models.nemotron_h", "NemotronHConfig", "NemotronHForCausalLM"
)
Qwen3MoeConfig, Qwen3MoeForCausalLM = _optional_import(
    "prime_rl.trainer.models.qwen3_moe", "Qwen3MoeConfig", "Qwen3MoeForCausalLM"
)
Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM = _optional_import(
    "prime_rl.trainer.models.qwen3_5_moe", "Qwen3_5MoeConfig", "Qwen3_5MoeForCausalLM"
)

# Make custom config discoverable by AutoConfig
register_olmo3_sink()
for _model_type, _config_cls in (
    ("afmoe", AfmoeConfig),
    ("glm4_moe", Glm4MoeConfig),
    ("glm_moe_dsa", GlmMoeDsaConfig),
    ("laguna", LagunaConfig),
    ("minimax_m2", MiniMaxM2Config),
    ("nemotron_h", NemotronHConfig),
    ("qwen3_moe", Qwen3MoeConfig),
    ("qwen3_5_text", Qwen3_5TextConfig),
    ("qwen3_5_moe_text", Qwen3_5MoeConfig),
):
    if _config_cls is not None:
        AutoConfig.register(_model_type, _config_cls, exist_ok=True)
# GptOssConfig is just HF's class - already registered by transformers, no override needed.

_CUSTOM_CAUSAL_LM_MAPPING = _LazyAutoMapping(CONFIG_MAPPING_NAMES, OrderedDict())
for _config_cls, _model_cls in (
    (LlamaConfig, LlamaForCausalLM),
    (Qwen3Config, Qwen3ForCausalLM),
    (AfmoeConfig, AfmoeForCausalLM),
    (Glm4MoeConfig, Glm4MoeForCausalLM),
    (GlmMoeDsaConfig, GlmMoeDsaForCausalLM),
    (LagunaConfig, LagunaForCausalLM),
    (MiniMaxM2Config, MiniMaxM2ForCausalLM),
    (NemotronHConfig, NemotronHForCausalLM),
    (Qwen3MoeConfig, Qwen3MoeForCausalLM),
    (Qwen3_5TextConfig, Qwen3_5ForCausalLM),
    (Qwen3_5MoeConfig, Qwen3_5MoeForCausalLM),
    (GptOssConfig, GptOssForCausalLM),
):
    if _config_cls is not None and _model_cls is not None:
        _CUSTOM_CAUSAL_LM_MAPPING.register(_config_cls, _model_cls, exist_ok=True)
_CUSTOM_CAUSAL_LM_MAPPING.register(Olmo3SinkConfig, Olmo3SinkForCausalLM, exist_ok=True)


class AutoModelForCausalLMPrimeRL(_BaseAutoModelClass):
    _model_mapping = _CUSTOM_CAUSAL_LM_MAPPING


AutoModelForCausalLMPrimeRL = auto_class_update(AutoModelForCausalLMPrimeRL, head_doc="causal language modeling")


def supports_custom_impl(model_config: PretrainedConfig) -> bool:
    """Check if the model configuration supports the custom PrimeRL implementation.

    Args:
        model_config: The model configuration to check.

    Returns:
        True if the model supports custom implementation, False otherwise.
    """
    return type(model_config) in _CUSTOM_CAUSAL_LM_MAPPING


# Mapping from HF composite VLM model_type to custom PrimeRL class.
# Used by get_model() to dispatch VLMs that have a custom text model implementation.
# Points to the same unified class — the config drives text-only vs VLM behavior.
_CUSTOM_VLM_MAPPING: dict[str, type] = {
    key: cls
    for key, cls in {
        "qwen3_5_moe": Qwen3_5MoeForCausalLM,
    }.items()
    if cls is not None
}


def get_custom_vlm_cls(model_config: PretrainedConfig) -> type | None:
    """Return the custom PrimeRL VLM class for this config, or None if unsupported."""
    return _CUSTOM_VLM_MAPPING.get(getattr(model_config, "model_type", None))


__all__ = [
    "AutoModelForCausalLMPrimeRL",
    "Olmo3SinkConfig",
    "Olmo3SinkForCausalLM",
    "PreTrainedModelPrimeRL",
    "register_olmo3_sink",
    "supports_custom_impl",
    "get_custom_vlm_cls",
    "PrimeLmOutput",
    "cast_float_and_contiguous",
]
