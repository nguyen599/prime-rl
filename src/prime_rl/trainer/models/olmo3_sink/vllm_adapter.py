"""vLLM adapter for converted ``olmo3_sink`` checkpoints.

The generic vLLM attention layer already supports a per-head ``sinks`` tensor.
This adapter reuses vLLM's OLMo2/OLMo3 implementation and only swaps the
attention modules so checkpoints with ``Olmo3SinkForCausalLM`` can be served by
verl rollouts.
"""

from __future__ import annotations

from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from torch import nn

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.distributed.communication_op import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.distributed.utils import split_tensor_along_last_dim
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import EagleModelMixin, SupportsEagle3
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.olmo2 import (
    Olmo2MLP,
    maybe_prefix,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors


def _normalize_rope_parameters(config, rope_parameters: dict) -> dict:
    normalized = {}
    for key, value in rope_parameters.items():
        # Some OLMo3 configs store per-layer-type RoPE settings under nested
        # dictionaries. vLLM caches RoPE modules by hashing this dict's values,
        # so any still-nested dict would crash in get_rope().
        if isinstance(value, dict):
            continue
        normalized[key] = value
    if "type" in normalized and "rope_type" not in normalized:
        normalized["rope_type"] = normalized.pop("type")
    if "attention_factor" in normalized and "attn_factor" not in normalized:
        normalized["attn_factor"] = normalized.pop("attention_factor")
    normalized.setdefault("rope_theta", getattr(config, "rope_theta", 500000))
    return normalized


def _base_rope_parameters(config, sliding_window: int | None) -> dict:
    raw_parameters = getattr(config, "rope_parameters", None)
    if not raw_parameters:
        raw_parameters = getattr(config, "rope_scaling", None) or {}
    raw_parameters = dict(raw_parameters)

    layer_type = "sliding_attention" if sliding_window is not None else "full_attention"
    if isinstance(raw_parameters.get(layer_type), dict):
        raw_parameters = dict(raw_parameters[layer_type])

    return _normalize_rope_parameters(config, raw_parameters)


def _rope_parameters_for_layer(config, sliding_window: int | None) -> dict:
    """Match OLMo3/olmo3_sink serving: YaRN only on full-attention layers."""
    rope_parameters = _base_rope_parameters(config, sliding_window)
    if sliding_window is None:
        return rope_parameters
    return {
        "rope_type": "default",
        "rope_theta": rope_parameters.get(
            "rope_theta", getattr(config, "rope_theta", 500000)
        ),
    }


class Olmo3SinkAttention(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config

        hidden_size = self.config.hidden_size
        self.tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = self.config.num_attention_heads

        assert hidden_size % self.total_num_heads == 0
        assert self.total_num_heads % self.tp_size == 0

        self.num_heads = self.total_num_heads // self.tp_size
        self.total_num_kv_heads = (
            self.config.num_key_value_heads or self.total_num_heads
        )
        if self.total_num_kv_heads >= self.tp_size:
            assert self.total_num_kv_heads % self.tp_size == 0
        else:
            assert self.tp_size % self.total_num_kv_heads == 0

        self.num_kv_heads = max(1, self.total_num_kv_heads // self.tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.max_position_embeddings = self.config.max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.tp_rank = get_tensor_model_parallel_rank()
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=self.config.rms_norm_eps,
        )
        self.q_norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)

        self.scaling = self.head_dim**-0.5

        layer_idx = extract_layer_index(prefix)
        sliding_window = None
        layer_types = getattr(self.config, "layer_types", None)
        if layer_types is not None and layer_types[layer_idx] == "sliding_attention":
            sliding_window = self.config.sliding_window

        init_value = float(getattr(self.config, "sink_init_value", -10.0))
        self.sinks = nn.Parameter(torch.full((self.num_heads,), init_value), requires_grad=False)

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=self.max_position_embeddings,
            rope_parameters=_rope_parameters_for_layer(self.config, sliding_window),
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=vllm_config.quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
            sinks=self.sinks,
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.o_proj",
        )

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.tp_size > 1:
            q = tensor_model_parallel_all_gather(q.contiguous())
            k = tensor_model_parallel_all_gather(k.contiguous())
        q = self.q_norm(q)
        k = self.k_norm(k)
        if self.tp_size > 1:
            splitter = partial(split_tensor_along_last_dim, num_partitions=self.tp_size)
            q = splitter(q)[self.tp_rank]
            k = splitter(k)[self.tp_rank]
        return q, k

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Olmo3SinkDecoderLayer(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.self_attn = Olmo3SinkAttention(vllm_config=vllm_config, prefix=f"{prefix}.self_attn")
        self.mlp = Olmo2MLP(vllm_config=vllm_config, prefix=f"{prefix}.mlp")
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Olmo3SinkModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: Olmo3SinkDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            assert isinstance(hidden_states, torch.Tensor)

        aux_hidden_states: list[torch.Tensor] = []
        if 0 in self.aux_hidden_state_layers:
            aux_hidden_states.append(hidden_states)

        for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
            hidden_states = layer(positions, hidden_states)
            layer_id = self.start_layer + idx + 1
            if layer_id in self.aux_hidden_state_layers:
                aux_hidden_states.append(hidden_states)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states})

        hidden_states = self.norm(hidden_states)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if is_pp_missing_parameter(name, self):
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader  # type: ignore[attr-defined]
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Olmo3SinkForCausalLM(nn.Module, SupportsEagle3):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = Olmo3SinkModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        if self.config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens
        else:
            self.lm_head = ParallelLMHead(
                self.config.vocab_size,
                self.config.hidden_size,
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        self.model._set_aux_hidden_state_layers(layers)

    def get_eagle3_default_aux_hidden_state_layers(self) -> tuple[int, ...]:
        num_layers = len(self.model.layers)
        return (2, num_layers // 2, num_layers - 3)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        heads_per_rank = self.config.num_attention_heads // tp_size
        head_start = tp_rank * heads_per_rank
        params = dict(self.named_parameters(remove_duplicate=False))
        loaded_sinks: set[str] = set()

        def filtered_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.endswith(".self_attn.sinks"):
                    param = params.get(name)
                    if param is None:
                        continue
                    shard = weight.narrow(0, head_start, heads_per_rank)
                    shard = shard.to(dtype=param.dtype)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, shard)
                    loaded_sinks.add(name)
                    continue
                yield name, weight

        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head.weight"] if self.config.tie_word_embeddings else None),
        )
        loaded = loader.load_weights(filtered_weights())
        return set(loaded) | loaded_sinks
