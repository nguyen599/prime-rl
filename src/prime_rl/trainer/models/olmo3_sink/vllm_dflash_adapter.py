"""vLLM DFlash draft adapter for OLMo3Sink checkpoints.

The upstream vLLM ``DFlashDraftModel`` implementation is Qwen3-shaped: it uses
per-head Q/K RMSNorm and Qwen residual ordering. The OLMo3Sink DFlash drafts in
this workspace were trained with the OLMo3 target layout instead:

* full-projection Q/K RMSNorm,
* per-head attention sinks,
* OLMo post-attention/post-MLP RMSNorm residual order,
* a separately trained ``mask_embed`` tensor.

This module keeps vLLM's DFlash runtime contract, but swaps the draft backbone
to match those weights. Non-OLMo DFlash configs fall back to vLLM's stock Qwen3
implementation so registering ``DFlashDraftModel`` remains conservative.
"""

from __future__ import annotations

import io
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from torch import nn

from vllm import _custom_ops as ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3ForCausalLM
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    get_draft_quant_config,
    maybe_prefix,
    process_eagle_weight,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.transformers_utils.repo_utils import get_hf_file_bytes
from vllm.v1.attention.backend import AttentionType

from prime_rl.trainer.models.olmo3_sink.vllm_adapter import (
    _rope_parameters_for_layer,
)

logger = init_logger(__name__)


def _dflash_config(config) -> dict:
    raw = getattr(config, "dflash_config", None) or {}
    return dict(raw)


def _is_olmo3_sink_dflash(config) -> bool:
    dflash_config = _dflash_config(config)
    return bool(
        dflash_config.get("use_attention_sink")
        or getattr(config, "sink_init_value", None) is not None
    )


def _draft_sliding_window(config) -> int | None:
    dflash_config = _dflash_config(config)
    value = dflash_config.get("sliding_window", getattr(config, "sliding_window", None))
    if value is None:
        return None
    return int(value)


class DFlashOlmo3SinkMLP(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, config, prefix: str = ""):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=f"{prefix}.gate_up_proj",
        )
        self.act_fn = SiluAndMul()
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=get_draft_quant_config(vllm_config),
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class DFlashOlmo3SinkAttention(nn.Module):
    """OLMo3Sink draft attention for vLLM DFlash.

    vLLM's DFlash runtime pre-populates the draft KV cache from target hidden
    states. The normal forward path below only processes the query block.
    """

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        if tp_size != 1:
            raise NotImplementedError(
                "OLMo3Sink DFlash draft currently supports draft TP=1 only. "
                "Use policy TP=1 with data parallel rollout replicas."
            )

        self.layer_name = prefix
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads or self.total_num_heads
        self.num_kv_heads = self.total_num_kv_heads
        self.head_dim = getattr(config, "head_dim", None) or (
            self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.sliding_window = _draft_sliding_window(config)

        quant_config = get_draft_quant_config(vllm_config)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=getattr(config, "attention_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.q_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(
            self.total_num_kv_heads * self.head_dim,
            eps=config.rms_norm_eps,
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=_rope_parameters_for_layer(config, self.sliding_window),
        )
        init_value = float(getattr(config, "sink_init_value", -10.0))
        self.sinks = nn.Parameter(
            torch.full((self.num_heads,), init_value), requires_grad=False
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=vllm_config.cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=self.sliding_window,
            prefix=f"{prefix}.attn",
            attn_type=AttentionType.DECODER,
            sinks=self.sinks,
        )
        self.causal = bool(_dflash_config(config).get("causal", False))

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class DFlashOlmo3SinkDecoderLayer(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = DFlashOlmo3SinkAttention(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = DFlashOlmo3SinkMLP(
            vllm_config=vllm_config,
            config=config,
            prefix=f"{prefix}.mlp",
        )
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
        "input_embeds": 0,
    }
)
class DFlashOlmo3SinkModel(nn.Module):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.vocab_size = self.config.vocab_size
        self.quant_config = get_draft_quant_config(vllm_config)
        self.start_layer_id = start_layer_id

        dflash_config = _dflash_config(self.config)
        self.use_aux_hidden_state = bool(dflash_config.get("use_aux_hidden_state", True))
        self.mask_token_id = dflash_config.get(
            "mask_token_id", getattr(self.config, "mask_token_id", None)
        )
        self.mask_embedding = nn.Parameter(
            torch.zeros(self.config.hidden_size, dtype=vllm_config.model_config.dtype),
            requires_grad=False,
        )
        self.has_separate_mask_embedding = False

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )

        current_vllm_config = get_current_vllm_config()
        self.layers = nn.ModuleList(
            [
                DFlashOlmo3SinkDecoderLayer(
                    vllm_config=current_vllm_config,
                    config=self.config,
                    prefix=maybe_prefix(prefix, f"layers.{layer_idx + start_layer_id}"),
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )

        target_layer_ids = dflash_config.get("target_layer_ids") or []
        num_features_to_use = len(target_layer_ids) or self.config.num_hidden_layers
        target_hidden_size = getattr(
            self.config,
            "target_hidden_size",
            vllm_config.model_config.get_hidden_size(),
        )
        self.fc = ReplicatedLinear(
            input_size=target_hidden_size * num_features_to_use,
            output_size=self.config.hidden_size,
            bias=False,
            params_dtype=vllm_config.model_config.dtype,
            quant_config=self.quant_config,
            prefix=maybe_prefix(prefix, "fc"),
            return_bias=False,
        )
        self.hidden_norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )
        self.norm = RMSNorm(
            self.config.hidden_size,
            eps=self.config.rms_norm_eps,
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeds = self.embed_tokens(input_ids)
        if self.has_separate_mask_embedding and self.mask_token_id is not None:
            is_mask = (input_ids == self.mask_token_id).unsqueeze(-1)
            embeds = torch.where(is_mask, self.mask_embedding.to(embeds.dtype), embeds)
        return embeds

    def _build_context_kv_buffers(self) -> None:
        layers_attn = [layer.self_attn for layer in self.layers]
        attn0 = layers_attn[0]
        has_bias = attn0.qkv_proj.bias is not None

        self._hidden_norm_weight = self.hidden_norm.weight.data
        self._fused_kv_weight = torch.cat(
            [a.qkv_proj.weight[a.q_size :] for a in layers_attn], dim=0
        )
        if has_bias:
            self._fused_kv_bias: torch.Tensor | None = torch.cat(
                [a.qkv_proj.bias[a.q_size :] for a in layers_attn], dim=0
            )
        else:
            self._fused_kv_bias = None

        # OLMo3Sink K RMSNorm is over the full local KV projection, not per-head.
        self._k_norm_weights = torch.stack(
            [a.k_norm.weight.data for a in layers_attn], dim=0
        ).contiguous()

        self._rope_head_size = attn0.rotary_emb.head_size
        self._rope_cos_sin_cache = attn0.rotary_emb.cos_sin_cache
        self._rope_is_neox = attn0.rotary_emb.is_neox_style
        self._num_attn_layers = len(layers_attn)
        self._kv_size = attn0.kv_size
        self._head_dim = attn0.head_dim
        self._num_kv_heads = attn0.num_kv_heads
        self._rms_norm_eps = attn0.q_norm.variance_epsilon
        self._attn_layers = [layer.self_attn.attn for layer in self.layers]

        for attn in layers_attn[1:]:
            assert (
                attn.rotary_emb.head_size == self._rope_head_size
                and attn.rotary_emb.is_neox_style == self._rope_is_neox
            ), "All OLMo3Sink DFlash layers must share RoPE parameters"
            assert (
                attn.kv_size == self._kv_size
                and attn.head_dim == self._head_dim
                and attn.num_kv_heads == self._num_kv_heads
                and attn.q_norm.variance_epsilon == self._rms_norm_eps
            ), "All OLMo3Sink DFlash layers must share attention dimensions"

    def _project_context_kv(
        self,
        context_states: torch.Tensor,
        num_ctx: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed_context_states = torch.empty_like(context_states)
        ops.rms_norm(
            normed_context_states,
            context_states,
            self._hidden_norm_weight,
            self._rms_norm_eps,
        )
        all_kv_flat = F.linear(
            normed_context_states, self._fused_kv_weight, self._fused_kv_bias
        )
        all_kv = (
            all_kv_flat.view(num_ctx, num_layers, 2, num_kv_heads, head_dim)
            .permute(2, 1, 0, 3, 4)
            .contiguous()
        )
        return all_kv[0], all_kv[1]

    def _normalize_context_k(self, all_k: torch.Tensor) -> torch.Tensor:
        num_layers, num_ctx, num_kv_heads, head_dim = all_k.shape
        flat_k = all_k.reshape(num_layers, num_ctx, num_kv_heads * head_dim)
        flat_k_normed = torch.empty_like(flat_k)
        ops.rms_norm(
            flat_k_normed,
            flat_k,
            self._k_norm_weights,
            self._rms_norm_eps,
        )
        return flat_k_normed.view(num_layers, num_ctx, num_kv_heads, head_dim)

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        if not hasattr(self, "_num_attn_layers"):
            logger.warning_once(
                "OLMo3Sink DFlash buffers were not initialized before precompute; "
                "building them lazily."
            )
            self._build_context_kv_buffers()

        num_ctx = context_states.shape[0]
        all_k, all_v = self._project_context_kv(
            context_states,
            num_ctx,
            self._num_attn_layers,
            self._num_kv_heads,
            self._head_dim,
        )
        all_k_normed = self._normalize_context_k(all_k)

        all_k_flat = all_k_normed.reshape(
            self._num_attn_layers * num_ctx, self._kv_size
        )
        positions_repeated = context_positions.repeat(self._num_attn_layers)
        cos_sin_cache = self._rope_cos_sin_cache
        if cos_sin_cache.dtype != all_k_flat.dtype:
            cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
        ops.rotary_embedding(
            positions_repeated,
            all_k_flat,
            None,
            self._rope_head_size,
            cos_sin_cache,
            self._rope_is_neox,
        )

        if context_slot_mapping is None:
            return

        all_k_final = all_k_flat.view(
            self._num_attn_layers,
            num_ctx,
            self._num_kv_heads,
            self._head_dim,
        )
        per_layer = isinstance(context_slot_mapping, (list, tuple))
        for i in range(self._num_attn_layers):
            slot_mapping = context_slot_mapping[i] if per_layer else context_slot_mapping
            if slot_mapping is None:
                continue
            attn = self._attn_layers[i]
            attn.impl.do_kv_cache_update(
                attn,
                all_k_final[i],
                all_v[i],
                attn.kv_cache,
                slot_mapping,
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = input_embeds
        if hidden_states is None:
            hidden_states = self.embed_input_ids(input_ids)

        for layer in self.layers:
            hidden_states = layer(positions=positions, hidden_states=hidden_states)
        return self.norm(hidden_states)

    def _offset_layer_name(self, name: str) -> str:
        if self.start_layer_id <= 0 or not name.startswith("layers."):
            return name
        parts = name.split(".", 2)
        if len(parts) < 3 or not parts[1].isdigit():
            return name
        layer_idx = int(parts[1])
        offset_name = f"layers.{layer_idx + self.start_layer_id}.{parts[2]}"
        return offset_name

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()

        for name, loaded_weight in weights:
            name = self._offset_layer_name(name)
            if "scale" in name:
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
            if name.endswith(".self_attn.sinks"):
                if name not in params_dict:
                    continue
                param = params_dict[name]
                heads_per_rank = loaded_weight.shape[0] // tp_size
                shard = loaded_weight.narrow(0, tp_rank * heads_per_rank, heads_per_rank)
                param.data.copy_(shard.to(dtype=param.dtype))
                loaded_params.add(name)
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                param = params_dict[name]
                param.weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(name)
                break
            else:
                if name not in params_dict:
                    raise KeyError(f"Unexpected OLMo3Sink DFlash weight {name!r}")
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                loaded_params.add(name)

        return loaded_params


class DFlashOlmo3SinkForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash draft model registered as vLLM ``DFlashDraftModel``.

    The class subclasses vLLM's Qwen3 DFlash model so older vLLM proposer
    type-checks still accept it. It delegates to the stock implementation for
    non-OLMo DFlash configs.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        self._prime_olmo3_sink_dflash = _is_olmo3_sink_dflash(
            vllm_config.speculative_config.draft_model_config.hf_config
        )
        if not self._prime_olmo3_sink_dflash:
            super().__init__(vllm_config=vllm_config, prefix=prefix)
            return

        nn.Module.__init__(self)
        self.draft_model_config = vllm_config.speculative_config.draft_model_config
        self.config = self.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)
        self.has_own_embed_tokens = False
        self.has_own_lm_head = False

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.model = DFlashOlmo3SinkModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "model"),
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, inputs_embeds)

    def get_draft_kv_cache_layer_names(self) -> list[str]:
        return [layer.self_attn.attn.layer_name for layer in self.model.layers]

    def get_draft_attn_causal(self) -> list[bool]:
        return [layer.self_attn.causal for layer in self.model.layers]

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (logits.shape[0], self.config.vocab_size),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def precompute_and_store_context_kv(
        self,
        context_states: torch.Tensor,
        context_positions: torch.Tensor,
        context_slot_mapping: torch.Tensor | list[torch.Tensor | None] | None = None,
    ) -> None:
        self.model.precompute_and_store_context_kv(
            context_states, context_positions, context_slot_mapping
        )

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if not self.model.use_aux_hidden_state:
            return hidden_states
        needs_squeeze = hidden_states.dim() == 1
        if needs_squeeze:
            hidden_states = hidden_states.unsqueeze(0)
        result = self.model.fc(hidden_states)
        if needs_squeeze:
            result = result.squeeze(0)
        return result

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        if not getattr(self, "_prime_olmo3_sink_dflash", False):
            return super().load_weights(weights)

        model_weights = {}
        includes_draft_id_mapping = False
        includes_embed_tokens = False
        includes_lm_head = False
        for name, loaded_weight in weights:
            if name == "mask_embed":
                name = "model.mask_embedding"
                self.model.has_separate_mask_embedding = True
            elif "t2d" in name:
                continue
            elif "d2t" in name:
                name = name.replace("d2t", "draft_id_to_target_id")
                includes_draft_id_mapping = True
            elif name.startswith("model."):
                pass
            elif "lm_head" not in name:
                name = "model." + name
            if "embed_tokens" in name:
                includes_embed_tokens = True
                self.has_own_embed_tokens = True
            if "lm_head" in name:
                includes_lm_head = True
                self.has_own_lm_head = True
            model_weights[name] = loaded_weight
            process_eagle_weight(self, name)

        mask_embedding = self._read_mask_embedding()
        if mask_embedding is not None:
            model_weights["model.mask_embedding"] = mask_embedding
            self.model.has_separate_mask_embedding = True

        skip_substrs = []
        if not includes_draft_id_mapping:
            skip_substrs.append("draft_id_to_target_id")
        if not includes_embed_tokens:
            skip_substrs.append("embed_tokens")
        if not includes_lm_head:
            skip_substrs.append("lm_head")
        if not self.model.use_aux_hidden_state:
            skip_substrs.append("fc.")
        if not self.model.has_separate_mask_embedding:
            skip_substrs.append("mask_embedding")

        loader = AutoWeightsLoader(self, skip_substrs=skip_substrs)
        loaded = loader.load_weights(model_weights.items())
        self.model._build_context_kv_buffers()
        return loaded

    def _read_mask_embedding(self) -> torch.Tensor | None:
        mask_token_id = self.model.mask_token_id
        if mask_token_id is None:
            return None

        data = get_hf_file_bytes(
            "mask_embedding.pt",
            self.draft_model_config.model,
            self.draft_model_config.revision,
        )
        if data is None:
            return None

        state = torch.load(io.BytesIO(data), weights_only=True)
        if isinstance(state, dict):
            if state.get("mask_token_id", mask_token_id) != mask_token_id:
                raise ValueError(
                    "mask_embedding.pt mask_token_id does not match "
                    f"dflash_config.mask_token_id ({mask_token_id}). "
                    f"Got {state.get('mask_token_id')}."
                )
            state = state["embedding"]

        logger.info("Loaded OLMo3Sink DFlash mask embedding for token %s", mask_token_id)
        return state.reshape(-1)


# vLLM resolves the architecture string ``DFlashDraftModel``. Keep this alias
# explicit for lazy ModelRegistry registration.
DFlashDraftModel = DFlashOlmo3SinkForCausalLM
