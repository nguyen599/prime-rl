from __future__ import annotations

import torch
from torch import Tensor

from prime_rl.trainer.models.fp8 import quantize_to_fp8_blockwise


def convert_layer_to_vllm_kernel(
    state_dict: dict[str, Tensor],
    layer_idx: int,
    quantize_fp8: bool = False,
) -> dict[str, Tensor]:
    """Convert one Olmo3Sink HF-format layer to the vLLM adapter kernel format.

    Olmo3Sink trains in standard HF parameter names, while the vLLM adapter packs
    q/k/v and gate/up projections. NCCL quantized weight transfer bypasses
    vLLM's normal weight loaders, so we emit the packed parameter names directly.
    """
    out: dict[str, Tensor] = {}
    prefix = f"model.layers.{layer_idx}"

    def add(name: str, tensor: Tensor) -> None:
        out[name] = tensor

    def add_maybe_fp8(name: str, tensor: Tensor) -> None:
        if quantize_fp8 and tensor.ndim == 2:
            fp8_weight, scale = quantize_to_fp8_blockwise(tensor)
            out[name] = fp8_weight
            out[name.removesuffix(".weight") + ".weight_scale_inv"] = scale
            return
        out[name] = tensor

    for suffix in [
        "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "self_attn.sinks",
    ]:
        key = f"{prefix}.{suffix}"
        if key in state_dict:
            add(key, state_dict[key])

    q_key = f"{prefix}.self_attn.q_proj.weight"
    k_key = f"{prefix}.self_attn.k_proj.weight"
    v_key = f"{prefix}.self_attn.v_proj.weight"
    if q_key in state_dict and k_key in state_dict and v_key in state_dict:
        add_maybe_fp8(
            f"{prefix}.self_attn.qkv_proj.weight",
            torch.cat([state_dict[q_key], state_dict[k_key], state_dict[v_key]], dim=0),
        )

    o_key = f"{prefix}.self_attn.o_proj.weight"
    if o_key in state_dict:
        add_maybe_fp8(f"{prefix}.self_attn.o_proj.weight", state_dict[o_key])

    gate_key = f"{prefix}.mlp.gate_proj.weight"
    up_key = f"{prefix}.mlp.up_proj.weight"
    if gate_key in state_dict and up_key in state_dict:
        add_maybe_fp8(
            f"{prefix}.mlp.gate_up_proj.weight",
            torch.cat([state_dict[gate_key], state_dict[up_key]], dim=0),
        )

    down_key = f"{prefix}.mlp.down_proj.weight"
    if down_key in state_dict:
        add_maybe_fp8(f"{prefix}.mlp.down_proj.weight", state_dict[down_key])

    return out
