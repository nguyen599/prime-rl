import torch

from prime_rl.trainer.models.olmo3_sink.converting_olmo3_sink import convert_layer_to_vllm_kernel


def _state_dict() -> dict[str, torch.Tensor]:
    prefix = "model.layers.0"
    return {
        f"{prefix}.self_attn.q_proj.weight": torch.arange(32, dtype=torch.bfloat16).reshape(8, 4),
        f"{prefix}.self_attn.k_proj.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4),
        f"{prefix}.self_attn.v_proj.weight": torch.arange(16, dtype=torch.bfloat16).reshape(4, 4) + 16,
        f"{prefix}.self_attn.o_proj.weight": torch.arange(32, dtype=torch.bfloat16).reshape(4, 8),
        f"{prefix}.mlp.gate_proj.weight": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4),
        f"{prefix}.mlp.up_proj.weight": torch.arange(24, dtype=torch.bfloat16).reshape(6, 4) + 24,
        f"{prefix}.mlp.down_proj.weight": torch.arange(24, dtype=torch.bfloat16).reshape(4, 6),
        f"{prefix}.post_attention_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
    }


def test_quantized_olmo3_sink_conversion_matches_vllm_online_fp8_kernel_layout() -> None:
    state = _state_dict()
    converted = convert_layer_to_vllm_kernel(state, layer_idx=0, quantize_fp8=True)
    prefix = "model.layers.0"

    expected_hf_weights = {
        f"{prefix}.self_attn.qkv_proj.weight": torch.cat(
            [
                state[f"{prefix}.self_attn.q_proj.weight"],
                state[f"{prefix}.self_attn.k_proj.weight"],
                state[f"{prefix}.self_attn.v_proj.weight"],
            ],
            dim=0,
        ),
        f"{prefix}.self_attn.o_proj.weight": state[f"{prefix}.self_attn.o_proj.weight"],
        f"{prefix}.mlp.gate_up_proj.weight": torch.cat(
            [state[f"{prefix}.mlp.gate_proj.weight"], state[f"{prefix}.mlp.up_proj.weight"]], dim=0
        ),
        f"{prefix}.mlp.down_proj.weight": state[f"{prefix}.mlp.down_proj.weight"],
    }

    for name, expected_hf_weight in expected_hf_weights.items():
        kernel_weight = converted[name]
        scale = converted[name.removesuffix(".weight") + ".weight_scale"]

        assert kernel_weight.shape == expected_hf_weight.transpose(0, 1).shape
        assert kernel_weight.dtype == torch.float8_e4m3fn
        assert scale.shape == (1,)
        dequantized_hf_weight = kernel_weight.float().transpose(0, 1) * scale
        torch.testing.assert_close(
            dequantized_hf_weight,
            expected_hf_weight.float(),
            atol=float(scale.item()),
            rtol=0,
        )

    assert not any(name.endswith(".weight_scale_inv") for name in converted)
    assert converted[f"{prefix}.post_attention_layernorm.weight"].dtype == torch.bfloat16


def test_unquantized_olmo3_sink_conversion_preserves_hf_weight_orientation() -> None:
    state = _state_dict()
    converted = convert_layer_to_vllm_kernel(state, layer_idx=0, quantize_fp8=False)
    prefix = "model.layers.0"
    expected_qkv = torch.cat(
        [
            state[f"{prefix}.self_attn.q_proj.weight"],
            state[f"{prefix}.self_attn.k_proj.weight"],
            state[f"{prefix}.self_attn.v_proj.weight"],
        ],
        dim=0,
    )

    torch.testing.assert_close(converted[f"{prefix}.self_attn.qkv_proj.weight"], expected_qkv)
    assert not any("weight_scale" in name for name in converted)
