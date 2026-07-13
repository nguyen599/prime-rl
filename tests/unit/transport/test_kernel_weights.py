import torch

from prime_rl.transport.kernel_weights import (
    has_kernel_weight_manifest,
    iter_kernel_weights,
    save_kernel_weight_manifest,
    save_kernel_weight_shard,
)


def test_kernel_weight_manifest_round_trip(tmp_path):
    shard_names = ["kernel-non-layer.safetensors", "kernel-layer-00000.safetensors"]
    expected = {
        "model.embed_tokens.weight": torch.arange(12, dtype=torch.bfloat16).reshape(3, 4),
        "model.layers.0.self_attn.qkv_proj.weight": torch.arange(16, dtype=torch.float32)
        .reshape(4, 4)
        .to(torch.float8_e4m3fn),
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": torch.ones(1, 1, dtype=torch.float32),
    }

    save_kernel_weight_shard(
        tmp_path,
        shard_names[0],
        {"model.embed_tokens.weight": expected["model.embed_tokens.weight"]},
    )
    save_kernel_weight_shard(
        tmp_path,
        shard_names[1],
        {name: tensor for name, tensor in expected.items() if name != "model.embed_tokens.weight"},
    )
    save_kernel_weight_manifest(tmp_path, shard_names, quantized_fp8=True)

    assert has_kernel_weight_manifest(tmp_path)
    actual = dict(iter_kernel_weights(tmp_path))
    assert actual.keys() == expected.keys()
    for name, tensor in expected.items():
        assert actual[name].dtype == tensor.dtype
        assert torch.equal(actual[name], tensor)
