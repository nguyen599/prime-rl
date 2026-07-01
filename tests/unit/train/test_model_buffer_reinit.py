from types import SimpleNamespace

import torch
from torch import nn

from prime_rl.trainer.model import can_reinit_empty_buffers, fix_model_post_empty


class _RotaryEmbedding(nn.Module):
    def __init__(self, *, extra_buffer: bool = False):
        super().__init__()
        self.config = SimpleNamespace()
        self.rope_type = "default"
        inv_freq, self.attention_scaling = self.compute_default_rope_parameters(self.config, None)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        if extra_buffer:
            self.register_buffer("attention_bias", torch.ones(1), persistent=False)

    def compute_default_rope_parameters(self, config, device):
        del config
        rotary_dim = 8
        inv_freq = 1.0 / (
            10000.0 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device) / rotary_dim)
        )
        return inv_freq, 1.0


class _Olmo3SinkLikeModel(nn.Module):
    def __init__(self, *, extra_buffer: bool = False):
        super().__init__()
        self.model = nn.Module()
        self.model.rotary_embs = nn.ModuleDict(
            {
                "sliding_attention": _RotaryEmbedding(),
                "full_attention": _RotaryEmbedding(extra_buffer=extra_buffer),
            }
        )


def test_per_layer_rotary_buffers_can_reinit_after_empty_load():
    with torch.device("meta"):
        model = _Olmo3SinkLikeModel()

    assert can_reinit_empty_buffers(model)

    model.to_empty(device="cpu")
    fix_model_post_empty(model)

    for rotary_emb in model.model.rotary_embs.values():
        expected_inv_freq, _ = rotary_emb.compute_default_rope_parameters(rotary_emb.config, torch.device("cpu"))
        assert rotary_emb.inv_freq.device.type == "cpu"
        assert rotary_emb.original_inv_freq.device.type == "cpu"
        torch.testing.assert_close(rotary_emb.inv_freq, expected_inv_freq)
        torch.testing.assert_close(rotary_emb.original_inv_freq, expected_inv_freq)
        assert rotary_emb.attention_scaling == 1.0


def test_per_layer_rotary_buffers_reject_extra_buffers():
    with torch.device("meta"):
        model = _Olmo3SinkLikeModel(extra_buffer=True)

    assert not can_reinit_empty_buffers(model)
