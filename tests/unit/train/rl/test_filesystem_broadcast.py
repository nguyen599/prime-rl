import torch.nn as nn

from prime_rl.trainer.rl.broadcast.filesystem import _supports_kernel_weight_conversion


def test_kernel_weight_conversion_uses_model_capability() -> None:
    class DuckTypedCustomModel(nn.Module):
        @classmethod
        def convert_layer_to_vllm_kernel(cls, state_dict, layer_idx, quantize_fp8=False):
            return state_dict

    assert _supports_kernel_weight_conversion(DuckTypedCustomModel())
    assert not _supports_kernel_weight_conversion(nn.Linear(2, 2))
