import torch.nn as nn

from prime_rl.trainer.rl.broadcast.filesystem import _clean_applied_broadcasts, _supports_kernel_weight_conversion
from prime_rl.utils.pathing import WEIGHT_APPLIED_MARKER


def test_kernel_weight_conversion_uses_model_capability() -> None:
    class DuckTypedCustomModel(nn.Module):
        @classmethod
        def convert_layer_to_vllm_kernel(cls, state_dict, layer_idx, quantize_fp8=False):
            return state_dict

    assert _supports_kernel_weight_conversion(DuckTypedCustomModel())
    assert not _supports_kernel_weight_conversion(nn.Linear(2, 2))


def test_cleanup_keeps_unapplied_broadcasts(tmp_path) -> None:
    broadcast_dir = tmp_path / "broadcasts"
    unapplied = broadcast_dir / "step_1"
    applied = broadcast_dir / "step_2"
    current = broadcast_dir / "step_3"
    for path in (unapplied, applied, current):
        path.mkdir(parents=True)
    (applied / WEIGHT_APPLIED_MARKER).touch()

    _clean_applied_broadcasts(broadcast_dir, current_step=3, interval_to_keep=None)

    assert unapplied.exists()
    assert not applied.exists()
    assert current.exists()


def test_cleanup_preserves_checkpoint_interval(tmp_path) -> None:
    broadcast_dir = tmp_path / "broadcasts"
    checkpoint = broadcast_dir / "step_10"
    checkpoint.mkdir(parents=True)
    (checkpoint / WEIGHT_APPLIED_MARKER).touch()

    _clean_applied_broadcasts(broadcast_dir, current_step=11, interval_to_keep=10)

    assert checkpoint.exists()
