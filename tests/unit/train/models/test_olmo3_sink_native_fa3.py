import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from prime_rl.configs.trainer import ModelConfig


def _load_native_fa3_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = (
        repo_root
        / "src"
        / "prime_rl"
        / "trainer"
        / "models"
        / "olmo3_sink"
        / "native_fa3_sink.py"
    )
    spec = util.spec_from_file_location("_test_olmo3_native_fa3_sink", module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_native_fa3_dispatch_stays_separate_from_magi(monkeypatch):
    native_fa3 = _load_native_fa3_module()
    output = torch.empty(4, 2, 8)
    kernel = MagicMock(return_value=output)
    fake_module = SimpleNamespace(fa3_varlen_attn_with_sink_kernel=kernel)
    monkeypatch.setattr(native_fa3, "import_module", lambda name: fake_module)

    q = torch.empty_like(output)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    actual = native_fa3.native_fa3_varlen_attention_with_sink(
        q,
        q,
        q,
        torch.empty(2),
        cu,
        cu,
        4,
        4,
        softmax_scale=0.5,
        causal=True,
        window_size=(31, 0),
    )

    assert actual is output
    assert kernel.call_count == 1
    assert kernel.call_args.kwargs == {
        "softmax_scale": 0.5,
        "causal": True,
        "window_size": (31, 0),
    }


def test_native_fa3_is_valid_for_olmo3_sink_ulysses():
    config = ModelConfig(
        name="unused",
        impl="custom",
        attn="olmo3_sink_fa3_native",
        cp=2,
        cp_style="ulysses",
    )

    assert config.attn == "olmo3_sink_fa3_native"
