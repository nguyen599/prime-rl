import sys
from importlib import util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _load_magi_sink_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "src" / "prime_rl" / "trainer" / "models" / "olmo3_sink" / "magi_sink.py"
    spec = util.spec_from_file_location("_test_olmo3_magi_sink", module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("attn_impl", "function_name"),
    [
        ("olmo3_sink_fa2", "fa2_varlen_func_with_sink"),
        ("olmo3_sink_fa3", "fa3_varlen_func_with_sink"),
        ("olmo3_sink_fa4", "fa4_varlen_func_with_sink"),
    ],
)
def test_magi_backend_dispatch(monkeypatch, attn_impl, function_name):
    magi_sink = _load_magi_sink_module()
    flash_fn = MagicMock(return_value=torch.empty(4, 2, 8))
    fake_module = SimpleNamespace(**{function_name: flash_fn}, is_fa4_installed=True)
    monkeypatch.setattr(magi_sink, "import_module", lambda name: fake_module)

    q = torch.empty(4, 2, 8)
    cu = torch.tensor([0, 4], dtype=torch.int32)
    window = (-1, -1) if attn_impl == "olmo3_sink_fa4" else (31, 0)
    out = magi_sink.magi_varlen_attention_with_sink(
        q,
        q,
        q,
        torch.empty(2),
        cu,
        cu,
        4,
        4,
        attn_impl=attn_impl,
        softmax_scale=0.5,
        causal=True,
        window_size=window,
    )

    assert out.shape == q.shape
    assert flash_fn.call_args.kwargs["sink"].shape == (1, 2)
    assert flash_fn.call_args.kwargs["sink"].dtype == torch.float32
    assert flash_fn.call_args.kwargs["sink_layout"] == "sh"


def test_fa4_rejects_sliding_window(monkeypatch):
    magi_sink = _load_magi_sink_module()
    fake_module = SimpleNamespace(fa4_varlen_func_with_sink=MagicMock(), is_fa4_installed=True)
    monkeypatch.setattr(magi_sink, "import_module", lambda name: fake_module)
    q = torch.empty(4, 2, 8)
    cu = torch.tensor([0, 4], dtype=torch.int32)

    with pytest.raises(ValueError, match="does not support sliding-window"):
        magi_sink.magi_varlen_attention_with_sink(
            q,
            q,
            q,
            torch.empty(2),
            cu,
            cu,
            4,
            4,
            attn_impl="olmo3_sink_fa4",
            softmax_scale=None,
            causal=True,
            window_size=(31, 0),
        )
