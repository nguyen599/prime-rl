import sys
import types
from importlib import util
from pathlib import Path
from unittest.mock import MagicMock


def _load_ulysses_attn_module():
    repo_root = Path(__file__).resolve().parents[4]
    module_path = repo_root / "src" / "prime_rl" / "trainer" / "models" / "layers" / "ulysses_attn.py"
    spec = util.spec_from_file_location("_test_ulysses_attn", module_path)
    assert spec is not None and spec.loader is not None
    module = util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_olmo3_sink_ulysses_registration(monkeypatch):
    """Ulysses CP must wrap every Olmo3Sink custom attention key."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ulysses_attn = _load_ulysses_attn_module()

    fake_flash_attn = types.SimpleNamespace(flash_attn_varlen_func=object())
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash_attn)
    fake_sink_attention_module = types.SimpleNamespace()
    fake_sink_modeling_module = types.SimpleNamespace(Olmo3SinkAttention=type("Olmo3SinkAttention", (), {}))
    monkeypatch.setitem(
        sys.modules,
        "prime_rl.trainer.models.olmo3_sink",
        types.SimpleNamespace(attention=fake_sink_attention_module),
    )
    monkeypatch.setitem(
        sys.modules,
        "prime_rl.trainer.models.olmo3_sink.attention",
        fake_sink_attention_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "prime_rl.trainer.models.olmo3_sink.modeling_olmo3_sink",
        fake_sink_modeling_module,
    )
    monkeypatch.setattr(ulysses_attn.dist, "get_world_size", lambda group=None: 2)

    sink_impls = ("olmo3_sink_fa2", "olmo3_sink_fa3", "olmo3_sink_fa4")
    originals = {name: ALL_ATTENTION_FUNCTIONS.get(name) for name in sink_impls}
    try:
        ulysses_attn.substitute_hf_ulysses_attn(MagicMock())
        for name in sink_impls:
            assert ALL_ATTENTION_FUNCTIONS[name] is ulysses_attn.ulysses_olmo3_sink_attention_forward
    finally:
        for name, original in originals.items():
            if original is None:
                ALL_ATTENTION_FUNCTIONS.pop(name, None)
            else:
                ALL_ATTENTION_FUNCTIONS[name] = original
