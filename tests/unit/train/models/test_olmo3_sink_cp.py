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
    """Ulysses CP must wrap Olmo3Sink's custom attention key, not only FA2."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    ulysses_attn = _load_ulysses_attn_module()

    fake_flash_attn = types.SimpleNamespace(flash_attn_varlen_func=object())
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash_attn)
    monkeypatch.setattr(ulysses_attn.dist, "get_world_size", lambda group=None: 2)

    original = ALL_ATTENTION_FUNCTIONS.get("olmo3_sink_fa3")
    try:
        ulysses_attn.substitute_hf_ulysses_attn(MagicMock())
        assert ALL_ATTENTION_FUNCTIONS["olmo3_sink_fa3"] is ulysses_attn.ulysses_olmo3_sink_fa3_attention_forward
    finally:
        if original is None:
            ALL_ATTENTION_FUNCTIONS.pop("olmo3_sink_fa3", None)
        else:
            ALL_ATTENTION_FUNCTIONS["olmo3_sink_fa3"] = original
