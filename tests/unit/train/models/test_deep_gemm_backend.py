from __future__ import annotations

from types import ModuleType

from prime_rl.trainer.models.layers import deep_gemm_backend


def test_load_deep_gemm_falls_back_to_vllm(monkeypatch) -> None:
    bundled = ModuleType("vllm.third_party.deep_gemm")
    attempted: list[str] = []

    def fake_import(module_name: str) -> ModuleType:
        attempted.append(module_name)
        if module_name == "deep_gemm":
            raise OSError("standalone wheel targets another CUDA runtime")
        return bundled

    monkeypatch.setattr(deep_gemm_backend.importlib, "import_module", fake_import)

    assert deep_gemm_backend.load_deep_gemm() is bundled
    assert attempted == ["deep_gemm", "vllm.third_party.deep_gemm"]


def test_require_deep_gemm_reports_missing_backends(monkeypatch) -> None:
    monkeypatch.setattr(deep_gemm_backend, "deep_gemm", None)
    monkeypatch.setattr(deep_gemm_backend, "load_deep_gemm", lambda: None)

    try:
        deep_gemm_backend.require_deep_gemm()
    except RuntimeError as exc:
        assert "working DeepGEMM backend" in str(exc)
    else:
        raise AssertionError("missing DeepGEMM backends should fail explicitly")
