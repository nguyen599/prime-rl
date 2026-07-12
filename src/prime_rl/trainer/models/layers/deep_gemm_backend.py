from __future__ import annotations

import importlib
from types import ModuleType


def load_deep_gemm() -> ModuleType | None:
    """Load DeepGEMM from its standalone wheel or vLLM's bundled copy."""
    for module_name in ("deep_gemm", "vllm.third_party.deep_gemm"):
        try:
            return importlib.import_module(module_name)
        except (ImportError, OSError):
            continue
    return None


deep_gemm = load_deep_gemm()


def require_deep_gemm() -> ModuleType:
    global deep_gemm
    if deep_gemm is None:
        deep_gemm = load_deep_gemm()
    if deep_gemm is None:
        raise RuntimeError(
            "FP8 training requires a working DeepGEMM backend. Install a DeepGEMM "
            "wheel compatible with this CUDA runtime or use a vLLM build that bundles "
            "vllm.third_party.deep_gemm."
        )
    return deep_gemm
