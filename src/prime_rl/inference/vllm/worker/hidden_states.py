import base64
from typing import Any

import torch
from torch.nn import Module


class HiddenStateScoringMixin:
    """Worker RPC for full-vocab OPD teacher hidden-state scoring.

    vLLM tensor-parallel layers all-reduce row-parallel outputs, so the final
    hidden states returned by the model runner are expected to be replicated on
    each TP rank. The API server picks the first non-null worker response.
    """

    @torch.no_grad()
    def prefill_hidden_states(self, token_ids: list[int], dtype: str = "float16") -> dict[str, Any]:
        model_runner = self.model_runner
        if hasattr(model_runner.model, "runnable"):
            model = model_runner.model.runnable
        else:
            model = model_runner.model
        assert isinstance(model, Module)

        device = self.device
        input_ids = torch.tensor(token_ids, dtype=torch.long, device=device)
        positions = torch.arange(len(token_ids), dtype=torch.long, device=device)
        hidden_states = model(input_ids=input_ids, positions=positions)
        if not isinstance(hidden_states, torch.Tensor):
            raise RuntimeError(f"expected model forward to return hidden states tensor, got {type(hidden_states)!r}")
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        if hidden_states.dim() != 2:
            raise RuntimeError(f"expected hidden states [seq, hidden], got {tuple(hidden_states.shape)}")

        target_dtype = getattr(torch, dtype)
        hidden_cpu = hidden_states.to(dtype=target_dtype).cpu().contiguous()
        raw = hidden_cpu.view(torch.uint8).numpy().tobytes()
        return {
            "dtype": dtype,
            "shape": list(hidden_cpu.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }
