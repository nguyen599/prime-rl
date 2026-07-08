import base64
import functools
from typing import Any

import torch
from torch.nn import Module


class HiddenStateScoringMixin:
    """Worker RPC for full-vocab OPD teacher hidden-state scoring.

    vLLM tensor-parallel layers all-reduce row-parallel outputs, so the final
    hidden states returned by the model runner are expected to be replicated on
    each TP rank. The API server picks the first non-null worker response.
    """

    def _prime_hidden_capture_is_primary_worker(self) -> bool:
        device = getattr(self, "device", None)
        device_index = getattr(device, "index", None)
        if device_index is not None:
            return int(device_index) == 0
        local_rank = getattr(self, "local_rank", None)
        if local_rank is not None:
            return int(local_rank) == 0
        rank = getattr(self, "rank", None)
        if rank is not None:
            return int(rank) == 0
        return True

    def _prime_hidden_capture_state(self) -> dict[str, dict[str, Any]]:
        state = getattr(self, "_prime_hidden_state_captures", None)
        if state is None:
            state = {}
            setattr(self, "_prime_hidden_state_captures", state)
        return state

    def _prime_hidden_capture_install_hook(self) -> None:
        model_runner = self.model_runner
        if getattr(model_runner, "_prime_hidden_capture_hook_installed", False):
            return

        if not hasattr(model_runner, "_get_prompt_logprobs_dict"):
            raise RuntimeError(
                "full-vocab hidden-state capture requires a vLLM model runner "
                "with _get_prompt_logprobs_dict; this vLLM version does not expose it"
            )

        original = model_runner._get_prompt_logprobs_dict
        worker = self

        @functools.wraps(original)
        def wrapped_get_prompt_logprobs_dict(hidden_states, num_scheduled_tokens, *args, **kwargs):
            captures = getattr(worker, "_prime_hidden_state_captures", {})
            if captures:
                runner = worker.model_runner
                captured_any = False
                for req_id in list(getattr(runner.input_batch, "req_ids", [])):
                    capture = captures.get(req_id)
                    if capture is None:
                        continue
                    num_tokens = num_scheduled_tokens.get(req_id) if isinstance(num_scheduled_tokens, dict) else None
                    if not num_tokens:
                        continue
                    req_idx = runner.input_batch.req_id_to_index.get(req_id)
                    if req_idx is None:
                        continue
                    request_state = runner.requests.get(req_id)
                    if request_state is None:
                        continue
                    start_pos = int(request_state.num_computed_tokens)
                    target_len = int(capture["target_len"])
                    if start_pos >= target_len:
                        continue
                    copy_len = min(int(num_tokens), target_len - start_pos)
                    if copy_len <= 0:
                        continue
                    offset = int(runner.query_start_loc.np[req_idx])
                    target_dtype = getattr(torch, capture["dtype"])
                    chunk = hidden_states[offset : offset + copy_len].detach().to(dtype=target_dtype).cpu().contiguous()
                    capture["chunks"][start_pos] = chunk
                    captured_any = True

                if captured_any:
                    # This request uses prompt_logprobs only as a stable hook into
                    # vLLM's normal prefill path. Returning an empty dict avoids
                    # materializing real prompt-logprob tensors for 80k-token
                    # teacher sequences, which would be wasteful and can OOM.
                    return {}

            return original(hidden_states, num_scheduled_tokens, *args, **kwargs)

        model_runner._get_prompt_logprobs_dict = wrapped_get_prompt_logprobs_dict
        model_runner._prime_hidden_capture_hook_installed = True

    def prepare_hidden_state_capture(self, request_id: str, target_len: int, dtype: str = "float16") -> None:
        """Prepare the worker to capture hidden states from a normal vLLM prefill.

        The API server sends an internal generate request with ``prompt_logprobs``
        enabled. vLLM then executes the real scheduler/model-runner path with
        valid attention metadata. The hook above copies each prefill chunk's
        hidden states as the prompt-logprob path sees them.
        """
        self._prime_hidden_capture_install_hook()
        self._prime_hidden_capture_state()[request_id] = {
            "target_len": int(target_len),
            "dtype": dtype,
            "chunks": {},
        }

    def pop_hidden_state_capture(self, request_id: str) -> dict[str, Any] | None:
        capture = self._prime_hidden_capture_state().pop(request_id, None)
        if capture is None:
            return None
        if not self._prime_hidden_capture_is_primary_worker():
            return None

        target_len = int(capture["target_len"])
        chunks: dict[int, torch.Tensor] = capture["chunks"]
        if not chunks:
            raise RuntimeError(f"no hidden-state chunks captured for request {request_id!r}")

        ordered: list[torch.Tensor] = []
        expected = 0
        for start, chunk in sorted(chunks.items()):
            if start != expected:
                raise RuntimeError(
                    f"hidden-state capture for {request_id!r} has a gap: expected chunk at {expected}, got {start}"
                )
            ordered.append(chunk)
            expected += int(chunk.shape[0])
        if expected != target_len:
            raise RuntimeError(
                f"hidden-state capture for {request_id!r} incomplete: captured {expected} / {target_len} rows"
            )

        hidden_cpu = torch.cat(ordered, dim=0) if len(ordered) > 1 else ordered[0]
        raw = hidden_cpu.view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(capture["dtype"]),
            "shape": list(hidden_cpu.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }

    @torch.no_grad()
    def prefill_hidden_states(self, token_ids: list[int], dtype: str = "float16") -> dict[str, Any]:
        """Legacy direct forward path.

        Kept for older non-DeepSeek backends, but the API route now drives
        hidden-state capture through vLLM's normal prefill path so attention
        metadata and KV-cache state are valid.
        """
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
