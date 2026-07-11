import base64
import functools
import os
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import Module

from prime_rl.transport.hidden_state_files import sweep_tensor_files, write_tensor_chunks_file

_CAPTURE_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class HiddenStateScoringMixin:
    """Worker RPC for full-vocab OPD teacher hidden-state scoring.

    vLLM tensor-parallel layers all-reduce row-parallel outputs, so the final
    hidden states returned by the model runner are expected to be replicated on
    each TP rank. The API server picks the first non-null worker response.
    """

    def _prime_hidden_capture_is_primary_worker(self) -> bool:
        try:
            from vllm.distributed import get_tp_group

            return int(get_tp_group().rank_in_group) == 0
        except (AssertionError, RuntimeError):
            # Unit tests and older worker boot paths can call this before the
            # vLLM tensor-parallel group has been initialized.
            pass
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

    def _prime_hidden_capture_expected_width(self) -> int:
        """Return the width consumed by the model's output projection."""
        runner = self.model_runner
        model = getattr(runner, "model", None)
        if hasattr(model, "runnable"):
            model = model.runnable

        head_width = None
        lm_head = getattr(model, "lm_head", None)
        head_weight = getattr(lm_head, "weight", None)
        head_shape = getattr(head_weight, "shape", None)
        if head_shape is not None and len(head_shape) == 2:
            head_width = int(head_shape[-1])

        model_config = getattr(runner, "model_config", None)
        hf_config = getattr(model_config, "hf_config", None)
        config_width = getattr(hf_config, "hidden_size", None)
        if config_width is not None:
            config_width = int(config_width)

        if head_width is not None and config_width is not None and head_width != config_width:
            raise RuntimeError(
                "hidden-state capture found inconsistent LM-head/config widths: "
                f"lm_head={head_width}, config.hidden_size={config_width}"
            )
        expected_width = head_width if head_width is not None else config_width
        if expected_width is None or expected_width <= 0:
            raise RuntimeError("hidden-state capture could not determine the model LM-head input width")
        return expected_width

    @staticmethod
    def _prime_hidden_capture_bindings(
        captures: dict[str, dict[str, Any]], runner: Any, num_scheduled_tokens: Any
    ) -> dict[str, dict[str, Any]]:
        """Bind API capture IDs to model-runner IDs without guessing by order."""
        req_ids = list(getattr(runner.input_batch, "req_ids", []))
        bindings: dict[str, dict[str, Any]] = {}

        for capture_id, capture in captures.items():
            bound_req_id = capture.get("runner_req_id")
            if bound_req_id is None and capture_id in req_ids:
                bound_req_id = capture_id
                capture["runner_req_id"] = capture_id
            if bound_req_id in req_ids:
                bindings[str(bound_req_id)] = capture

        # A few vLLM adapters rewrite request IDs. Bind an unbound capture only
        # when prompt length identifies exactly one scheduled runner request.
        for capture in captures.values():
            if capture.get("runner_req_id") is not None:
                continue
            candidates: list[str] = []
            for req_id in req_ids:
                if req_id in bindings:
                    continue
                if not isinstance(num_scheduled_tokens, dict) or int(num_scheduled_tokens.get(req_id, 0)) <= 0:
                    continue
                request_state = runner.requests.get(req_id)
                prompt_token_ids = getattr(request_state, "prompt_token_ids", None)
                if prompt_token_ids is not None and len(prompt_token_ids) == int(capture["target_len"]):
                    candidates.append(req_id)
            if len(candidates) > 1:
                raise RuntimeError(
                    "hidden-state capture cannot safely bind a rewritten request id: "
                    f"{len(candidates)} scheduled requests have target length {int(capture['target_len'])}"
                )
            if len(candidates) == 1:
                bound_req_id = candidates[0]
                capture["runner_req_id"] = bound_req_id
                bindings[bound_req_id] = capture

        return bindings

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
                bindings = worker._prime_hidden_capture_bindings(captures, runner, num_scheduled_tokens)
                completed_capture_req_ids: set[str] = set()
                handled_capture_req_ids: set[str] = set()
                for req_id, capture in bindings.items():
                    capture.setdefault("seen_req_ids", set()).add(str(req_id))
                    num_tokens = num_scheduled_tokens.get(req_id) if isinstance(num_scheduled_tokens, dict) else None
                    if num_tokens is None:
                        continue
                    num_tokens = int(num_tokens)
                    if num_tokens <= 0:
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
                    if hidden_states.dim() != 2:
                        raise RuntimeError(
                            "hidden-state capture expected model-runner output [tokens, hidden], "
                            f"got {tuple(hidden_states.shape)}"
                        )
                    expected_width = int(capture["expected_width"])
                    if int(hidden_states.shape[-1]) != expected_width:
                        raise RuntimeError(
                            "hidden-state capture received the wrong LM-head input width: "
                            f"got {int(hidden_states.shape[-1])}, expected {expected_width}. "
                            "For DeepSeek-V4 this usually means a pre-hc_head/MTP tensor was captured."
                        )
                    offset_value = runner.query_start_loc.np[req_idx]
                    offset = int(offset_value.item() if hasattr(offset_value, "item") else offset_value)
                    if offset < 0 or offset + copy_len > int(hidden_states.shape[0]):
                        raise RuntimeError(
                            f"hidden-state capture slice [{offset}:{offset + copy_len}] exceeds "
                            f"model-runner output with {int(hidden_states.shape[0])} rows"
                        )
                    if worker._prime_hidden_capture_is_primary_worker():
                        target_dtype = _CAPTURE_DTYPES[capture["dtype"]]
                        chunk = (
                            hidden_states[offset : offset + copy_len]
                            .detach()
                            .to(dtype=target_dtype, device="cpu")
                            .contiguous()
                        )
                        capture["chunks"][start_pos] = chunk
                    handled_capture_req_ids.add(req_id)
                    if start_pos + copy_len >= target_len:
                        completed_capture_req_ids.add(req_id)

                # These requests use prompt_logprobs only as a stable hook into
                # vLLM's normal prefill path. Never fall through to the real
                # prompt-logprob path while a hidden-state capture is installed:
                # it would materialize huge logits and, for DeepSeek V4, can call
                # compute_logits outside the forward context required by MLA.
                num_prompt_logprobs = getattr(runner, "num_prompt_logprobs", None)
                removed_prompt_logprobs: dict[str, Any] = {}
                if isinstance(num_prompt_logprobs, dict):
                    for req_id in handled_capture_req_ids:
                        if req_id in num_prompt_logprobs:
                            removed_prompt_logprobs[req_id] = num_prompt_logprobs.pop(req_id)
                try:
                    # Preserve normal prompt-logprob behavior for unrelated
                    # requests that happen to share this scheduler batch.
                    result = original(hidden_states, num_scheduled_tokens, *args, **kwargs)
                finally:
                    if isinstance(num_prompt_logprobs, dict):
                        for req_id, value in removed_prompt_logprobs.items():
                            if req_id not in completed_capture_req_ids:
                                num_prompt_logprobs[req_id] = value
                        for req_id in completed_capture_req_ids:
                            request_state = runner.requests.get(req_id)
                            if request_state is not None:
                                request_state.in_progress_prompt_logprobs_cpu = None
                return result

            return original(hidden_states, num_scheduled_tokens, *args, **kwargs)

        model_runner._get_prompt_logprobs_dict = wrapped_get_prompt_logprobs_dict
        model_runner._prime_hidden_capture_hook_installed = True

    def prepare_hidden_state_capture(self, request_id: str, target_len: int, dtype: str = "bfloat16") -> None:
        """Prepare the worker to capture hidden states from a normal vLLM prefill.

        The API server sends an internal generate request with ``prompt_logprobs``
        enabled. vLLM then executes the real scheduler/model-runner path with
        valid attention metadata. The hook above copies each prefill chunk's
        hidden states as the prompt-logprob path sees them.
        """
        target_len = int(target_len)
        if target_len <= 0:
            raise ValueError(f"hidden-state capture target_len must be positive, got {target_len}")
        if dtype not in _CAPTURE_DTYPES:
            raise ValueError(
                f"unsupported hidden-state capture dtype {dtype!r}; expected one of {sorted(_CAPTURE_DTYPES)}"
            )
        self._prime_hidden_capture_install_hook()
        state = self._prime_hidden_capture_state()
        if request_id in state:
            raise RuntimeError(f"hidden-state capture {request_id!r} is already active")
        state[request_id] = {
            "target_len": target_len,
            "dtype": dtype,
            "expected_width": self._prime_hidden_capture_expected_width(),
            "chunks": {},
            "seen_req_ids": set(),
            "runner_req_id": None,
        }

    def discard_hidden_state_capture(self, request_id: str) -> bool:
        """Drop capture state after a failed or cancelled generation request."""
        return self._prime_hidden_capture_state().pop(request_id, None) is not None

    def _prime_maybe_sweep_hidden_files(self, directory: Path) -> None:
        interval = float(os.environ.get("PRIME_RL_HIDDEN_STATE_SWEEP_INTERVAL_SECONDS", "600"))
        now = time.monotonic()
        last = float(getattr(self, "_prime_hidden_state_last_sweep", 0.0))
        if now - last < interval:
            return
        self._prime_hidden_state_last_sweep = now
        ttl = float(os.environ.get("PRIME_RL_HIDDEN_STATE_TTL_SECONDS", "21600"))
        sweep_tensor_files(directory, ttl)

    def pop_hidden_state_capture(self, request_id: str, output_path: str | None = None) -> dict[str, Any] | None:
        capture = self._prime_hidden_capture_state().pop(request_id, None)
        if capture is None:
            return None
        if not self._prime_hidden_capture_is_primary_worker():
            return None

        target_len = int(capture["target_len"])
        chunks: dict[int, torch.Tensor] = capture["chunks"]
        if not chunks:
            seen_req_ids = sorted(str(req_id) for req_id in capture.get("seen_req_ids", []))
            raise RuntimeError(
                f"no hidden-state chunks captured for request {request_id!r}; seen_model_runner_req_ids={seen_req_ids}"
            )

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

        if output_path is not None:
            ref = write_tensor_chunks_file(output_path, ordered)
            self._prime_maybe_sweep_hidden_files(Path(output_path).parent)
            return {
                "transport": "filesystem",
                "path": ref.path,
                "dtype": ref.dtype,
                "shape": ref.shape,
                "offset": ref.offset,
                "nbytes": ref.nbytes,
            }

        hidden_cpu = torch.cat(ordered, dim=0) if len(ordered) > 1 else ordered[0]
        raw = hidden_cpu.view(torch.uint8).numpy().tobytes()
        return {
            "dtype": str(capture["dtype"]),
            "shape": list(hidden_cpu.shape),
            "data": base64.b64encode(raw).decode("ascii"),
        }

    @torch.no_grad()
    def prefill_hidden_states(self, token_ids: list[int], dtype: str = "bfloat16") -> dict[str, Any]:
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
