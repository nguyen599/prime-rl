import asyncio
import base64
from types import SimpleNamespace

import pytest
import torch

from prime_rl.inference.vllm.worker.hidden_states import HiddenStateScoringMixin


class _FakeModel(torch.nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.lm_head = torch.nn.Linear(hidden_size, 7, bias=False)


class _FakeRunner:
    def __init__(self, hidden_size: int = 4):
        self.model = _FakeModel(hidden_size)
        self.model_config = SimpleNamespace(hf_config=SimpleNamespace(hidden_size=hidden_size))
        self.input_batch = SimpleNamespace(req_ids=[], req_id_to_index={})
        self.query_start_loc = SimpleNamespace(np=torch.tensor([], dtype=torch.int64))
        self.requests = {}
        self.num_prompt_logprobs = {}
        self.original_calls = []

    def _get_prompt_logprobs_dict(self, hidden_states, num_scheduled_tokens):
        self.original_calls.append(set(self.num_prompt_logprobs))
        return {req_id: "original" for req_id in self.num_prompt_logprobs}


class _FakeWorker(HiddenStateScoringMixin):
    def __init__(self, hidden_size: int = 4, local_rank: int = 0):
        self.model_runner = _FakeRunner(hidden_size)
        self.device = torch.device("cpu")
        self.local_rank = local_rank


def _request(token_count: int, computed: int = 0):
    return SimpleNamespace(
        prompt_token_ids=list(range(token_count)),
        num_computed_tokens=computed,
        in_progress_prompt_logprobs_cpu=object(),
    )


def _decode_inline(result: dict) -> torch.Tensor:
    data = bytearray(base64.b64decode(result["data"]))
    return torch.frombuffer(data, dtype=getattr(torch, result["dtype"])).reshape(result["shape"]).clone()


def test_capture_uses_vllm_chunk_offsets_and_preserves_unrelated_prompt_logprobs():
    worker = _FakeWorker()
    runner = worker.model_runner
    worker.prepare_hidden_state_capture("capture", target_len=5)

    runner.input_batch = SimpleNamespace(
        req_ids=["unrelated", "capture"],
        req_id_to_index={"unrelated": 0, "capture": 1},
    )
    runner.query_start_loc = SimpleNamespace(np=torch.tensor([0, 2]))
    runner.requests = {"unrelated": _request(2), "capture": _request(5)}
    runner.num_prompt_logprobs = {"unrelated": 1, "capture": 1}
    hidden = torch.tensor(
        [
            [90, 90, 90, 90],
            [91, 91, 91, 91],
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [8, 9, 10, 11],
        ],
        dtype=torch.bfloat16,
    )

    result = runner._get_prompt_logprobs_dict(hidden, {"unrelated": 2, "capture": 3})

    assert result == {"unrelated": "original"}
    assert runner.original_calls == [{"unrelated"}]
    assert runner.num_prompt_logprobs == {"unrelated": 1, "capture": 1}

    runner.input_batch = SimpleNamespace(req_ids=["capture"], req_id_to_index={"capture": 0})
    runner.query_start_loc = SimpleNamespace(np=torch.tensor([0]))
    runner.requests["capture"].num_computed_tokens = 3
    hidden_tail = torch.tensor([[12, 13, 14, 15], [16, 17, 18, 19]], dtype=torch.bfloat16)
    runner._get_prompt_logprobs_dict(hidden_tail, {"capture": 2})

    assert "capture" not in runner.num_prompt_logprobs
    restored = _decode_inline(worker.pop_hidden_state_capture("capture"))
    torch.testing.assert_close(restored, torch.cat([hidden[2:], hidden_tail]))


def test_rewritten_request_id_binds_only_by_unique_prompt_length():
    worker = _FakeWorker()
    runner = worker.model_runner
    worker.prepare_hidden_state_capture("api-request", target_len=3)
    runner.input_batch = SimpleNamespace(
        req_ids=["decoy", "internal-request"],
        req_id_to_index={"decoy": 0, "internal-request": 1},
    )
    runner.query_start_loc = SimpleNamespace(np=torch.tensor([0, 2]))
    runner.requests = {"decoy": _request(2), "internal-request": _request(3)}
    runner.num_prompt_logprobs = {"internal-request": 1}
    hidden = torch.tensor(
        [[90, 90, 90, 90], [91, 91, 91, 91], [0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        dtype=torch.bfloat16,
    )

    runner._get_prompt_logprobs_dict(hidden, {"decoy": 2, "internal-request": 3})

    restored = _decode_inline(worker.pop_hidden_state_capture("api-request"))
    torch.testing.assert_close(restored, hidden[2:])


def test_rewritten_request_id_fails_when_prompt_length_is_ambiguous():
    worker = _FakeWorker()
    runner = worker.model_runner
    worker.prepare_hidden_state_capture("api-request", target_len=3)
    runner.input_batch = SimpleNamespace(req_ids=["first", "second"], req_id_to_index={"first": 0, "second": 1})
    runner.query_start_loc = SimpleNamespace(np=torch.tensor([0, 3]))
    runner.requests = {"first": _request(3), "second": _request(3)}

    with pytest.raises(RuntimeError, match="cannot safely bind"):
        runner._get_prompt_logprobs_dict(torch.zeros((6, 4)), {"first": 3, "second": 3})


def test_capture_rejects_pre_hc_head_width():
    worker = _FakeWorker(hidden_size=4)
    runner = worker.model_runner
    worker.prepare_hidden_state_capture("capture", target_len=2)
    runner.input_batch = SimpleNamespace(req_ids=["capture"], req_id_to_index={"capture": 0})
    runner.query_start_loc = SimpleNamespace(np=torch.tensor([0]))
    runner.requests = {"capture": _request(2)}

    with pytest.raises(RuntimeError, match="pre-hc_head"):
        runner._get_prompt_logprobs_dict(torch.zeros((2, 16)), {"capture": 2})


def test_discard_removes_failed_capture_state():
    worker = _FakeWorker()
    worker.prepare_hidden_state_capture("capture", target_len=2)

    assert worker.discard_hidden_state_capture("capture") is True
    assert worker.discard_hidden_state_capture("capture") is False
    assert worker._prime_hidden_capture_state() == {}


def test_non_primary_tp_worker_does_not_retain_hidden_payload():
    worker = _FakeWorker(local_rank=1)
    worker.device = torch.device("cuda", 1)
    runner = worker.model_runner
    worker.prepare_hidden_state_capture("capture", target_len=2)
    runner.input_batch = SimpleNamespace(req_ids=["capture"], req_id_to_index={"capture": 0})
    runner.query_start_loc = SimpleNamespace(np=torch.tensor([0]))
    runner.requests = {"capture": _request(2)}
    runner.num_prompt_logprobs = {"capture": 1}

    runner._get_prompt_logprobs_dict(torch.zeros((2, 4)), {"capture": 2})

    assert worker._prime_hidden_capture_state()["capture"]["chunks"] == {}
    assert worker.pop_hidden_state_capture("capture") is None


def test_api_discards_capture_when_generation_fails(monkeypatch):
    from prime_rl.inference.vllm import server

    class FakeRequest:
        async def json(self):
            return {"token_ids": [1, 2, 3]}

    class FakeClient:
        def __init__(self):
            self.rpc_calls = []

        async def collective_rpc(self, method, args=()):
            self.rpc_calls.append((method, args))
            return [True]

        def generate(self, *args, **kwargs):
            async def failing_stream():
                raise RuntimeError("generation failed")
                yield  # pragma: no cover

            return failing_stream()

    client = FakeClient()
    monkeypatch.setattr(server, "engine_client", lambda request: client)

    async def run_request():
        with pytest.raises(RuntimeError, match="generation failed"):
            await server.prefill_hidden_states(FakeRequest())

    asyncio.run(run_request())

    assert [method for method, _ in client.rpc_calls] == [
        "prepare_hidden_state_capture",
        "discard_hidden_state_capture",
    ]
