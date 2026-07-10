import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
from openai import APIConnectionError
from verifiers.v1.clients.config import EvalClientConfig

from prime_rl.configs.shared import ClientConfig
from prime_rl.transport.types import EncodedTensor
from prime_rl.utils.client import PrefillScorer, _is_retryable_lora_error, load_lora_adapter, setup_clients


def test_is_retryable_lora_error_returns_true_for_404():
    response = MagicMock()
    response.status_code = 404
    error = httpx.HTTPStatusError("Not found", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_true_for_500():
    response = MagicMock()
    response.status_code = 500
    error = httpx.HTTPStatusError("Server error", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is True


def test_is_retryable_lora_error_returns_false_for_400():
    response = MagicMock()
    response.status_code = 400
    error = httpx.HTTPStatusError("Bad request", request=MagicMock(), response=response)
    assert _is_retryable_lora_error(error) is False


def test_is_retryable_lora_error_returns_false_for_non_http_error():
    assert _is_retryable_lora_error(ValueError("some error")) is False


def test_load_lora_adapter_succeeds_on_first_attempt():
    mock_client = AsyncMock()
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response

    asyncio.run(load_lora_adapter([mock_client], "test-lora", Path("/test/path")))

    mock_client.post.assert_called_once_with(
        "/load_lora_adapter",
        json={"lora_name": "test-lora", "lora_path": "/test/path"},
        timeout=httpx.Timeout(connect=10.0, read=30.0, write=60.0, pool=10.0),
    )


def test_setup_clients_assigns_renderer_and_dp_rank_headers():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
        headers={"X-Test": "test"},
        dp_rank_count=2,
        extra_headers_from_state={"X-Session-ID": "session_id"},
    )

    renderer_settings = Qwen3VLRendererConfig()
    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=renderer_settings,
    )

    assert [client.type for client in clients] == ["train", "train"]
    assert [client.renderer for client in clients] == [renderer_settings, renderer_settings]
    assert [client.renderer_model_name for client in clients] == [None, None]
    assert [client.base_url for client in clients] == ["http://worker-a:8000/v1"] * 2
    assert [client.headers["X-data-parallel-rank"] for client in clients] == ["0", "1"]
    assert clients[0].headers["X-Test"] == "test"


def test_setup_clients_assigns_renderer_model_name():
    from renderers import Qwen3VLRendererConfig

    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(
        client_config,
        client_type="renderer",
        renderer_config=Qwen3VLRendererConfig(),
        renderer_model_name="Qwen/Qwen3-VL-4B-Instruct",
    )

    assert clients[0].renderer_model_name == "Qwen/Qwen3-VL-4B-Instruct"


def test_setup_clients_preserves_chat_client_defaults():
    client_config = ClientConfig(
        base_url=["http://worker-a:8000/v1"],
        api_key_var="PRIME_API_KEY",
    )

    clients = setup_clients(client_config)

    assert clients == [
        EvalClientConfig(
            api_key_var="PRIME_API_KEY",
            base_url="http://worker-a:8000/v1",
            headers={},
        )
    ]


def test_prefill_hidden_states_retries_connection_errors(monkeypatch):
    calls = 0

    async def fake_prefill(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise APIConnectionError(request=httpx.Request("POST", "http://teacher/prime_rl/prefill_hidden_states"))
        return EncodedTensor(dtype="bfloat16", shape=[1, 2], data=b"\0\0\0\0")

    monkeypatch.setattr("prime_rl.utils.client.prefill_hidden_states", fake_prefill)
    monkeypatch.setenv("PRIME_RL_PREFILL_HIDDEN_RETRY_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("PRIME_RL_PREFILL_HIDDEN_RETRY_MIN_SECONDS", "0")
    monkeypatch.setenv("PRIME_RL_PREFILL_HIDDEN_RETRY_MAX_SECONDS", "0")
    scorer = PrefillScorer()
    config = EvalClientConfig(api_key_var="PRIME_API_KEY", base_url="http://teacher/v1", headers={})

    result = asyncio.run(scorer.score_hidden_states([config], "teacher", [1]))

    assert result.shape == [1, 2]
    assert calls == 2


def test_prefill_hidden_states_does_not_retry_correctness_errors(monkeypatch):
    calls = 0

    async def fake_prefill(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise RuntimeError("invalid hidden-state response")

    monkeypatch.setattr("prime_rl.utils.client.prefill_hidden_states", fake_prefill)
    monkeypatch.setenv("PRIME_RL_PREFILL_HIDDEN_RETRY_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("PRIME_RL_PREFILL_HIDDEN_RETRY_MIN_SECONDS", "0")
    monkeypatch.setenv("PRIME_RL_PREFILL_HIDDEN_RETRY_MAX_SECONDS", "0")
    scorer = PrefillScorer()
    config = EvalClientConfig(api_key_var="PRIME_API_KEY", base_url="http://teacher/v1", headers={})

    try:
        asyncio.run(scorer.score_hidden_states([config], "teacher", [1]))
    except RuntimeError as exc:
        assert str(exc) == "invalid hidden-state response"
    else:
        raise AssertionError("expected RuntimeError")

    assert calls == 1
