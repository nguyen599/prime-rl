import asyncio
import json

import httpx

from prime_rl.transport.types import EncodedTensor, TensorFileReference
from prime_rl.utils.client import prefill_hidden_states, prefill_logprobs


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict):
        # Match what AsyncOpenAI exposes — prefill_logprobs reads ``str(openai.base_url)``.
        self.base_url = "http://fake-host:8000/v1"
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=200,
            content=json.dumps(self._payload).encode(),
            request=request,
        )


def test_prefill_logprobs_uses_inference_generate():
    async def _run():
        fake_openai = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [None, {"11": {"logprob": -0.7}}, {"12": {"logprob": -0.3}}],
                "kv_transfer_params": None,
            }
        )
        result = await prefill_logprobs(fake_openai, "ref-model", [1, 2, 3])

        assert result == [0.0, -0.7, -0.3]
        assert fake_openai.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
                "body": {
                    "model": "ref-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]

    asyncio.run(_run())


def test_prefill_hidden_states_filesystem_returns_handle_without_decoding_payload(tmp_path):
    async def _run():
        output = tmp_path / "teacher" / "result.prlhs"
        fake_openai = _FakeOpenAIClient(
            {
                "transport": "filesystem",
                "path": str(output),
                "dtype": "bfloat16",
                "shape": [3, 4096],
                "offset": 64,
                "nbytes": 3 * 4096 * 2,
            }
        )
        result = await prefill_hidden_states(
            fake_openai,
            "teacher",
            [1, 2, 3],
            storage_dir=tmp_path / "teacher",
        )

        assert isinstance(result, TensorFileReference)
        assert result.dtype == "bfloat16"
        call = fake_openai.calls[0]
        assert call["url"] == "http://fake-host:8000/prime_rl/prefill_hidden_states"
        assert call["body"]["transport"] == "filesystem"
        assert call["body"]["dtype"] == "bfloat16"
        assert call["body"]["output_path"].startswith(str(tmp_path / "teacher"))
        assert call["body"]["output_path"].endswith(".prlhs")

    asyncio.run(_run())


def test_prefill_hidden_states_inline_remains_backward_compatible():
    async def _run():
        fake_openai = _FakeOpenAIClient(
            {
                "dtype": "bfloat16",
                "shape": [1, 2],
                "data": "AAAAAA==",
            }
        )
        result = await prefill_hidden_states(fake_openai, "teacher", [1])

        assert isinstance(result, EncodedTensor)
        assert result.dtype == "bfloat16"
        assert result.shape == [1, 2]
        assert fake_openai.calls[0]["body"] == {
            "model": "teacher",
            "token_ids": [1],
            "dtype": "bfloat16",
        }

    asyncio.run(_run())
