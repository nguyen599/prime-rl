"""Validate Prime-RL vLLM hidden capture against teacher prompt logprobs.

Run this against a live teacher server. It proves that each captured row is the
exact post-model state consumed by the checkpoint's LM head and that row p
predicts token p+1. Prompt logprobs and hidden states come from the same vLLM
forward, matching Proof-Pilot's SGLang teacher-extraction validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from pathlib import Path

import torch
from openai import AsyncOpenAI
from safetensors import safe_open
from transformers import AutoTokenizer

# Always validate the checkout containing this script, not a previously
# installed prime-rl package from site-packages.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from prime_rl.transport import EncodedTensor, TensorFileReference
from prime_rl.transport.hidden_state_codec import INT6_CODEC
from prime_rl.transport.hidden_state_files import materialize_tensor_files
from prime_rl.utils.client import prefill_hidden_states_with_prompt_logprobs


def load_head_weight(checkpoint: Path, key: str) -> torch.Tensor:
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        tensor_path = checkpoint / weight_map[key]
    else:
        candidates = sorted(checkpoint.glob("*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"no safetensors found under {checkpoint}")
        tensor_path = candidates[0]
    with safe_open(tensor_path, framework="pt", device="cpu") as handle:
        if key not in handle.keys():
            raise KeyError(f"{key!r} is not present in {tensor_path}")
        weight = handle.get_tensor(key)
    if weight.dim() != 2:
        raise ValueError(f"expected rank-2 LM-head weight, got {tuple(weight.shape)}")
    return weight.contiguous()


def decode_hidden(encoded: EncodedTensor) -> torch.Tensor:
    dtypes = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    data = bytearray(encoded.data)
    return torch.frombuffer(data, dtype=dtypes[encoded.dtype]).reshape(encoded.shape).clone()


async def validate(args: argparse.Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    token_ids = tokenizer.encode(args.prompt, add_special_tokens=False)
    if args.tokens > 0:
        if not token_ids:
            raise ValueError("validation prompt encoded to no tokens")
        token_ids = (token_ids * math.ceil(args.tokens / len(token_ids)))[: args.tokens]
    if len(token_ids) < 2:
        raise ValueError("validation prompt must encode to at least two tokens")

    config = json.loads((args.checkpoint / "config.json").read_text())
    expected_hidden_size = int(config["hidden_size"])
    expected_vocab_size = int(config["vocab_size"])
    head = load_head_weight(args.checkpoint, args.head_key)
    if tuple(head.shape) != (expected_vocab_size, expected_hidden_size):
        raise ValueError(f"LM-head shape {tuple(head.shape)} != expected {(expected_vocab_size, expected_hidden_size)}")

    count = min(args.positions, len(token_ids) - 1)
    positions = torch.linspace(0, len(token_ids) - 2, steps=count, dtype=torch.float64).round().long().unique()
    selected_positions = positions.tolist() if args.codec == INT6_CODEC else None
    storage_dir = args.storage_dir if args.transport == "filesystem" else None
    if args.codec == INT6_CODEC and storage_dir is None:
        raise ValueError(f"{INT6_CODEC} validation requires --transport filesystem")
    if storage_dir is not None:
        storage_dir.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    try:
        hidden_payload, engine_logprobs = await prefill_hidden_states_with_prompt_logprobs(
            client,
            args.model_name,
            token_ids,
            dtype="bfloat16",
            storage_dir=storage_dir,
            selected_positions=selected_positions,
            codec=args.codec,
        )
    finally:
        await client.close()

    if isinstance(hidden_payload, EncodedTensor):
        hidden = decode_hidden(hidden_payload)
    elif isinstance(hidden_payload, TensorFileReference):
        try:
            hidden = materialize_tensor_files([hidden_payload], expected_rows=len(token_ids), unlink_owned=False)
        finally:
            Path(hidden_payload.path).unlink(missing_ok=True)
    else:
        raise TypeError(f"unsupported hidden payload {type(hidden_payload)!r}")
    if tuple(hidden.shape) != (len(token_ids), expected_hidden_size):
        raise AssertionError(
            f"captured hidden shape {tuple(hidden.shape)} != expected {(len(token_ids), expected_hidden_size)}"
        )
    if len(engine_logprobs) != len(token_ids):
        raise AssertionError(f"engine returned {len(engine_logprobs)} prompt logprobs for {len(token_ids)} tokens")

    targets = torch.tensor(token_ids, dtype=torch.long)[positions + 1]
    logits = hidden[positions].bfloat16() @ head.bfloat16().t()
    reconstructed = torch.log_softmax(logits.float(), dim=-1)[torch.arange(len(positions)), targets]
    selected_engine_logprobs = [engine_logprobs[int(position) + 1] for position in positions]
    if any(value is None for value in selected_engine_logprobs):
        raise AssertionError("same-pass teacher response omitted a selected target-token logprob")
    engine = torch.tensor(selected_engine_logprobs, dtype=torch.float32)
    difference = (reconstructed - engine).abs()

    print(
        f"hidden_shape={tuple(hidden.shape)} dtype={hidden.dtype} transport={args.transport} "
        f"codec={args.codec} positions={len(positions)} "
        f"logprob_max_abs={difference.max().item():.6g} "
        f"logprob_mean_abs={difference.mean().item():.6g}"
    )
    mean_tolerance = args.mean_tolerance
    if mean_tolerance is None:
        mean_tolerance = 0.2 if args.codec == INT6_CODEC else 1e-3
    if difference.mean().item() > mean_tolerance:
        raise AssertionError(
            f"hidden/head logprob parity failed: mean absolute error {difference.mean().item():.6g} > {mean_tolerance}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--head-key", default="head.weight")
    parser.add_argument("--prompt", default="Prove that the sum of two even integers is even.")
    parser.add_argument("--tokens", type=int, default=0, help="Repeat and truncate the prompt to exactly this size")
    parser.add_argument("--positions", type=int, default=32)
    parser.add_argument("--transport", choices=("inline", "filesystem"), default="inline")
    parser.add_argument("--codec", choices=("raw", INT6_CODEC), default="raw")
    parser.add_argument("--storage-dir", type=Path, default=Path("/tmp/prime-hidden-validation"))
    parser.add_argument("--mean-tolerance", type=float)
    parser.add_argument("--timeout", type=float, default=600.0)
    asyncio.run(validate(parser.parse_args()))


if __name__ == "__main__":
    main()
