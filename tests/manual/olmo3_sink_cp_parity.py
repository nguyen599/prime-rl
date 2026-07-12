"""Manual CP canary for Olmo3Sink Ulysses attention.

Run on a GPU host with MagiAttention sink extensions available:

    OLMO3_SINK_ATTN=olmo3_sink_fa2 torchrun --standalone --nproc-per-node=2 \
        tests/manual/olmo3_sink_cp_parity.py

The test compares a small CP=2 Ulysses sink model against the
sink-aware eager full-sequence reference, then verifies sink gradients are
nonzero after a backward pass.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from prime_rl.trainer.models.layers.ulysses_attn import substitute_hf_ulysses_attn, substitute_ulysses_attn
from prime_rl.trainer.models.olmo3_sink.configuration_olmo3_sink import Olmo3SinkConfig
from prime_rl.trainer.models.olmo3_sink.grad_check import assert_sink_grad_nonzero
from prime_rl.trainer.models.olmo3_sink.modeling_olmo3_sink import Olmo3SinkForCausalLM
from prime_rl.utils.cp import setup_cp_params


class _Logger:
    def info(self, message, *args):
        if dist.get_rank() == 0:
            print(message % args if args else message, flush=True)


def _make_config(attn_impl: str) -> Olmo3SinkConfig:
    config = Olmo3SinkConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=32,
        attention_dropout=0.0,
        sink_init_value=-2.0,
    )
    config._attn_implementation = attn_impl
    return config


def _gather_sequence(tensor: torch.Tensor) -> torch.Tensor:
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=1)


def main() -> None:
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16

    if dist.get_world_size() != 2:
        raise RuntimeError("This canary is intentionally written for CP=2.")

    attn_impl = os.environ.get("OLMO3_SINK_ATTN", "olmo3_sink_fa2")
    torch.manual_seed(1234)
    full_model = Olmo3SinkForCausalLM(_make_config("eager")).to(device=device, dtype=dtype).eval()
    cp_model = Olmo3SinkForCausalLM(_make_config(attn_impl)).to(device=device, dtype=dtype).train()
    cp_model.load_state_dict(full_model.state_dict())

    cp_group = dist.group.WORLD
    substitute_hf_ulysses_attn(cp_group)
    substitute_ulysses_attn(cp_group, attn_impl=attn_impl)

    seq_len = 64
    input_ids = (torch.arange(seq_len, device=device).reshape(1, seq_len) % 200) + 3
    position_ids = torch.arange(seq_len, device=device).reshape(1, seq_len)

    with torch.no_grad():
        ref_logits = full_model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits

    cp_input_ids, cp_position_ids = setup_cp_params(
        input_ids,
        position_ids,
        cp_rank=dist.get_rank(),
        cp_world_size=dist.get_world_size(),
        cp_group=cp_group,
        cp_style="ulysses",
    )
    cp_logits_local = cp_model(input_ids=cp_input_ids, position_ids=cp_position_ids, use_cache=False).logits
    cp_logits = _gather_sequence(cp_logits_local)

    max_diff = (ref_logits - cp_logits).abs().max()
    dist.all_reduce(max_diff, op=dist.ReduceOp.MAX)
    if dist.get_rank() == 0:
        print(f"max_logit_diff={max_diff.item():.6e}", flush=True)
    torch.testing.assert_close(cp_logits, ref_logits, rtol=5e-2, atol=5e-2)

    cp_logits_local.float().square().mean().backward()
    assert_sink_grad_nonzero(cp_model, _Logger(), context="Manual CP parity Olmo3Sink")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
