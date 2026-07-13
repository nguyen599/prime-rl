"""Single-GPU forward/backward canary for OLMo3 sink adapters."""

from __future__ import annotations

import argparse

import torch

from prime_rl.trainer.models.olmo3_sink.magi_sink import (
    magi_varlen_attention_with_sink,
    validate_magi_sink_backend,
)
from prime_rl.trainer.models.olmo3_sink.native_fa3_sink import (
    NATIVE_FA3_SINK_ATTN_IMPL,
    native_fa3_varlen_attention_with_sink,
    validate_native_fa3_sink_backend,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="olmo3_sink_fa2")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--sliding-window", type=int, default=64)
    args = parser.parse_args()

    if args.backend == NATIVE_FA3_SINK_ATTN_IMPL:
        validate_native_fa3_sink_backend()
    else:
        validate_magi_sink_backend(args.backend)
    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    seq_len = args.seq_len
    num_heads = 8
    head_dim = 128

    q = torch.randn(seq_len, num_heads, head_dim, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    sink = torch.full((num_heads,), -2.0, device=device, dtype=torch.float32, requires_grad=True)
    cu_seqlens = torch.tensor([0, seq_len // 2, seq_len], device=device, dtype=torch.int32)
    max_seqlen = seq_len // 2
    window_size = (args.sliding_window - 1, 0) if args.sliding_window > 0 else (-1, -1)

    common_args = (
        q,
        k,
        v,
        sink,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
    )
    common_kwargs = {
        "softmax_scale": head_dim**-0.5,
        "causal": True,
        "window_size": window_size,
    }
    if args.backend == NATIVE_FA3_SINK_ATTN_IMPL:
        out = native_fa3_varlen_attention_with_sink(*common_args, **common_kwargs)
    else:
        out = magi_varlen_attention_with_sink(
            *common_args,
            attn_impl=args.backend,
            **common_kwargs,
        )
    out.float().square().mean().backward()

    tensors = {"q": q, "k": k, "v": v, "sink": sink}
    for name, tensor in tensors.items():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise RuntimeError(f"{args.backend}: invalid {name} gradient")
    if sink.grad.abs().max().item() == 0:
        raise RuntimeError(f"{args.backend}: sink gradient is zero")

    print(
        f"backend={args.backend} output_shape={tuple(out.shape)} "
        f"sink_grad_max={sink.grad.abs().max().item():.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
