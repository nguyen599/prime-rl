"""Compare Magi and pre-Magi FA3 sink forward/backward numerics on Hopper."""

from __future__ import annotations

import argparse
from collections.abc import Callable

import torch

from prime_rl.trainer.models.olmo3_sink.magi_sink import (
    magi_varlen_attention_with_sink,
    validate_magi_sink_backend,
)
from prime_rl.trainer.models.olmo3_sink.native_fa3_sink import (
    native_fa3_varlen_attention_with_sink,
    validate_native_fa3_sink_backend,
)


def run_attention(
    attention: Callable,
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    sink_base: torch.Tensor,
    upstream: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    window_size: tuple[int, int],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    sink = sink_base.detach().clone().requires_grad_(True)
    out = attention(
        q,
        k,
        v,
        sink,
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        window_size=window_size,
    )
    grads = torch.autograd.grad(out, (q, k, v, sink), upstream)
    return out.detach().float(), tuple(grad.detach().float() for grad in grads)


def report_and_check(
    label: str,
    native: torch.Tensor,
    magi: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    diff = native - magi
    relative_l2 = diff.norm() / native.norm().clamp_min(torch.finfo(torch.float32).tiny)
    print(
        f"{label}: max_abs={diff.abs().max().item():.6e} "
        f"relative_l2={relative_l2.item():.6e}",
        flush=True,
    )
    torch.testing.assert_close(magi, native, rtol=rtol, atol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--sliding-window", type=int, default=64)
    parser.add_argument("--rtol", type=float, default=0.1)
    parser.add_argument("--atol", type=float, default=0.02)
    args = parser.parse_args()

    validate_native_fa3_sink_backend()
    validate_magi_sink_backend("olmo3_sink_fa3")
    if args.seq_len % 2:
        raise ValueError("--seq-len must be even so the probe can pack two equal sequences")

    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 8
    head_dim = 128
    q = torch.randn(args.seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sink = torch.linspace(-4.0, 1.0, num_heads, device=device, dtype=torch.float32)
    upstream = torch.randn_like(q)
    cu_seqlens = torch.tensor([0, args.seq_len // 2, args.seq_len], device=device, dtype=torch.int32)
    max_seqlen = args.seq_len // 2

    def native_attention(*attention_args, **attention_kwargs):
        return native_fa3_varlen_attention_with_sink(*attention_args, **attention_kwargs)

    def magi_attention(*attention_args, **attention_kwargs):
        return magi_varlen_attention_with_sink(
            *attention_args,
            attn_impl="olmo3_sink_fa3",
            **attention_kwargs,
        )

    window_sizes = [(-1, -1)]
    if args.sliding_window > 0:
        window_sizes.append((args.sliding_window - 1, 0))

    for window_size in window_sizes:
        print(f"window_size={window_size}", flush=True)
        native_out, native_grads = run_attention(
            native_attention,
            q,
            k,
            v,
            sink,
            upstream,
            cu_seqlens,
            max_seqlen,
            window_size,
        )
        magi_out, magi_grads = run_attention(
            magi_attention,
            q,
            k,
            v,
            sink,
            upstream,
            cu_seqlens,
            max_seqlen,
            window_size,
        )
        report_and_check("output", native_out, magi_out, rtol=args.rtol, atol=args.atol)
        for name, native_grad, magi_grad in zip(
            ("dq", "dk", "dv", "dsink"),
            native_grads,
            magi_grads,
            strict=True,
        ):
            report_and_check(name, native_grad, magi_grad, rtol=args.rtol, atol=args.atol)


if __name__ == "__main__":
    main()
