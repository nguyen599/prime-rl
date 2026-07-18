"""Compare Magi FA3 and FA4 sink kernels on causal sliding attention."""

from __future__ import annotations

import argparse

import torch

from prime_rl.trainer.models.olmo3_sink.magi_sink import (
    magi_varlen_attention_with_sink,
    validate_magi_sink_backend,
)


def _run_backend(
    backend: str,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink: torch.Tensor,
    grad_out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    window_size: tuple[int, int],
) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
    inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (q, k, v, sink)]
    out = magi_varlen_attention_with_sink(
        inputs[0],
        inputs[1],
        inputs[2],
        inputs[3],
        cu_seqlens,
        cu_seqlens,
        max_seqlen,
        max_seqlen,
        attn_impl=backend,
        softmax_scale=q.shape[-1] ** -0.5,
        causal=True,
        window_size=window_size,
    )
    grads = torch.autograd.grad(out, inputs, grad_out)
    return out.detach(), tuple(grad.detach() for grad in grads)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--sliding-window", type=int, default=128)
    parser.add_argument("--atol", type=float, default=4e-2)
    parser.add_argument("--rtol", type=float, default=4e-2)
    args = parser.parse_args()

    for backend in ("olmo3_sink_fa3", "olmo3_sink_fa4"):
        validate_magi_sink_backend(backend)

    torch.manual_seed(1234)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    num_heads = 8
    head_dim = 128
    q = torch.randn(args.seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    sink = torch.full((num_heads,), -2.0, device=device, dtype=torch.float32)
    grad_out = torch.randn_like(q)
    cu_seqlens = torch.tensor(
        [0, args.seq_len // 2, args.seq_len], device=device, dtype=torch.int32
    )
    max_seqlen = args.seq_len // 2
    window_size = (args.sliding_window - 1, 0)

    fa3_out, fa3_grads = _run_backend(
        "olmo3_sink_fa3",
        q,
        k,
        v,
        sink,
        grad_out,
        cu_seqlens,
        max_seqlen,
        window_size,
    )
    fa4_out, fa4_grads = _run_backend(
        "olmo3_sink_fa4",
        q,
        k,
        v,
        sink,
        grad_out,
        cu_seqlens,
        max_seqlen,
        window_size,
    )

    torch.testing.assert_close(fa4_out, fa3_out, atol=args.atol, rtol=args.rtol)
    for name, fa4_grad, fa3_grad in zip(
        ("q", "k", "v", "sink"), fa4_grads, fa3_grads, strict=True
    ):
        if not torch.isfinite(fa4_grad).all():
            raise RuntimeError(f"FA4 produced a non-finite {name} gradient")
        torch.testing.assert_close(fa4_grad, fa3_grad, atol=args.atol, rtol=args.rtol)

    print(
        "FA3/FA4 sliding-window sink parity passed: "
        f"seq_len={args.seq_len} window={window_size} "
        f"output_max_diff={(fa4_out.float() - fa3_out.float()).abs().max().item():.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
