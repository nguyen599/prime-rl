import torch

from prime_rl.trainer.models.layers import fp8_linear


class _FakeDeepGEMM:
    def __init__(self) -> None:
        self.reduction_sizes: list[int] = []

    def fp8_gemm_nt(self, a, b, d, c=None, recipe=None) -> None:
        lhs = a[0]
        rhs = b[0]
        self.reduction_sizes.append(lhs.size(1))
        previous = c.float().clone() if c is not None else torch.zeros_like(d, dtype=torch.float32)
        d.copy_(lhs.float() @ rhs.float().T + previous)


def _fake_transpose_cast(x, *_args):
    return x.T.contiguous(), torch.empty(0)


def test_fp8_weight_gradient_chunks_long_reduction_and_pads_tail(monkeypatch) -> None:
    torch.manual_seed(123)
    deep_gemm = _FakeDeepGEMM()
    monkeypatch.setattr(fp8_linear, "require_deep_gemm", lambda: deep_gemm)
    monkeypatch.setattr(fp8_linear, "per_token_cast_to_fp8_tp_triton", _fake_transpose_cast)

    x = torch.randn(641, 128, dtype=torch.bfloat16)
    grad_output = torch.randn(641, 128, dtype=torch.bfloat16)
    weight = torch.empty(128, 128, dtype=torch.bfloat16)

    actual = fp8_linear._fp8_weight_gradient(
        x,
        grad_output,
        weight,
        block_size=128,
        max_reduction_tokens=256,
    )
    expected = (grad_output.float().T @ x.float()).to(torch.bfloat16)

    assert deep_gemm.reduction_sizes == [256, 256, 256]
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.25)


def test_fp8_weight_gradient_rejects_unaligned_chunk_size() -> None:
    x = torch.empty(128, 128, dtype=torch.bfloat16)
    grad_output = torch.empty_like(x)
    weight = torch.empty_like(x)

    try:
        fp8_linear._fp8_weight_gradient(
            x,
            grad_output,
            weight,
            block_size=128,
            max_reduction_tokens=255,
        )
    except ValueError as exc:
        assert "positive multiple" in str(exc)
    else:
        raise AssertionError("unaligned FP8 weight-gradient chunk size was accepted")
