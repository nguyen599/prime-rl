import torch

from prime_rl.trainer.models.layers.lm_head import _chunked_full_vocab_reverse_kl


def test_chunked_full_vocab_reverse_kl_matches_direct_distribution():
    generator = torch.Generator().manual_seed(17)
    student_hidden = torch.randn((1, 5, 3), generator=generator, requires_grad=True)
    teacher_hidden = torch.randn((1, 5, 4), generator=generator)
    student_weight = torch.randn((11, 3), generator=generator, requires_grad=True)
    teacher_weight = torch.randn((11, 4), generator=generator)
    mask = torch.tensor([[False, True, False, True, True]])
    weights = torch.tensor([[0.0, 0.5, 0.0, 1.5, 2.0]])
    inverse_temperature = torch.tensor([[1.0, 1.25, 0.5, 0.75, 1.5]])

    actual_loss, actual_metric = _chunked_full_vocab_reverse_kl(
        student_hidden,
        teacher_hidden,
        student_weight,
        teacher_weight,
        mask,
        weights,
        inverse_temperature,
        token_chunk_size=2,
        vocab_chunk_size=3,
    )

    student_logits = (student_hidden @ student_weight.t()) * inverse_temperature.unsqueeze(-1)
    teacher_logits = teacher_hidden @ teacher_weight.t()
    student_logp = torch.log_softmax(student_logits.float(), dim=-1)
    teacher_logp = torch.log_softmax(teacher_logits.float(), dim=-1)
    per_token_kl = (student_logp.exp() * (student_logp - teacher_logp)).sum(dim=-1)
    expected_loss = (per_token_kl[mask] * weights[mask]).sum()
    expected_metric = per_token_kl[mask].mean()

    torch.testing.assert_close(actual_loss, expected_loss, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_metric, expected_metric, atol=2e-5, rtol=2e-5)

    actual_loss.backward()
    assert student_hidden.grad is not None and torch.isfinite(student_hidden.grad).all()
    assert student_weight.grad is not None and torch.isfinite(student_weight.grad).all()
