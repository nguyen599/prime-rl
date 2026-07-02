from __future__ import annotations

import torch
import torch.distributed as dist

try:
    from torch.distributed._tensor import DTensor
except Exception:  # pragma: no cover - older torch builds
    DTensor = ()  # type: ignore[assignment]


def _to_local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def assert_sink_grad_nonzero(model: torch.nn.Module, logger, *, context: str = "Olmo3Sink") -> None:
    """Fail fast if sink-aware training produced no sink gradients.

    This catches two silent configuration failures:
    - a generic attention backend dropped ``s_aux``;
    - CP sharded the sequence but never routed Olmo3Sink attention through the
      Ulysses all-to-all wrapper.
    """
    found = torch.tensor(0, device="cuda", dtype=torch.int32)
    nonzero = torch.tensor(0, device="cuda", dtype=torch.int32)
    abs_sum = torch.tensor(0.0, device="cuda", dtype=torch.float32)

    for name, param in model.named_parameters():
        if not name.endswith("self_attn.sinks") and ".self_attn.sinks" not in name:
            continue
        found += 1
        grad = param.grad
        if grad is None:
            continue
        local_grad = _to_local_tensor(grad.detach())
        grad_abs_sum = local_grad.float().abs().sum()
        abs_sum += grad_abs_sum
        if torch.isfinite(grad_abs_sum) and grad_abs_sum > 0:
            nonzero += 1

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(found, op=dist.ReduceOp.SUM)
        dist.all_reduce(nonzero, op=dist.ReduceOp.SUM)
        dist.all_reduce(abs_sum, op=dist.ReduceOp.SUM)

    if found.item() == 0:
        return
    if nonzero.item() == 0 or abs_sum.item() == 0.0:
        raise RuntimeError(
            f"{context} sink-gradient canary failed: found {found.item()} sink "
            "parameters but all sink gradients were zero. Check that "
            "attn='olmo3_sink_fa3' is selected and the CP/Ulysses sink wrapper "
            "is registered."
        )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    if rank == 0:
        logger.info(
            "{} sink-gradient canary passed: sink_params={} nonzero_grad_params={} grad_abs_sum={:.6e}",
            context,
            found.item(),
            nonzero.item(),
            abs_sum.item(),
        )
