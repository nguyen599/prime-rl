from __future__ import annotations

import math

import torch

INT6_BLOCK_SIZE = 32
INT6_ROTATION_SEED = 7
INT6_CODEC = "had_int6_blk32"


def _signs(hidden_size: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(INT6_ROTATION_SEED)
    return (torch.randint(0, 2, (hidden_size,), generator=generator) * 2 - 1).to(device=device, dtype=torch.float32)


def _fwht(values: torch.Tensor) -> torch.Tensor:
    hidden_size = int(values.shape[-1])
    if hidden_size <= 0 or hidden_size & (hidden_size - 1):
        raise ValueError(f"Hadamard INT6 requires a power-of-two hidden size, got {hidden_size}")
    output = values.to(torch.float32)
    width = 1
    while width < hidden_size:
        reshaped = output.reshape(*output.shape[:-1], -1, 2, width)
        left = reshaped[..., 0, :]
        right = reshaped[..., 1, :]
        output = torch.cat((left + right, left - right), dim=-1).reshape(*output.shape)
        width *= 2
    return output / math.sqrt(hidden_size)


def encode_had_int6(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode ``[rows, hidden]`` states as packed INT6 plus FP16 block scales."""
    if hidden.ndim != 2:
        raise ValueError(f"hidden states must be two-dimensional, got {tuple(hidden.shape)}")
    rows, hidden_size = hidden.shape
    if hidden_size % INT6_BLOCK_SIZE or hidden_size % 4:
        raise ValueError(f"INT6 hidden size must be divisible by 32 and 4, got {hidden_size}")
    rotated = _fwht(hidden.to(torch.float32) * _signs(hidden_size, hidden.device))
    blocks = rotated.reshape(rows, hidden_size // INT6_BLOCK_SIZE, INT6_BLOCK_SIZE)
    scales = (blocks.abs().amax(dim=-1, keepdim=True) / 31.0).to(torch.float16)
    safe_scales = torch.where(scales == 0, torch.ones_like(scales), scales).to(torch.float32)
    quantized = (blocks / safe_scales).round().clamp(-31, 31).to(torch.int32).reshape(rows, hidden_size) + 31
    groups = quantized.reshape(rows, hidden_size // 4, 4)
    words = groups[..., 0] | (groups[..., 1] << 6) | (groups[..., 2] << 12) | (groups[..., 3] << 18)
    packed = torch.stack(
        (words & 0xFF, (words >> 8) & 0xFF, (words >> 16) & 0xFF),
        dim=-1,
    ).to(torch.uint8)
    return packed.reshape(rows, hidden_size * 6 // 8).contiguous(), scales.reshape(rows, -1).contiguous()


def decode_had_int6(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Decode packed states back into the original (unrotated) BF16 space."""
    if packed.ndim != 2 or scales.ndim != 2:
        raise ValueError("packed INT6 values and scales must be two-dimensional")
    rows = int(packed.shape[0])
    hidden_size = int(packed.shape[1]) * 8 // 6
    if scales.shape != (rows, hidden_size // INT6_BLOCK_SIZE):
        raise ValueError(
            f"INT6 scale shape {tuple(scales.shape)} does not match packed shape {tuple(packed.shape)}"
        )
    triples = packed.reshape(rows, -1, 3).to(torch.int32)
    words = triples[..., 0] | (triples[..., 1] << 8) | (triples[..., 2] << 16)
    quantized = torch.stack(
        (words & 0x3F, (words >> 6) & 0x3F, (words >> 12) & 0x3F, (words >> 18) & 0x3F),
        dim=-1,
    ).reshape(rows, hidden_size) - 31
    rotated = (
        quantized.reshape(rows, hidden_size // INT6_BLOCK_SIZE, INT6_BLOCK_SIZE).to(torch.float32)
        * scales.reshape(rows, hidden_size // INT6_BLOCK_SIZE, 1).to(torch.float32)
    ).reshape(rows, hidden_size)
    return (_fwht(rotated) * _signs(hidden_size, rotated.device)).to(torch.bfloat16)
