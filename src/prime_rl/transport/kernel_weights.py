import json
from collections.abc import Iterator
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

KERNEL_WEIGHT_MANIFEST = "prime_rl_kernel_weights.json"
KERNEL_WEIGHT_FORMAT = "prime_rl_vllm_kernel_v1"


def has_kernel_weight_manifest(weight_dir: Path) -> bool:
    return (weight_dir / KERNEL_WEIGHT_MANIFEST).is_file()


def save_kernel_weight_shard(weight_dir: Path, filename: str, state_dict: dict[str, torch.Tensor]) -> None:
    if not state_dict:
        return
    cpu_state = {
        name: tensor.detach().to("cpu", non_blocking=False).contiguous() for name, tensor in state_dict.items()
    }
    save_file(cpu_state, weight_dir / filename, metadata={"format": KERNEL_WEIGHT_FORMAT})


def save_kernel_weight_manifest(weight_dir: Path, filenames: list[str], *, quantized_fp8: bool) -> None:
    manifest = {
        "format": KERNEL_WEIGHT_FORMAT,
        "quantized_fp8": quantized_fp8,
        "files": filenames,
    }
    manifest_path = weight_dir / KERNEL_WEIGHT_MANIFEST
    temporary_path = manifest_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(manifest_path)


def iter_kernel_weights(weight_dir: Path) -> Iterator[tuple[str, torch.Tensor]]:
    manifest_path = weight_dir / KERNEL_WEIGHT_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != KERNEL_WEIGHT_FORMAT:
        raise ValueError(f"Unsupported Prime-RL kernel weight format in {manifest_path}: {manifest.get('format')!r}")

    filenames = manifest.get("files")
    if not isinstance(filenames, list) or not all(isinstance(filename, str) for filename in filenames):
        raise ValueError(f"Invalid kernel weight file list in {manifest_path}")

    for filename in filenames:
        shard_path = weight_dir / filename
        with safe_open(shard_path, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                yield name, shard.get_tensor(name)
