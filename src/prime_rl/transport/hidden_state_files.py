from __future__ import annotations

import os
import struct
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Iterable

from prime_rl.transport.types import TensorFileReference

if TYPE_CHECKING:
    import torch


MAGIC = b"PRLHS001"
VERSION = 1
HEADER_SIZE = 64
_HEADER = struct.Struct("<8sIIQQQ")
_DTYPE_TO_CODE = {"float16": 1, "bfloat16": 2, "float32": 3}
_CODE_TO_DTYPE = {code: dtype for dtype, code in _DTYPE_TO_CODE.items()}
_DTYPE_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}


def normalize_dtype(dtype: str) -> str:
    value = str(dtype).removeprefix("torch.")
    if value not in _DTYPE_TO_CODE:
        raise ValueError(f"unsupported hidden-state dtype {dtype!r}")
    return value


def tensor_nbytes(dtype: str, shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * _DTYPE_ITEMSIZE[normalize_dtype(dtype)]


def copy_tensor_file_reference(
    ref: TensorFileReference, *, unlink_after_read: bool | None = None
) -> TensorFileReference:
    return TensorFileReference(
        path=ref.path,
        dtype=ref.dtype,
        shape=list(ref.shape),
        offset=ref.offset,
        nbytes=ref.nbytes,
        unlink_after_read=ref.unlink_after_read if unlink_after_read is None else unlink_after_read,
    )


def slice_tensor_file_rows(ref: TensorFileReference, rows: int) -> TensorFileReference:
    if not ref.shape:
        raise ValueError("hidden-state file reference must have at least one dimension")
    rows = int(rows)
    if rows < 0 or rows > int(ref.shape[0]):
        raise ValueError(f"cannot slice {rows} rows from hidden-state shape {ref.shape}")
    shape = [rows, *ref.shape[1:]]
    return TensorFileReference(
        path=ref.path,
        dtype=ref.dtype,
        shape=shape,
        offset=ref.offset,
        nbytes=tensor_nbytes(ref.dtype, shape),
        unlink_after_read=ref.unlink_after_read,
    )


def _header(dtype: str, shape: list[int], nbytes: int) -> bytes:
    if len(shape) != 2:
        raise ValueError(f"teacher hidden states must be 2D [tokens, hidden], got {shape}")
    core = _HEADER.pack(
        MAGIC,
        VERSION,
        _DTYPE_TO_CODE[normalize_dtype(dtype)],
        int(shape[0]),
        int(shape[1]),
        int(nbytes),
    )
    return core + b"\0" * (HEADER_SIZE - len(core))


def write_tensor_file(path: str | Path, tensor: torch.Tensor) -> TensorFileReference:
    """Atomically write a contiguous CPU ``[tokens, hidden]`` tensor."""
    import torch

    if tensor.device.type != "cpu":
        raise ValueError(f"hidden-state tensor must be on CPU before filesystem write, got {tensor.device}")
    if tensor.ndim != 2:
        raise ValueError(f"teacher hidden states must be 2D [tokens, hidden], got {tuple(tensor.shape)}")
    tensor = tensor.contiguous()
    dtype = normalize_dtype(str(tensor.dtype))
    shape = [int(dim) for dim in tensor.shape]
    nbytes = tensor.numel() * tensor.element_size()
    if nbytes != tensor_nbytes(dtype, shape):
        raise ValueError(f"hidden-state byte-size mismatch for dtype={dtype} shape={shape}: {nbytes}")

    output = Path(path)
    if not output.is_absolute():
        raise ValueError(f"hidden-state shared path must be absolute, got {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(_header(dtype, shape, nbytes))
            handle.write(memoryview(tensor.view(torch.uint8).numpy()))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return TensorFileReference(
        path=str(output),
        dtype=dtype,
        shape=shape,
        offset=HEADER_SIZE,
        nbytes=nbytes,
    )


def write_tensor_chunks_file(path: str | Path, chunks: list[torch.Tensor]) -> TensorFileReference:
    """Atomically write ordered CPU tensor chunks without concatenating them."""
    import torch

    if not chunks:
        raise ValueError("cannot write an empty hidden-state chunk list")
    first = chunks[0]
    if first.device.type != "cpu" or first.ndim != 2:
        raise ValueError(f"hidden-state chunks must be 2D CPU tensors, got {first.device} {tuple(first.shape)}")
    dtype = normalize_dtype(str(first.dtype))
    hidden_size = int(first.shape[1])
    rows = 0
    for chunk in chunks:
        if (
            chunk.device.type != "cpu"
            or chunk.ndim != 2
            or normalize_dtype(str(chunk.dtype)) != dtype
            or int(chunk.shape[1]) != hidden_size
            or not chunk.is_contiguous()
        ):
            raise ValueError("hidden-state chunks must be contiguous and share CPU dtype/hidden size")
        rows += int(chunk.shape[0])
    shape = [rows, hidden_size]
    nbytes = tensor_nbytes(dtype, shape)

    output = Path(path)
    if not output.is_absolute():
        raise ValueError(f"hidden-state shared path must be absolute, got {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("wb") as handle:
            handle.write(_header(dtype, shape, nbytes))
            for chunk in chunks:
                handle.write(memoryview(chunk.view(torch.uint8).numpy()))
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    return TensorFileReference(
        path=str(output),
        dtype=dtype,
        shape=shape,
        offset=HEADER_SIZE,
        nbytes=nbytes,
    )


def validate_tensor_file(ref: TensorFileReference) -> None:
    path = Path(ref.path)
    with path.open("rb") as handle:
        raw_header = handle.read(HEADER_SIZE)
    if len(raw_header) != HEADER_SIZE:
        raise ValueError(f"hidden-state file is shorter than its header: {path}")
    magic, version, dtype_code, rows, hidden, payload_nbytes = _HEADER.unpack(raw_header[: _HEADER.size])
    if magic != MAGIC:
        raise ValueError(f"invalid hidden-state file magic in {path}: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported hidden-state file version {version} in {path}")
    dtype = _CODE_TO_DTYPE.get(dtype_code)
    if dtype is None or dtype != normalize_dtype(ref.dtype):
        raise ValueError(f"hidden-state dtype mismatch in {path}: file={dtype!r}, ref={ref.dtype!r}")
    if len(ref.shape) != 2 or int(ref.shape[1]) != int(hidden) or int(ref.shape[0]) > int(rows):
        raise ValueError(f"hidden-state shape mismatch in {path}: file={[int(rows), int(hidden)]}, ref={ref.shape}")
    expected = tensor_nbytes(ref.dtype, ref.shape)
    if ref.offset != HEADER_SIZE or ref.nbytes != expected or ref.nbytes > int(payload_nbytes):
        raise ValueError(
            f"hidden-state payload mismatch in {path}: offset={ref.offset}, nbytes={ref.nbytes}, "
            f"expected={expected}, file_payload={payload_nbytes}"
        )
    if path.stat().st_size < ref.offset + ref.nbytes:
        raise ValueError(f"hidden-state payload is truncated: {path}")


def map_tensor_file(ref: TensorFileReference) -> torch.Tensor:
    """Memory-map one file reference without copying it through Python bytes."""
    import torch

    validate_tensor_file(ref)
    file_size = Path(ref.path).stat().st_size
    mapped = torch.from_file(ref.path, shared=False, size=file_size, dtype=torch.uint8)
    payload = mapped.narrow(0, ref.offset, ref.nbytes)
    dtype = getattr(torch, normalize_dtype(ref.dtype))
    return payload.view(dtype).reshape(ref.shape)


def unlink_owned_tensor_files(refs: Iterable[TensorFileReference]) -> None:
    """Remove private consumer links after every trainer rank has mapped them."""
    for path in {ref.path for ref in refs if ref.unlink_after_read}:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def materialize_tensor_files(
    refs: list[TensorFileReference], expected_rows: int, *, unlink_owned: bool = True
) -> torch.Tensor:
    """Map ordered file segments, concatenate them, and add implicit zero padding."""
    import torch

    if not refs:
        raise ValueError("filesystem-backed hidden states require at least one file reference")
    dtype = normalize_dtype(refs[0].dtype)
    trailing_shape = refs[0].shape[1:]
    tensors: list[torch.Tensor] = []
    rows = 0
    for ref in refs:
        if normalize_dtype(ref.dtype) != dtype or ref.shape[1:] != trailing_shape:
            raise ValueError(
                "filesystem hidden-state segments must share dtype and trailing shape: "
                f"expected {dtype} {trailing_shape}, got {ref.dtype} {ref.shape[1:]}"
            )
        tensors.append(map_tensor_file(ref))
        rows += int(ref.shape[0])
    if rows > expected_rows:
        raise ValueError(f"filesystem hidden states have {rows} rows for a {expected_rows}-token microbatch")
    if rows < expected_rows:
        tensors.append(torch.zeros([expected_rows - rows, *trailing_shape], dtype=getattr(torch, dtype)))

    result = tensors[0] if len(tensors) == 1 else torch.cat(tensors, dim=0)
    # Private hard links can be removed after mmap/cat. POSIX mappings retain
    # the inode until the tensor is released.
    if unlink_owned:
        unlink_owned_tensor_files(refs)
    return result


def sweep_tensor_files(directory: str | Path, ttl_seconds: float) -> int:
    """Remove abandoned producer files older than ``ttl_seconds``."""
    root = Path(directory)
    now = time.time()
    removed = 0
    try:
        entries = list(root.glob("*.prlhs"))
    except OSError:
        return 0
    for path in entries:
        try:
            if now - path.stat().st_mtime > ttl_seconds:
                path.unlink()
                removed += 1
        except (FileNotFoundError, OSError):
            pass
    return removed
