import os
from pathlib import Path

import torch

from prime_rl.trainer.batch import prepare_batch, prepare_sample
from prime_rl.trainer.utils import build_bin_cost
from prime_rl.transport.filesystem import FileSystemMicroBatchReceiver, FileSystemMicroBatchSender
from prime_rl.transport.hidden_state_files import (
    copy_tensor_file_reference,
    materialize_tensor_files,
    slice_tensor_file_rows,
    write_tensor_chunks_file,
    write_tensor_file,
)
from prime_rl.transport.types import EncodedTensor, TrainingSample


def _sample(token_ids: list[int], ref) -> TrainingSample:
    length = len(token_ids)
    return TrainingSample(
        token_ids=token_ids,
        mask=[False] + [True] * (length - 1),
        logprobs=[0.0] + [-0.1] * (length - 1),
        temperatures=[1.0] * length,
        env_name="test",
        rl_weights=[0.0] * length,
        ref_kl_weights=[0.0] + [1.0] * (length - 1),
        ref_hidden_states_file=ref,
    )


def test_tensor_file_slice_materialize_and_owned_cleanup(tmp_path: Path):
    source = tmp_path / "source.prlhs"
    values = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    ref = write_tensor_file(source, values)

    sliced = slice_tensor_file_rows(ref, 3)
    private = tmp_path / "private.prlhs"
    os.link(source, private)
    sliced = copy_tensor_file_reference(sliced, unlink_after_read=True)
    sliced.path = str(private)

    result = materialize_tensor_files([sliced], expected_rows=5)

    torch.testing.assert_close(result[:3], values[:3])
    torch.testing.assert_close(result[3:], torch.zeros((2, 3), dtype=torch.bfloat16))
    assert source.exists()
    assert not private.exists()


def test_chunked_writer_does_not_require_a_concatenated_tensor(tmp_path: Path):
    chunks = [
        torch.arange(6, dtype=torch.bfloat16).reshape(2, 3),
        torch.arange(6, 12, dtype=torch.bfloat16).reshape(2, 3),
    ]
    ref = write_tensor_chunks_file(tmp_path / "chunks.prlhs", chunks)

    restored = materialize_tensor_files([ref], expected_rows=4, unlink_owned=False)

    torch.testing.assert_close(restored, torch.arange(12, dtype=torch.bfloat16).reshape(4, 3))


def test_filesystem_hidden_states_stay_as_handles_through_packing(tmp_path: Path):
    first_values = torch.arange(6, dtype=torch.float16).reshape(2, 3)
    second_values = torch.arange(9, dtype=torch.float16).reshape(3, 3) + 10
    first_ref = write_tensor_file(tmp_path / "first.prlhs", first_values)
    second_ref = write_tensor_file(tmp_path / "second.prlhs", second_values)

    grid = prepare_batch(
        rollouts=[_sample([1, 2], first_ref), _sample([3, 4, 5], second_ref)],
        seq_len=8,
        num_train_workers=1,
        idxs=[0, 0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
        pad_to_multiple_of=8,
    )
    micro_batch = grid[0][0]

    assert micro_batch.ref_hidden_states is None
    assert micro_batch.ref_hidden_state_files is not None
    assert [ref.shape for ref in micro_batch.ref_hidden_state_files] == [[3, 3], [2, 3]]
    assert len(micro_batch.input_ids) == 8

    # Filesystem sender gives this rank private hard links and removes the
    # producer names only after the rank batch is durable.
    output_dir = tmp_path / "trainer_output"
    sender = FileSystemMicroBatchSender(output_dir, data_world_size=1)
    sender.send(grid)
    assert not (tmp_path / "first.prlhs").exists()
    assert not (tmp_path / "second.prlhs").exists()

    receiver = FileSystemMicroBatchReceiver(output_dir, data_rank=0)
    received = receiver.receive()[0]
    assert received.ref_hidden_state_files is not None
    assert all(ref.unlink_after_read for ref in received.ref_hidden_state_files)
    private_paths = [Path(ref.path) for ref in received.ref_hidden_state_files]
    assert all(path.exists() for path in private_paths)

    result = materialize_tensor_files(received.ref_hidden_state_files, expected_rows=len(received.input_ids))
    # Packing sorts by descending sequence length.
    expected = torch.cat([second_values, first_values, torch.zeros((3, 3), dtype=torch.float16)])
    torch.testing.assert_close(result, expected)
    assert all(not path.exists() for path in private_paths)


def test_prepare_batch_truncates_file_reference_without_reading_payload(tmp_path: Path):
    values = torch.arange(30, dtype=torch.float32).reshape(10, 3)
    ref = write_tensor_file(tmp_path / "long.prlhs", values)

    grid = prepare_batch(
        rollouts=[_sample(list(range(10)), ref)],
        seq_len=6,
        num_train_workers=1,
        idxs=[0],
        num_loras=1,
        bin_cost=build_bin_cost(None),
    )
    micro_batch = grid[0][0]
    assert micro_batch.ref_hidden_state_files is not None
    assert micro_batch.ref_hidden_state_files[0].shape == [6, 3]
    result = materialize_tensor_files(micro_batch.ref_hidden_state_files, expected_rows=6, unlink_owned=False)
    torch.testing.assert_close(result, values[:6])


def test_inline_bfloat16_hidden_states_still_support_truncation():
    values = torch.arange(30, dtype=torch.bfloat16).reshape(10, 3)
    sample = _sample(list(range(10)), ref=None)
    sample.ref_hidden_states = EncodedTensor(
        dtype="bfloat16",
        shape=[10, 3],
        data=values.view(torch.uint8).numpy().tobytes(),
    )

    micro_batch = prepare_sample(sample, seq_len=6)

    assert micro_batch.ref_hidden_states is not None
    assert micro_batch.ref_hidden_states.shape == [6, 3]
    restored = torch.frombuffer(bytearray(micro_batch.ref_hidden_states.data), dtype=torch.bfloat16).reshape(6, 3)
    torch.testing.assert_close(restored, values[:6])
