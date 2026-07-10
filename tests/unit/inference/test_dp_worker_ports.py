from types import SimpleNamespace

import pytest

from prime_rl.inference.patches import _deterministic_dp_worker_port_start


def test_deterministic_dp_worker_port_ranges_do_not_overlap():
    starts = [
        _deterministic_dp_worker_port_start(
            SimpleNamespace(data_parallel_size=8, data_parallel_rpc_port=37002, data_parallel_index=rank)
        )
        for rank in range(8)
    ]

    assert starts == [38002 + rank * 64 for rank in range(8)]
    assert all(right - left == 64 for left, right in zip(starts, starts[1:]))


def test_deterministic_dp_worker_port_is_disabled_for_non_dp():
    config = SimpleNamespace(data_parallel_size=1, data_parallel_rpc_port=37002, data_parallel_index=0)
    assert _deterministic_dp_worker_port_start(config) is None


def test_deterministic_dp_worker_port_rejects_invalid_range():
    config = SimpleNamespace(data_parallel_size=8, data_parallel_rpc_port=65000, data_parallel_index=7)
    with pytest.raises(ValueError, match="outside the valid TCP port range"):
        _deterministic_dp_worker_port_start(config)
