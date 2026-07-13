import json
import sys
from types import ModuleType
from types import SimpleNamespace

try:
    import wandb_gql  # noqa: F401
except ImportError:
    wandb_gql = ModuleType("wandb_gql")
    wandb_gql.gql = lambda query: query
    sys.modules["wandb_gql"] = wandb_gql

from prime_rl.utils.monitor.wandb import (
    WandbMonitor,
    _proof_opd_finish_reason,
    _proof_opd_problem,
    _proof_opd_stage,
    _proof_opd_task_type,
    _proof_opd_token_counts,
    _proof_opd_trace,
)
from prime_rl.configs.shared import LogExtrasConfig, WandbWithExtrasConfig


def _single_turn_rollout():
    node = SimpleNamespace(
        sampled=True,
        message=SimpleNamespace(role="assistant", content="checked proof"),
        usage=SimpleNamespace(
            prompt_tokens=120,
            completion_tokens=30,
            reasoning_tokens=10,
            total_tokens=150,
        ),
        token_ids=list(range(32)),
        mask=[False, False] + [True] * 30,
        finish_reason="length",
    )
    task_record = {
        "idx": 7,
        "answer": json.dumps(
            {
                "problem": "Prove that the construction is cyclic.",
                "stage": "verify",
                "task_type": "opd_single_turn",
            }
        ),
        "stage": "verify",
        "source_index": 7,
    }
    task_data = SimpleNamespace(idx=7, model_dump=lambda mode="json": task_record)
    rollout = SimpleNamespace(
        env_name="proof_math",
        info={},
        metrics={},
        task=SimpleNamespace(data=task_data),
        branches=[],
        reward=0.0,
    )
    branch = SimpleNamespace(nodes=[node], token_ids=[1, 2, 3])
    rollout.branches = [branch]
    return rollout, branch


def test_proof_opd_fallback_exposes_single_turn_table_fields():
    rollout, branch = _single_turn_rollout()
    trace = _proof_opd_trace(rollout, branch)

    assert trace["reason"] == "fallback_from_rollout_message_nodes"
    assert _proof_opd_task_type(rollout, trace) == "opd_single_turn"
    assert _proof_opd_stage(rollout, trace) == "verify"
    assert _proof_opd_problem(rollout, trace) == "Prove that the construction is cyclic."
    assert _proof_opd_finish_reason(rollout, branch) == "length"
    assert _proof_opd_token_counts(rollout, branch) == {
        "prompt_tokens": 120,
        "generated_tokens": 30,
        "reasoning_tokens": 10,
        "total_tokens": 150,
    }


def test_proof_opd_table_helpers_prefer_environment_info():
    rollout, branch = _single_turn_rollout()
    rollout.info = {
        "task_type": "custom_single_turn",
        "stage": "select",
        "problem": "Choose the best proof.",
        "finish_reason": "stop",
        "token_counts": {
            "prompt_tokens": 20,
            "generated_tokens": 5,
            "reasoning_tokens": 1,
            "total_tokens": 25,
        },
        "proof_opd_trace": {"source_index": 9, "raw_output_excerpt": "candidate 2"},
    }
    trace = _proof_opd_trace(rollout, branch)

    assert _proof_opd_task_type(rollout, trace) == "custom_single_turn"
    assert _proof_opd_stage(rollout, trace) == "select"
    assert _proof_opd_problem(rollout, trace) == "Choose the best proof."
    assert _proof_opd_finish_reason(rollout, branch) == "stop"
    assert _proof_opd_token_counts(rollout, branch)["total_tokens"] == 25


def test_sample_table_logs_on_interval_and_replaces_previous_rows(monkeypatch):
    rollout, _ = _single_turn_rollout()
    logged = []

    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.data = []

        def add_data(self, *values):
            self.data.append(values)

    monkeypatch.setattr("prime_rl.utils.monitor.wandb.wandb.Table", FakeTable)
    monkeypatch.setattr("prime_rl.utils.monitor.wandb.wandb.log", lambda payload: logged.append(payload))

    monitor = WandbMonitor.__new__(WandbMonitor)
    monitor.is_master = True
    monitor.config = WandbWithExtrasConfig(
        log_extras=LogExtrasConfig(samples=True, distributions=False, interval=10, sample_ratio=1.0)
    )
    monitor.tokenizer = SimpleNamespace(decode=lambda token_ids: f"decoded:{len(token_ids)}")
    monitor.logger = SimpleNamespace(info=lambda *_args, **_kwargs: None, debug=lambda *_args, **_kwargs: None)
    monitor.last_log_samples_step = -1
    monitor.samples_cols = [
        "step",
        "env_name",
        "task",
        "task_idx",
        "messages",
        "input_ids",
        "reward",
        "task_type",
        "stage",
        "token_counts",
        "finish_reason",
        "problem",
        "proof_opd_trace",
    ]
    monitor.samples_table = FakeTable(monitor.samples_cols)

    monitor.log_samples([rollout], step=1)
    assert logged == []

    monitor.log_samples([rollout], step=10)
    step_10_table = logged[-1]["samples"]
    assert len(step_10_table.data) == 1
    assert step_10_table.data[0][0] == 10

    monitor.log_samples([rollout], step=20)
    step_20_table = logged[-1]["samples"]
    assert step_20_table is not step_10_table
    assert len(step_20_table.data) == 1
    assert step_20_table.data[0][0] == 20
