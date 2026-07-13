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
    _proof_opd_finish_reason,
    _proof_opd_problem,
    _proof_opd_stage,
    _proof_opd_task_type,
    _proof_opd_token_counts,
    _proof_opd_trace,
)


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
    task_data = {
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
    rollout = SimpleNamespace(
        env_name="proof_math",
        info={},
        metrics={},
        task=SimpleNamespace(data=task_data),
        branches=[],
        reward=0.0,
    )
    branch = SimpleNamespace(nodes=[node])
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
