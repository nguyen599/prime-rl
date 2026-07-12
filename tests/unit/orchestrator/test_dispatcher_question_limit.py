from types import SimpleNamespace

from prime_rl.orchestrator.dispatcher import RolloutDispatcher


def make_dispatcher(limit: int | None, kinds: list[str]) -> RolloutDispatcher:
    dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
    dispatcher.max_inflight_questions = limit
    dispatcher.groups = {
        index: SimpleNamespace(kind=kind)
        for index, kind in enumerate(kinds)
    }
    return dispatcher


def test_question_limit_blocks_only_fresh_groups_at_capacity() -> None:
    dispatcher = make_dispatcher(2, ["train", "train"])

    assert not dispatcher.can_open_fresh_question("train")
    assert dispatcher.can_open_fresh_question("eval")


def test_question_limit_can_be_disabled() -> None:
    dispatcher = make_dispatcher(None, ["train", "train", "train"])

    assert dispatcher.can_open_fresh_question("train")
