from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from prime_rl.configs.algorithm import OPDAlgoConfig
from prime_rl.orchestrator.algo.base import Algorithm
from prime_rl.transport import EncodedTensor, TensorFileReference
from prime_rl.utils.client import StaticInferencePool

if TYPE_CHECKING:
    from prime_rl.orchestrator.types import Rollout
    from prime_rl.transport import TrainingSample
    from prime_rl.utils.client import InferencePool


class OPDAlgorithm(Algorithm):
    """On-policy distillation. Needs a teacher: the frozen reference model the
    per-token reverse KL is computed against.

    The policy samples its own rollouts; at ship time each sample's full
    context is prefill-scored under the teacher (``ref_logprobs`` on the
    wire), and the trainer evaluates the KL against the live policy. No
    credit is assigned — rollouts keep ``advantages=None`` (advantage-based
    filters never fire) and samples ship no advantage stream; ``group_size``
    only fans out sampling."""

    action_loss_type = "ref_kl"

    def __init__(self, config: OPDAlgoConfig, policy_pool: InferencePool):
        super().__init__(config, policy_pool)
        self.opd_config = config
        self.teacher = config.teacher
        self.teacher_pool: StaticInferencePool | None = None  # static teacher endpoint, connected in setup()
        self._teacher_ready = False
        self._teacher_ready_lock = asyncio.Lock()

    async def setup(self) -> None:
        # Policy rollout generation can start while a large teacher endpoint
        # is still loading/compiling. The first completed rollout waits for
        # readiness immediately before teacher scoring.
        pool = await self.connect(self.teacher, wait_for_ready=False)
        if not isinstance(pool, StaticInferencePool):
            raise TypeError("opd teacher must be a static endpoint — prefill scoring needs fixed endpoints")
        self.teacher_pool = pool

    async def _ensure_teacher_ready(self, pool: StaticInferencePool) -> None:
        if self._teacher_ready:
            return
        async with self._teacher_ready_lock:
            if self._teacher_ready:
                return
            await pool.wait_for_ready(self.teacher.name)
            self._teacher_ready = True

    async def score_rollout(self, rollout: Rollout) -> None:
        pool = self.teacher_pool
        assert pool is not None, "teacher pool not connected — Algorithm.setup() must run first"
        await self._ensure_teacher_ready(pool)

        async def score_sample(sample: TrainingSample) -> None:
            token_ids = list(sample.token_ids)
            if self.opd_config.distill_mode == "full_vocab_hidden":
                storage_dir = (
                    self.opd_config.teacher_hidden_path
                    if self.opd_config.teacher_hidden_transport == "filesystem"
                    else None
                )
                selected_positions = None
                if self.opd_config.teacher_hidden_codec != "raw":
                    token_weights = sample.ref_kl_weights
                    if token_weights is None:
                        token_weights = [1.0 if value else 0.0 for value in sample.mask]
                    # A causal hidden row at p predicts token p+1. Persist only
                    # rows whose next token participates in the ref-KL loss.
                    selected_positions = [
                        token_index - 1
                        for token_index, weight in enumerate(token_weights)
                        if token_index > 0 and float(weight) != 0.0
                    ]
                    if not selected_positions:
                        raise ValueError("full-vocab OPD sample has no selected ref-KL hidden-state rows")
                score_kwargs = {
                    "dtype": self.opd_config.teacher_hidden_dtype,
                    "storage_dir": storage_dir,
                }
                if self.opd_config.teacher_hidden_codec != "raw":
                    score_kwargs.update(
                        selected_positions=selected_positions,
                        codec=self.opd_config.teacher_hidden_codec,
                    )
                hidden = await pool.score_hidden_states(token_ids, **score_kwargs)
                if isinstance(hidden, TensorFileReference):
                    sample.ref_hidden_states_file = hidden
                    sample.ref_hidden_states = None
                elif isinstance(hidden, EncodedTensor):
                    sample.ref_hidden_states = hidden
                    sample.ref_hidden_states_file = None
                else:
                    raise TypeError(f"unexpected teacher hidden-state payload: {type(hidden)!r}")
                sample.ref_logprobs = None
            else:
                sample.ref_logprobs = await pool.score(token_ids)

        await asyncio.gather(*(score_sample(sample) for sample in rollout.samples))
