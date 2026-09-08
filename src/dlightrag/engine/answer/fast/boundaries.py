# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Fast execution stage boundaries."""

from collections.abc import Mapping
from typing import Any

from dlightrag.engine.agent.session.effects import canonical_json
from dlightrag.engine.agent.session.ids import StageIntentId
from dlightrag.engine.runtime.coordinator import (
    LeaseLostError,
    RunSession,
)
from dlightrag.engine.runtime.errors import RunExecutionError
from dlightrag.engine.runtime.progress import (
    RunProgressStore,
    StageCommit,
    StageCommitResult,
    StageTerminalCommit,
    StageTerminalCommitResult,
)
from dlightrag.engine.runtime.records import TerminalOutcome


class FastRunBoundaries:
    """Durable three-stage Fast boundaries (planner, retrieval, final_generation)."""

    def __init__(
        self,
        *,
        session: RunSession,
        progress: RunProgressStore,
        run_id: str,
        initial_progress_version: int,
        plan: Mapping[str, Any],
    ) -> None:
        self._session = session
        self._progress = progress
        self._run_id = run_id
        self._plan = plan
        self._progress_version = initial_progress_version

    def observe_session_progress(self) -> None:
        """Account for the canonical Fast Assistant Session commit."""
        self._progress_version += 1

    async def enter_phase(self, phase: str) -> None:
        await self._session.enter_phase(phase)  # type: ignore[arg-type]

    async def check_cancelled(self) -> None:
        await self._session.check_cancelled()

    async def settle_planner(self) -> None:
        stage_id = StageIntentId.deterministic(run_id=self._run_id, name="fast:planner:0")
        committed = await self._progress.settle_stage(
            expected_progress_version=self._progress_version,
            stage_intent_id=stage_id,
            stage_name="planner",
            state=dict(self._plan),
            evidence=(),
        )
        await self._observe(committed)

    async def settle_retrieval(self, contexts: Any) -> None:
        stage_id = StageIntentId.deterministic(run_id=self._run_id, name="fast:retrieval:1")
        committed = await self._progress.settle_stage(
            expected_progress_version=self._progress_version,
            stage_intent_id=stage_id,
            stage_name="retrieval",
            state={"contexts": _contexts_summary(contexts)},
            evidence=(),
        )
        await self._observe(committed)

    async def load_settled_result(self) -> Mapping[str, Any] | None:
        """Load the canonical Host result staged before Session settlement."""
        stage = await self._progress.load_stage(self._final_stage_id())
        if stage is None:
            return None
        state = stage.state
        if stage.stage_name != "final_generation" or not isinstance(state, Mapping):
            raise RunExecutionError(
                "run_execution_failed",
                "The settled Fast Host result is malformed.",
            )
        result = state.get("result")
        digest = state.get("result_digest")
        if not isinstance(result, Mapping) or digest != canonical_json(result):
            raise RunExecutionError(
                "run_execution_failed",
                "The settled Fast Host result failed its canonical digest check.",
            )
        return dict(result)

    async def stage_result(self, *, result: Mapping[str, Any], result_digest: str) -> None:
        committed = await self._progress.settle_stage(
            expected_progress_version=self._progress_version,
            stage_intent_id=self._final_stage_id(),
            stage_name="final_generation",
            state={"result": dict(result), "result_digest": result_digest},
            evidence=(),
        )
        await self._observe(committed)

    async def settle_final(
        self, *, result: Mapping[str, Any], result_digest: str
    ) -> TerminalOutcome:
        committed = await self._progress.settle_terminal(
            expected_progress_version=self._progress_version,
            stage_intent_id=self._final_stage_id(),
            state={"result": dict(result), "result_digest": result_digest},
            result=result,
        )
        terminal = await self._observe(committed)
        if terminal is None:
            raise RuntimeError("terminal settlement returned a non-terminal commit")
        return terminal

    def _final_stage_id(self) -> StageIntentId:
        return StageIntentId.deterministic(
            run_id=self._run_id,
            name="fast:final_generation:2",
        )

    async def _observe(
        self, committed: StageCommitResult | StageTerminalCommitResult
    ) -> TerminalOutcome | None:
        if isinstance(committed, StageCommit):
            self._progress_version = committed.progress_version
            return None
        if isinstance(committed, StageTerminalCommit):
            self._progress_version = committed.progress_version
            return TerminalOutcome(
                committed=True,
                status=committed.status,
                event_sequence=committed.terminal_event_sequence,
            )
        raise LeaseLostError


def _contexts_summary(contexts: Any) -> list[dict[str, Any]]:
    return [
        {
            "kind": kind,
            "rows": len(contexts.get(kind, []) or []),
        }
        for kind in ("chunks", "entities", "relationships")
    ]
