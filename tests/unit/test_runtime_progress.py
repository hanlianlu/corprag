# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for durable progress arithmetic, bound stores, and host updates (3A)."""

import inspect
from typing import get_args

import pytest

from dlightrag.engine.agent.session.ids import StageIntentId
from dlightrag.engine.runtime.progress import (
    RunProgressStore,
    StageCommitResult,
    StageTerminalCommitResult,
)
from dlightrag.engine.runtime.records import (
    AlreadyCommittedTerminal,
    Deferred,
    Failed,
    ReclaimDecision,
    ReclaimState,
    RunExecutionContext,
    RunExecutionOutcome,
    Succeeded,
    WaitingForRepair,
    advance_reclaim,
)
from dlightrag.engine.runtime.settlements import (
    CompleteBlobDescriptor,
    EffectHostUpdate,
    OpaqueEvidenceResourceWrite,
    OpaqueEvidenceWrite,
    OpaqueFetchedResourceWrite,
)


class TestReclaimArithmetic:
    def test_first_reclaim_without_progress_counts_one(self) -> None:
        decision = advance_reclaim(
            ReclaimState(
                durable_progress_version=0,
                last_reclaim_progress_version=0,
                reclaims_without_progress=0,
            )
        )
        assert decision == ReclaimDecision(
            abandoned=False, reclaims_without_progress=1, last_reclaim_progress_version=0
        )

    def test_fourth_reclaim_without_progress_abandons(self) -> None:
        state = ReclaimState(
            durable_progress_version=5,
            last_reclaim_progress_version=5,
            reclaims_without_progress=3,
        )
        decision = advance_reclaim(state)
        assert decision.abandoned
        assert decision.reclaims_without_progress == 4

    def test_third_reclaim_without_progress_still_claims(self) -> None:
        decision = advance_reclaim(
            ReclaimState(
                durable_progress_version=5,
                last_reclaim_progress_version=5,
                reclaims_without_progress=2,
            )
        )
        assert not decision.abandoned

    def test_progress_between_reclaims_resets_counter_to_one(self) -> None:
        # Durable progress advanced past the last reclaim's observed version:
        # the no-progress counter resets and this reclaim counts as one.
        decision = advance_reclaim(
            ReclaimState(
                durable_progress_version=9,
                last_reclaim_progress_version=5,
                reclaims_without_progress=3,
            )
        )
        assert decision == ReclaimDecision(
            abandoned=False, reclaims_without_progress=1, last_reclaim_progress_version=9
        )

    def test_max_reclaims_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            advance_reclaim(
                ReclaimState(
                    durable_progress_version=0,
                    last_reclaim_progress_version=0,
                    reclaims_without_progress=0,
                ),
                max_reclaims=0,
            )


class TestBoundStores:
    def test_progress_store_methods_carry_no_fencing_parameters(self) -> None:
        load_params = list(inspect.signature(RunProgressStore.load_stage).parameters)
        settle_params = list(inspect.signature(RunProgressStore.settle_stage).parameters)
        terminal_params = list(inspect.signature(RunProgressStore.settle_terminal).parameters)
        assert load_params == ["self", "stage_intent_id"]
        assert set(settle_params) == {
            "self",
            "expected_progress_version",
            "stage_intent_id",
            "stage_name",
            "state",
            "evidence",
        }
        assert set(terminal_params) == {
            "self",
            "expected_progress_version",
            "stage_intent_id",
            "state",
            "result",
        }
        for name in ("owner_id", "run_id", "worker_id", "lease_owner", "fencing_epoch"):
            assert name not in settle_params
            assert name not in terminal_params

    def test_stage_and_execution_results_are_closed_value_unions(self) -> None:
        from dlightrag.engine.runtime.progress import (
            StageCommit,
            StageConflict,
            StageEvidenceConflict,
            StageLeaseLost,
            StageProgressConflict,
            StageTerminalCommit,
        )

        assert set(get_args(StageCommitResult.__value__)) == {
            StageCommit,
            StageProgressConflict,
            StageLeaseLost,
            StageConflict,
            StageEvidenceConflict,
        }
        assert set(get_args(StageTerminalCommitResult.__value__)) == {
            StageTerminalCommit,
            StageProgressConflict,
            StageLeaseLost,
            StageConflict,
        }
        assert set(get_args(RunExecutionOutcome.__value__)) == {
            Succeeded,
            Failed,
            Deferred,
            WaitingForRepair,
            AlreadyCommittedTerminal,
        }

    async def test_fast_boundaries_do_not_fall_back_to_stage_settlement(self) -> None:
        from dlightrag.engine.answer.fast import FastRunBoundaries

        class _LegacyProgress:
            stage_calls = 0

            async def settle_stage(self, **kwargs):  # type: ignore[no-untyped-def]
                self.stage_calls += 1
                return None

        progress = _LegacyProgress()
        boundaries = FastRunBoundaries(
            session=None,  # type: ignore[arg-type]
            progress=progress,  # type: ignore[arg-type]
            run_id="run",
            initial_progress_version=0,
            plan={},
        )

        with pytest.raises(AttributeError, match="settle_terminal"):
            await boundaries.settle_final(result={"answer": "x"}, result_digest="digest")
        assert progress.stage_calls == 0

    def test_already_committed_execution_requires_a_known_terminal(self) -> None:
        from dlightrag.engine.runtime import AlreadyCommittedTerminal, TerminalOutcome

        for terminal in (
            TerminalOutcome(committed=False, status=None, event_sequence=None),
            TerminalOutcome(committed=True, status=None, event_sequence=1),
            TerminalOutcome(committed=True, status="running", event_sequence=1),
            TerminalOutcome(committed=True, status="succeeded", event_sequence=None),
        ):
            with pytest.raises(ValueError, match="known terminal"):
                AlreadyCommittedTerminal(terminal)


class TestRunExecutionContext:
    def test_context_is_frozen_and_carries_the_claim_binding(self) -> None:
        from tests.in_memory_session_repository import InMemoryAgentSessionRepository

        class _NoopProgress:
            async def load_stage(self, stage_intent_id):  # type: ignore[no-untyped-def]
                return None

            async def settle_stage(self, **kwargs):  # type: ignore[no-untyped-def]
                return None

            async def settle_terminal(self, **kwargs):  # type: ignore[no-untyped-def]
                return None

        context = RunExecutionContext(
            owner_id="owner",
            run_id="run",
            worker_id="worker",
            lease_owner="worker",
            fencing_epoch=3,
            session_repository=InMemoryAgentSessionRepository(),  # type: ignore[arg-type]
            progress_store=_NoopProgress(),  # type: ignore[arg-type]
        )
        from dataclasses import FrozenInstanceError

        with pytest.raises(FrozenInstanceError):
            context.fencing_epoch = 4  # type: ignore[misc]

    def test_context_never_exposes_mutable_fencing(self) -> None:
        from dataclasses import is_dataclass

        params = list(inspect.signature(RunExecutionContext.__init__).parameters)
        assert "fencing_epoch" in params
        assert is_dataclass(RunExecutionContext)


class TestEffectHostUpdate:
    def test_evidence_write_validates_digests_and_ordinal(self) -> None:
        with pytest.raises(ValueError):
            OpaqueEvidenceWrite(
                session_id="s",
                intent_id="i",
                result_ordinal=-1,
                content_digest="a" * 64,
                locator_digest="b" * 64,
                content=b"c",
                locator=b"l",
            )
        with pytest.raises(ValueError):
            OpaqueEvidenceWrite(
                session_id="s",
                intent_id="i",
                result_ordinal=0,
                content_digest="short",
                locator_digest="b" * 64,
                content=b"c",
                locator=b"l",
            )

    def test_complete_blob_requires_chunk_sum_match(self) -> None:
        with pytest.raises(ValueError):
            CompleteBlobDescriptor(digest="a" * 64, total_bytes=10, chunks=(b"abc",))

    def test_aggregate_has_all_atomic_effect_channels(self) -> None:
        from dataclasses import fields

        assert {field.name for field in fields(EffectHostUpdate)} == {
            "evidence",
            "resources",
            "fetched",
            "committed_outputs",
            "workspace_inventory",
            "artifact_attachment",
            "memory_operation",
        }

    def test_fetched_resource_write_validates_digests(self) -> None:
        with pytest.raises(ValueError):
            OpaqueFetchedResourceWrite(
                resource_id="r",
                ordinal=0,
                safe_name="n",
                media_type="text/plain",
                capabilities={},
                blob_digest="a" * 64,
                source_locator_digest="short",
                source_locator=b"l",
                session_id="s",
                intent_id="i",
            )

    def test_evidence_resource_write_validates_identity(self) -> None:
        with pytest.raises(ValueError):
            OpaqueEvidenceResourceWrite(
                resource_id="  ",
                safe_name="n",
                media_type="text/plain",
                capabilities={},
                session_id="s",
                intent_id="i",
                result_ordinal=0,
                locator_digest="a" * 64,
            )


class TestStageIdentities:
    def test_fast_stage_intents_are_deterministic(self) -> None:
        first = StageIntentId.deterministic(run_id="run-1", name="fast:planner:0")
        again = StageIntentId.deterministic(run_id="run-1", name="fast:planner:0")
        other = StageIntentId.deterministic(run_id="run-1", name="fast:retrieval:1")
        assert first == again
        assert first != other
