# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Shared test fixtures for dlightrag tests."""

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid7

import pytest

from dlightrag.adapters.postgres.runtime.run_store import (
    PGRunStore,
)
from dlightrag.application.config import DlightragConfig, reset_config, set_config
from dlightrag.engine.ai.settings import (
    EmbeddingSettings,
    ModelRoleSettings,
    ModelSettings,
)
from dlightrag.engine.answer.runs.envelope import accepted_input_envelope
from dlightrag.engine.answer.runs.routing import RoutingAcceptance
from dlightrag.engine.runtime.policy import DEFAULT_RUN_RETENTION_SECONDS
from dlightrag.engine.runtime.records import (
    PendingArtifact,
    PendingArtifactReference,
    PreparedRunEnvelope,
    RunAccessScope,
    RunCreation,
    run_request_fingerprint,
)


class FingerprintingRunStore(PGRunStore):
    """Test adapter for low-level suites whose raw request is the public input."""

    async def create_run(
        self,
        *,
        envelope: PreparedRunEnvelope | None = None,
        run_id: str | None = None,
        owner_id: str | None = None,
        request: Mapping[str, Any] | None = None,
        prepared_input: Mapping[str, Any] | None = None,
        idempotency_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        resources: Sequence[Mapping[str, object]] = (),
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> RunCreation:
        if envelope is not None:
            if run_id is None:
                raise ValueError("run_id is required with an envelope")
            return await super().create_run(
                envelope=envelope,
                run_id=run_id,
                resources=resources,
                artifacts=artifacts,
                references=references,
                routing=routing,
            )
        if owner_id is None:
            raise ValueError("owner_id is required for raw test acceptance")

        from dlightrag.engine.agent.session.ids import SessionId

        if prepared_input is not None:
            prepared = {
                "agent_session_id": SessionId.new().value,
                "agent_lane_id": "main",
                **dict(prepared_input),
            }
            return await self._create_answer_run(
                owner_id=owner_id,
                prepared=prepared,
                fingerprint=idempotency_fingerprint or run_request_fingerprint(prepared_input),
                idempotency_key=idempotency_key,
                artifacts=artifacts,
                references=references,
            )
        request = request or {}
        prepared: dict[str, Any] = {
            "agent_session_id": SessionId.new().value,
            "agent_lane_id": "main",
            **dict(request),
        }
        return await self._create_answer_run(
            owner_id=owner_id,
            prepared=prepared,
            fingerprint=idempotency_fingerprint or run_request_fingerprint(request),
            idempotency_key=idempotency_key,
            artifacts=artifacts,
            references=references,
        )

    async def _create_answer_run(
        self,
        *,
        owner_id: str,
        prepared: Mapping[str, Any],
        fingerprint: str,
        idempotency_key: str | None,
        artifacts: Sequence[PendingArtifact],
        references: Sequence[PendingArtifactReference],
    ) -> RunCreation:
        run_id = str(uuid7())
        return await super().create_run(
            envelope=_answer_envelope(
                owner_id=owner_id,
                prepared=prepared,
                fingerprint=fingerprint,
                submission_key=idempotency_key or run_id,
            ),
            run_id=run_id,
            artifacts=artifacts,
            references=references,
        )

    async def create_run_in(
        self,
        conn: Any,
        *,
        envelope: PreparedRunEnvelope | None = None,
        run_id: str | None = None,
        owner_id: str | None = None,
        request: Mapping[str, Any] | None = None,
        idempotency_fingerprint: str | None = None,
        idempotency_key: str | None = None,
        artifacts: Sequence[PendingArtifact] = (),
        references: Sequence[PendingArtifactReference] = (),
        routing: RoutingAcceptance | None = None,
    ) -> RunCreation:
        if envelope is not None:
            if run_id is None:
                raise ValueError("run_id is required with an envelope")
            return await super().create_run_in(
                conn,
                envelope=envelope,
                run_id=run_id,
                artifacts=artifacts,
                references=references,
                routing=routing,
            )
        if owner_id is None:
            raise ValueError("owner_id is required for raw test acceptance")

        from dlightrag.engine.agent.session.ids import SessionId

        request = request or {}
        prepared: dict[str, Any] = {
            "agent_session_id": SessionId.new().value,
            "agent_lane_id": "main",
            **dict(request),
        }
        run_id = str(uuid7())
        return await super().create_run_in(
            conn,
            envelope=_answer_envelope(
                owner_id=owner_id,
                prepared=prepared,
                fingerprint=idempotency_fingerprint or run_request_fingerprint(request),
                submission_key=idempotency_key or run_id,
            ),
            run_id=run_id,
            artifacts=artifacts,
            references=references,
            routing=routing,
        )


def _answer_envelope(
    *, owner_id: str, prepared: Mapping[str, Any], fingerprint: str, submission_key: str
) -> PreparedRunEnvelope:
    return PreparedRunEnvelope(
        run_kind="answer",
        lane="query",
        submitted_by=owner_id,
        access_scope=RunAccessScope(kind="owner", scope_id=owner_id),
        submission_key=submission_key,
        request_fingerprint=fingerprint,
        payload=prepared,
        accepted_input=accepted_input_envelope(prepared),
        retention_seconds=DEFAULT_RUN_RETENTION_SECONDS,
    )


@pytest.fixture(autouse=True)
def _reset_config_singleton():
    """Reset the config singleton before each test."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def tmp_working_dir(tmp_path: Path) -> Path:
    """Create a temporary working directory structure."""
    working_dir = tmp_path / "dlightrag_storage"
    (working_dir / "artifacts" / "local").mkdir(parents=True)
    return working_dir


@pytest.fixture
def test_config(tmp_working_dir: Path) -> DlightragConfig:
    """Create a test config with temporary paths.

    Also sets the global singleton so that code calling get_config()
    directly (e.g. /health endpoint) gets the test config.
    """
    cfg = DlightragConfig(  # pyright: ignore[reportCallIssue, reportArgumentType]
        # type: ignore[call-arg]
        deployment={"working_dir": str(tmp_working_dir)},
        models={
            "chat": ModelRoleSettings(
                default=ModelSettings(
                    model="z-ai/glm-5.3-flash",
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests"),
                )
            ),
            "embedding": EmbeddingSettings(
                provider="voyage",
                model="voyage-multimodal-3.5",
                api_key=os.getenv("DLIGHTRAG_OPENAI_API_KEY", "test-key-for-unit-tests"),
                startup_probe=False,
            ),
        },
    )
    set_config(cfg)
    return cfg
