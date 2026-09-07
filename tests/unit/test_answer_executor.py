# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer executor ownership and failure behavior."""

import asyncio
import datetime
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from PIL import Image

from dlightrag.application.answer_runs.capabilities import (
    AnswerCapabilities,
    RequestModelContext,
)
from dlightrag.application.answer_runs.capability import AnswerImageCapability
from dlightrag.application.answer_runs.errors import (
    CurrentDocumentParseError,
    CurrentImagePayloadError,
)
from dlightrag.application.answer_runs.execution import (
    AnswerRunRequest,
    AttachmentReference,
    LinkReference,
    PinnedModelProfile,
    build_current_answer_resources,
    in_memory_attachment_loader,
)
from dlightrag.application.errors import CorpusUnavailableError
from dlightrag.engine.agent.session.ids import LaneId, SessionId
from dlightrag.engine.agent.session.memory import MemoryAgentSessionRepository
from dlightrag.engine.agent.session.plan import AgentRunPlan
from dlightrag.engine.ai.capacity import CONTEXT_POLICY_REVISION, ModelProfile
from dlightrag.engine.ai.catalog import current_model_catalog_revision
from dlightrag.engine.ai.fingerprints import ModelFingerprint
from dlightrag.engine.ai.reasoning import best_effort_reasoning_profile
from dlightrag.engine.ai.scheduler import ModelScheduler
from dlightrag.engine.ai.telemetry import NOOP_TELEMETRY
from dlightrag.engine.answer.execution import (
    AnswerExecutor,
    AnswerExecutorSettings,
    AnswerResourceResolver,
    AnswerResourceSettings,
)
from dlightrag.engine.answer.execution.executor import (
    _close_execution_resources,
    _memory_recall_allowed,
    _stage_publications,
)
from dlightrag.engine.answer.fast import ensure_session_lane
from dlightrag.engine.answer.highlights import SemanticHighlightSettings
from dlightrag.engine.answer.publication import prepare_artifact_attachment, validate_publication
from dlightrag.engine.answer.resources import ResourceInput
from dlightrag.engine.answer.resources.models import TextWindowBudget
from dlightrag.engine.dependencies import ProviderUnavailableError
from dlightrag.engine.runtime import (
    Deferred,
    RunExecutionError,
    RunExecutionOutcome,
    RunSession,
    Succeeded,
    artifact_digest,
)
from tests.unit.conftest import answer_image_policy


@pytest.mark.asyncio
async def test_missing_fork_session_is_a_typed_run_conflict() -> None:
    store = MemoryAgentSessionRepository[None]()
    session_id = SessionId.new()

    with pytest.raises(RunExecutionError) as raised:
        await ensure_session_lane(
            repository=store,
            snapshot=await store.load(session_id),
            fencing_epoch=1,
            session_id=session_id,
            lane_id=LaneId.new(),
            source_lane_id=LaneId.main(),
        )

    assert raised.value.kind == "agent_session_conflict"


def _fingerprint(role: str) -> ModelFingerprint:
    return ModelFingerprint("openai", f"test-{role}", None)


def _executor() -> AnswerExecutor:
    return AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planner_history_input_measure=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=AnswerExecutorSettings(
            default_top_k=10,
            default_chunk_top_k=20,
            semantic_highlights=SemanticHighlightSettings(
                enabled=True,
                timeout=10.0,
                max_concurrency=8,
                batch_size=8,
                max_input_chars=4096,
                cache_size=500,
            ),
        ),
        telemetry=NOOP_TELEMETRY,
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
    )


def test_markdown_artifacts_keep_independent_citation_sources(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "analysis.md").write_text("Primary fact [1-1].", encoding="utf-8")
    (root / "appendix.md").write_text("Appendix fact [2-1].", encoding="utf-8")
    plan = validate_publication(
        root,
        answer=("[Open analysis](artifact:analysis.md) [Open appendix](artifact:appendix.md)"),
        attachments=(
            prepare_artifact_attachment(root, path="analysis.md"),
            prepare_artifact_attachment(root, path="appendix.md"),
        ),
    )
    contexts = {
        "chunks": [
            {
                "chunk_id": "chunk-primary",
                "reference_id": "1",
                "file_path": "primary.pdf",
                "content": "Primary fact.",
                "_workspace": "default",
                "full_doc_id": "doc-primary",
                "metadata": {
                    "source_uri": "local://default/primary.pdf",
                    "source_download_locator": "/private/primary.pdf",
                    "source_file_name": "primary.pdf",
                },
            },
            {
                "chunk_id": "chunk-appendix",
                "reference_id": "2",
                "file_path": "appendix.pdf",
                "content": "Appendix fact.",
                "_workspace": "default",
                "full_doc_id": "doc-appendix",
                "metadata": {
                    "source_uri": "local://default/appendix.pdf",
                    "source_download_locator": "/private/appendix.pdf",
                    "source_file_name": "appendix.pdf",
                },
            },
        ]
    }

    publications, descriptors, artifact_sources = _stage_publications(
        plan=plan,
        answer=plan.answer,
        contexts=contexts,
        require_answer=True,
    )

    resource_by_filename = {
        str(descriptor["filename"]): str(descriptor["resource_id"]) for descriptor in descriptors
    }
    assert [source.id for source in artifact_sources[resource_by_filename["analysis.md"]]] == ["1"]
    assert [source.id for source in artifact_sources[resource_by_filename["appendix.md"]]] == ["2"]
    content_by_filename = {
        publication.filename: publication.content for publication in publications
    }
    assert content_by_filename["analysis.md"] == b"Primary fact [1-1]."
    assert content_by_filename["appendix.md"] == b"Appendix fact [2-1]."


def test_acceptance_research_tools_include_every_configured_non_resource_surface() -> None:
    from pydantic import BaseModel

    from dlightrag.engine.agent.skills import SkillsBundle
    from dlightrag.engine.agent.tools import AgentTool, ToolResult

    class Args(BaseModel):
        value: str

    async def external(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("unused")

    executor = AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planner_history_input_measure=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
        execution_environment="trust",
        memory_store=MagicMock(),
        external_tools=(AgentTool("remote_lookup", "Remote lookup.", Args, external),),
        skills_bundle_factory=lambda owner_id, requested_skill=None: SkillsBundle(
            global_root=Path("/nonexistent-global-skills"),
        ),
    )

    names = {tool.name for tool in executor.acceptance_research_tools()}

    assert {
        "read",
        "write",
        "edit",
        "attach_artifact",
        "grep",
        "bash",
        "spawn_agent",
        "subagent_status",
        "wait_subagent",
        "cancel_subagent",
        "remember",
        "forget",
        "recall_memory",
        "load_skill",
        "remote_lookup",
    } <= names


def test_acceptance_plan_matches_runtime_tool_composition(tmp_path: Path) -> None:
    from dlightrag.engine.agent.environment.local import LocalExecutionEnvironment
    from dlightrag.engine.answer.evidence import EvidenceLedger
    from dlightrag.engine.answer.tools.composition import compose_research_tools
    from dlightrag.engine.answer.tools.subagents import SubagentHost

    executor = AnswerExecutor(
        store=MagicMock(),
        blob_store=MagicMock(),
        pool=MagicMock(),
        warm=Mock(),
        retrieve=AsyncMock(),
        planner_history_input_measure=AsyncMock(),
        models=MagicMock(),
        capabilities=MagicMock(),
        resources=MagicMock(),
        settings=_executor()._settings,
        telemetry=NOOP_TELEMETRY,
        model_fingerprint_for_role=_fingerprint,  # type: ignore[arg-type]
        execution_environment="trust",
    )
    accepted = executor.acceptance_research_tools()

    async def retrieve(_query: str) -> Any:
        raise RuntimeError("tool definitions are never executed")

    async def read_resource(_request: Any, _runtime: Any) -> Any:
        raise RuntimeError("tool definitions are never executed")

    runtime_tools = compose_research_tools(
        evidence=EvidenceLedger(),
        trace={},
        retrieve_knowledge_base=retrieve,  # type: ignore[arg-type]
        search_web=None,
        resource_tools=[],
        register_web_source=None,
        resource_reader=read_resource,
        environment=LocalExecutionEnvironment(tmp_path),
        artifacts_root=tmp_path / "artifacts",
        publication_limits=executor._settings.publication,
        subagent_host=SubagentHost(),
        skill_tools=[],
    )
    runtime_by_name = {tool.name: tool for tool in runtime_tools}
    runtime_surface = tuple(runtime_by_name[tool.name] for tool in accepted)

    accepted_plan = AgentRunPlan.from_tools(
        accepted,
        model_role="query",
        context_policy_revision="policy-1",
    )
    runtime_plan = AgentRunPlan.from_tools(
        runtime_surface,
        model_role="query",
        context_policy_revision="policy-1",
    )

    assert runtime_plan.digest == accepted_plan.digest


def test_execution_rejects_tools_that_differ_from_the_accepted_agent_plan() -> None:
    from pydantic import BaseModel

    from dlightrag.engine.agent.tools import AgentTool, ToolResult
    from dlightrag.engine.runtime import IncompatibleActiveRunError

    class Args(BaseModel):
        value: str

    async def execute(_args: BaseModel, _runtime: object) -> ToolResult:
        return ToolResult.text("unused")

    accepted_tool = AgentTool("lookup", "Accepted description.", Args, execute)
    plan = AgentRunPlan.from_tools(
        (accepted_tool,),
        model_role="query",
        context_policy_revision="policy-1",
    )
    request = MagicMock(agent_run_plan=plan, context_policy_revision="policy-1")

    AnswerExecutor.validate_pinned_agent_run_plan(request, (accepted_tool,))
    with pytest.raises(IncompatibleActiveRunError, match="missing"):
        AnswerExecutor.validate_pinned_agent_run_plan(
            MagicMock(agent_run_plan=None),
            (accepted_tool,),
        )
    with pytest.raises(IncompatibleActiveRunError, match="differs"):
        AnswerExecutor.validate_pinned_agent_run_plan(
            request,
            (AgentTool("lookup", "Changed description.", Args, execute),),
        )


def test_pinned_model_profile_preserves_unverified_reasoning_semantics() -> None:
    pinned = PinnedModelProfile(
        role="query",
        fingerprint=_fingerprint("query"),
        profile=ModelProfile(
            context_window_tokens=10_000,
            reasoning=best_effort_reasoning_profile("openrouter"),
        ),
    )

    restored = PinnedModelProfile.from_json(pinned.as_json())

    assert restored == pinned
    assert restored.profile.reasoning is not None
    assert restored.profile.reasoning.best_effort is True


def test_execution_rejects_changed_context_or_model_pins() -> None:
    from dlightrag.engine.runtime import IncompatibleActiveRunError

    pins = tuple(
        PinnedModelProfile(
            role=role,
            fingerprint=_fingerprint(role),
            profile=ModelProfile(context_window_tokens=10_000),
        )
        for role in ("extract", "keyword", "query", "vlm")
    )
    executor = _executor()
    request = MagicMock(
        pinned_models=pins,
        context_policy_revision=CONTEXT_POLICY_REVISION,
        model_catalog_revision=current_model_catalog_revision(),
    )
    executor.validate_pinned_model_profiles(request)

    request.model_catalog_revision = "stale-catalog"
    with pytest.raises(IncompatibleActiveRunError, match="model catalog"):
        executor.validate_pinned_model_profiles(request)

    request.model_catalog_revision = current_model_catalog_revision()
    request.context_policy_revision = "stale-policy"
    with pytest.raises(IncompatibleActiveRunError, match="context policy"):
        executor.validate_pinned_model_profiles(request)

    request.context_policy_revision = CONTEXT_POLICY_REVISION
    mismatched = _executor()
    mismatched._model_fingerprint_for_role = lambda role: ModelFingerprint(
        "other", f"test-{role}", None
    )
    with pytest.raises(IncompatibleActiveRunError, match="model endpoint"):
        mismatched.validate_pinned_model_profiles(request)


def _resource_resolver() -> AnswerResourceResolver:
    capabilities = MagicMock()
    capabilities.refresh_answer = AsyncMock(
        return_value=AnswerCapabilities(
            answer=AnswerImageCapability(
                status="supported",
                configured_ceiling=3,
                effective_max_images=3,
                provider="test",
                base_url=None,
                model="test-model",
                failure_kind=None,
            ),
            vlm_status="unknown",
        )
    )
    return AnswerResourceResolver(
        settings=AnswerResourceSettings(
            max_attachments=6,
            max_attachment_bytes=10_000_000,
            max_total_attachment_bytes=20_000_000,
            image_max_bytes=5_000_000,
            image_max_pixels=4_000_000,
        ),
        models=MagicMock(),
        capabilities=capabilities,
        resource_identity_secret=b"i" * 32,
        resource_cursor_secret=b"c" * 32,
    )


def test_resource_identity_is_stable_within_run_and_isolated_across_runs() -> None:
    resolver = _resource_resolver()
    resources = [ResourceInput(url="https://example.com/report")]
    profile = ModelProfile(context_window_tokens=10_000, supports_images=False)

    first, _ = resolver.build_resource_context(
        resources,
        text_window_budget=TextWindowBudget(tokens=100),
        vlm_profile=profile,
        resource_scope="owner-a\0run-1",
    )
    again, _ = resolver.build_resource_context(
        resources,
        text_window_budget=TextWindowBudget(tokens=100),
        vlm_profile=profile,
        resource_scope="owner-a\0run-1",
    )
    other, _ = resolver.build_resource_context(
        resources,
        text_window_budget=TextWindowBudget(tokens=100),
        vlm_profile=profile,
        resource_scope="owner-a\0run-2",
    )

    assert first is not None and again is not None and other is not None
    assert first.manifest()[0].resource_id == again.manifest()[0].resource_id
    assert first.manifest()[0].resource_id != other.manifest()[0].resource_id


def _png_bytes(color: str = "white") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _multimodal_resolver(
    capability: AnswerImageCapability,
    *,
    policy_overrides: Mapping[str, int] | None = None,
) -> AnswerResourceResolver:
    models = MagicMock()
    models.web_sources.return_value = None
    models.vlm_func.return_value = AsyncMock()
    capabilities = MagicMock()
    overrides = dict(policy_overrides or {})
    capabilities.answer_image_policy.side_effect = lambda profile: answer_image_policy(
        max_images=capability.configured_ceiling if profile.supports_images else 0,
        **overrides,
    )
    capabilities.vlm_image_policy.side_effect = lambda profile: answer_image_policy(
        max_images=capability.configured_ceiling if profile.supports_images else 0,
        **overrides,
    )
    return AnswerResourceResolver(
        settings=AnswerResourceSettings(
            max_attachments=6,
            max_attachment_bytes=10_000_000,
            max_total_attachment_bytes=20_000_000,
            image_max_bytes=5_000_000,
            image_max_pixels=4_000_000,
        ),
        models=models,
        capabilities=capabilities,
    )


def _image_capability(
    status: str,
    *,
    configured_ceiling: int = 3,
) -> AnswerImageCapability:
    return AnswerImageCapability(
        status=cast(Any, status),
        configured_ceiling=configured_ceiling,
        effective_max_images=configured_ceiling if status == "supported" else 0,
        provider="test",
        base_url=None,
        model="test-model",
        failure_kind=None,
    )


def _request_models(*, query_images: bool, vlm_images: bool) -> RequestModelContext:
    return RequestModelContext(
        extract=ModelProfile(context_window_tokens=100_000),
        query=ModelProfile(context_window_tokens=100_000, supports_images=query_images),
        vlm=ModelProfile(context_window_tokens=100_000, supports_images=vlm_images),
    )


async def test_memory_recall_allowed_gating() -> None:
    """No settings checker keeps memory enabled; a false checker disables it."""
    assert await _memory_recall_allowed(None, owner_id="o") is True

    async def deny(**kwargs: Any) -> bool:
        del kwargs
        return False

    assert await _memory_recall_allowed(deny, owner_id="o") is False

    calls: list[str] = []

    async def allow(**kwargs: Any) -> bool:
        calls.append(kwargs["owner_id"])
        return True

    assert await _memory_recall_allowed(allow, owner_id="o") is True
    assert calls == ["o"]


async def test_child_model_calls_inherit_run_scheduler_ownership() -> None:
    scheduler = ModelScheduler(max_concurrency=1)
    first_started = asyncio.Event()
    second_queued = asyncio.Event()
    release_first = asyncio.Event()
    order: list[str] = []

    async def operation(label: str, *, block: bool = False) -> str:
        order.append(label)
        if block:
            first_started.set()
            await release_first.wait()
        return label

    async def execute(session: Any) -> RunExecutionOutcome:
        if session.run_id == "run-a":
            first = asyncio.create_task(scheduler.run(lambda: operation("a1", block=True)))
            await first_started.wait()
            second = asyncio.create_task(scheduler.run(lambda: operation("a2")))
            await asyncio.sleep(0)
            second_queued.set()
            await asyncio.gather(first, second)
            return Succeeded({"run": "a"})
        await scheduler.run(lambda: operation("b1"))
        return Succeeded({"run": "b"})

    executor = _executor()
    executor._execute = execute  # type: ignore[method-assign]
    run_a = asyncio.create_task(
        executor.execute(cast(RunSession, MagicMock(owner_id="owner", run_id="run-a")))
    )
    await second_queued.wait()
    run_b = asyncio.create_task(
        executor.execute(cast(RunSession, MagicMock(owner_id="owner", run_id="run-b")))
    )
    await asyncio.sleep(0)
    release_first.set()

    assert await asyncio.gather(run_a, run_b) == [
        Succeeded({"run": "a"}),
        Succeeded({"run": "b"}),
    ]
    assert order == ["a1", "b1", "a2"]


async def test_actionable_answer_errors_keep_their_public_message() -> None:
    executor = _executor()
    executor._execute = AsyncMock(  # type: ignore[method-assign]
        side_effect=CurrentDocumentParseError("report.pdf")
    )

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, MagicMock()))

    assert raised.value.kind == "CURRENT_DOCUMENT_PARSE_FAILED"
    assert "report.pdf" in raised.value.public_message


@pytest.mark.parametrize(
    ("error", "checkpoint_key"),
    [
        (CorpusUnavailableError("offline"), "corpus_unavailable_attempt"),
        (ProviderUnavailableError(), "providers_unavailable_attempt"),
    ],
)
async def test_transient_answer_dependency_interruption_defers_same_run(
    error: Exception,
    checkpoint_key: str,
) -> None:
    now = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
    executor = _executor()
    executor._now = lambda: now
    executor._execute = AsyncMock(side_effect=error)  # type: ignore[method-assign]
    session = MagicMock(
        owner_id="owner",
        run_id="same-run",
        checkpoint={checkpoint_key: 2},
    )
    session.check_cancelled = AsyncMock()
    session.reset_output = AsyncMock()

    outcome = await executor.execute(cast(RunSession, session))

    assert isinstance(outcome, Deferred)
    assert outcome.checkpoint == {checkpoint_key: 3}
    assert (outcome.next_attempt_at - now).total_seconds() == 20
    session.check_cancelled.assert_awaited_once_with()
    session.reset_output.assert_awaited_once_with()


async def test_unknown_errors_map_to_generic_public_message(
    caplog: pytest.LogCaptureFixture,
) -> None:
    executor = _executor()
    executor._execute = AsyncMock(  # type: ignore[method-assign]
        side_effect=RuntimeError("postgres://user:secret@host/db")
    )
    session = MagicMock(owner_id="owner", run_id="run-correlated")

    with pytest.raises(RunExecutionError) as raised:
        await executor.execute(cast(RunSession, session))

    assert raised.value.kind == "ANSWER_STREAM_FAILED"
    assert raised.value.public_message == "Answer run failed."
    assert "Answer run run-correlated execution failed" in caplog.text
    assert "postgres://user:secret@host/db" not in caplog.text


async def test_url_current_image_is_pinned_once_for_durable_replay() -> None:
    resolver = _resource_resolver()
    image_bytes = _png_bytes()
    inline_bytes = b"notes"
    resolver.materialize_link_image = AsyncMock(return_value=image_bytes)  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png",
                filename="chart.png",
                ordinal=0,
                mime_type=None,
            ),
        ),
        attachments=(
            AttachmentReference(
                digest=artifact_digest(inline_bytes),
                filename="notes.txt",
                mime_type="text/plain",
                ordinal=0,
            ),
        ),
    )

    pinned, artifacts = await resolver.pin_current_image_links(request, (inline_bytes,))

    assert pinned.links == ()
    assert [item.filename for item in pinned.attachments] == ["chart.png", "notes.txt"]
    assert [item.ordinal for item in pinned.attachments] == [0, 1]
    assert artifacts == [image_bytes, inline_bytes]
    resources = await build_current_answer_resources(
        links=pinned.links,
        attachments=pinned.attachments,
        attachment_loaders=[
            in_memory_attachment_loader(image_bytes),
            in_memory_attachment_loader(inline_bytes),
        ],
    )
    resolver.materialize_link_image = AsyncMock(  # type: ignore[method-assign]
        side_effect=AssertionError("durable replay must not refetch the URL")
    )
    images, _remaining, _image_resources = await resolver.prepare_current_images(resources)

    assert len(images) == 1
    resolver.materialize_link_image.assert_not_awaited()  # type: ignore[attr-defined]


async def test_research_multimodal_query_gets_all_raw_images_and_resource_handles() -> None:
    capability = _image_capability("supported", configured_ceiling=2)
    resolver = _multimodal_resolver(capability)
    models = _request_models(query_images=True, vlm_images=True)
    resources = [
        ResourceInput(filename="white.png", content=_png_bytes("white"), declared_mime="image/png"),
        ResourceInput(filename="black.png", content=_png_bytes("black"), declared_mime="image/png"),
    ]

    resolved = await resolver.resolve(
        resources,
        models=models,
        text_window_budget=TextWindowBudget(10_000),
        confirm_image_context=AsyncMock(return_value=(models, capability)),
        resolved_mode="research",
    )

    try:
        assert [block["type"] for block in resolved.query_images or ()] == [
            "text",
            "image_url",
            "text",
            "image_url",
        ]
        assert len(resolved.resource_manifest) == 2
        assert all(
            entry.resource_id in str(resolved.query_images) for entry in resolved.resource_manifest
        )
    finally:
        assert resolved.registry is not None
        await resolved.registry.aclose()


@pytest.mark.parametrize("query_status", ["unsupported", "unknown"])
async def test_research_text_query_with_inspect_gets_zero_raw_and_all_handles(
    query_status: str,
) -> None:
    capability = _image_capability(query_status, configured_ceiling=2)
    resolver = _multimodal_resolver(capability)
    models = _request_models(query_images=False, vlm_images=True)
    resources = [
        ResourceInput(filename="white.png", content=_png_bytes("white"), declared_mime="image/png"),
        ResourceInput(filename="black.png", content=_png_bytes("black"), declared_mime="image/png"),
    ]

    resolved = await resolver.resolve(
        resources,
        models=models,
        text_window_budget=TextWindowBudget(10_000),
        confirm_image_context=AsyncMock(return_value=(models, capability)),
        resolved_mode="research",
    )

    try:
        assert resolved.query_images is None
        assert len(resolved.resource_manifest) == 2
        assert "inspect" in {tool.name for tool in resolved.resource_tools}
    finally:
        assert resolved.registry is not None
        await resolved.registry.aclose()


async def test_research_inspect_still_enforces_configured_image_count() -> None:
    capability = _image_capability("unsupported", configured_ceiling=1)
    resolver = _multimodal_resolver(capability)
    models = _request_models(query_images=False, vlm_images=True)

    with pytest.raises(CurrentImagePayloadError, match="at most 1"):
        await resolver.resolve(
            [
                ResourceInput(
                    filename="white.png",
                    content=_png_bytes("white"),
                    declared_mime="image/png",
                ),
                ResourceInput(
                    filename="black.png",
                    content=_png_bytes("black"),
                    declared_mime="image/png",
                ),
            ],
            models=models,
            text_window_budget=TextWindowBudget(10_000),
            confirm_image_context=AsyncMock(return_value=(models, capability)),
            resolved_mode="research",
        )


async def test_research_current_image_budget_is_all_or_error() -> None:
    first = _png_bytes("white")
    capability = _image_capability("supported", configured_ceiling=2)
    resolver = _multimodal_resolver(
        capability,
        policy_overrides={
            "max_total_bytes": len(first),
            "max_bytes_per_image": len(first),
        },
    )
    models = _request_models(query_images=True, vlm_images=True)

    with pytest.raises(CurrentImagePayloadError, match="query_image_2"):
        await resolver.resolve(
            [
                ResourceInput(filename="one.png", content=first, declared_mime="image/png"),
                ResourceInput(
                    filename="two.png",
                    content=_png_bytes("black"),
                    declared_mime="image/png",
                ),
            ],
            models=models,
            text_window_budget=TextWindowBudget(10_000),
            confirm_image_context=AsyncMock(return_value=(models, capability)),
            resolved_mode="research",
        )


async def test_unavailable_url_image_rejects_the_whole_current_request() -> None:
    resolver = _resource_resolver()
    materialize = AsyncMock(return_value=None)
    resolver.materialize_link_image = materialize  # type: ignore[method-assign]
    request = AnswerRunRequest(
        query="inspect",
        links=(
            LinkReference(
                url="https://example.com/chart.png?version=1",
                filename=None,
                ordinal=0,
                mime_type=None,
            ),
        ),
    )

    with pytest.raises(CurrentImagePayloadError, match="could not be fetched and verified"):
        await resolver.pin_current_image_links(request, ())

    materialize.assert_awaited_once()


async def test_stream_close_failure_does_not_skip_registry_close(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self) -> str:
            raise StopAsyncIteration

        async def aclose(self) -> None:
            raise RuntimeError("stream close failed")

    registry = MagicMock(aclose=AsyncMock())

    await _close_execution_resources(Stream(), registry)

    registry.aclose.assert_awaited_once()
    assert "Failed to close Answer stream" in caplog.text


async def test_durable_child_usage_aggregates_roster_rows() -> None:
    from dlightrag.engine.answer.research.runtime import _durable_child_usage

    store = MagicMock()
    store.list_child_sessions = AsyncMock(
        return_value=(
            {"usage": {"input_tokens": 3, "output_tokens": 2}},
            {"usage": {"input_tokens": 5, "output_tokens": 1}},
            {"usage": None},
        )
    )

    assert await _durable_child_usage(store, owner_id="owner", run_id="run-1") == {
        "input_tokens": 8,
        "output_tokens": 3,
    }
