# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for MCP workspace lifecycle tools."""

import datetime
import json
import logging
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from mcp import Client, MCPError
from mcp.types import INVALID_PARAMS, CallToolResult, InputRequiredResult, TextContent

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.application.access import (
    RequestScope,
    owner_id_from_principal,
    request_scope_context,
)
from dlightrag.application.config import (
    AccessControlConfig,
    AccessControlRuleConfig,
    DlightragConfig,
)
from dlightrag.application.corpus_admin import (
    FilePanelCursorCodec,
    IngestSpec,
    WorkspaceCatalogPage,
)
from dlightrag.application.retrieval import (
    RetrieveProjection,
    RetrieveResponse,
    restore_retrieval_result,
)
from dlightrag.application.retrieval._answer_projection import project_answer_retrieval
from dlightrag.application.runs import RunView
from dlightrag.engine.runtime.records import (
    RunAccessScope,
    RunRecord,
)
from tests.config_helpers import mutate_config, replace_config
from tests.unit.conftest import answer_capability_view

_IMAGE_BLOCK = {"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}


def _project_stored_retrieval(
    stored: dict[str, Any], projection: RetrieveProjection
) -> RetrieveResponse:
    result = restore_retrieval_result(stored)
    projected = project_answer_retrieval(result, projection=projection)
    return RetrieveResponse(
        contexts=projected.contexts,
        sources=projected.sources,
        trace=result.trace,
        image_descriptions=tuple(result.image_descriptions),
    )


def _completed_tool_result(result: CallToolResult | InputRequiredResult) -> CallToolResult:
    assert isinstance(result, CallToolResult)
    return result


def _tool_text(result: CallToolResult | InputRequiredResult) -> str:
    result = _completed_tool_result(result)
    assert result.content
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


def _tool_json(result: CallToolResult | InputRequiredResult) -> Any:
    return json.loads(_tool_text(result))


@pytest.fixture
def mock_mcp_application(monkeypatch, test_config: DlightragConfig):
    application = AsyncMock()
    application.config = test_config
    application.corpora = SimpleNamespace(
        list_workspaces=AsyncMock(return_value=["default"]),
        alist_workspace_records=AsyncMock(return_value=[{"workspace": "default"}]),
        list_workspace_records_page=AsyncMock(
            return_value=WorkspaceCatalogPage(
                items=(
                    {
                        "workspace": "default",
                        "display_name": "default",
                        "embedding_model": "voyage-multimodal-3.5",
                        "created_at": None,
                        "updated_at": None,
                    },
                ),
                next_cursor=None,
                fetched_rows=1,
            )
        ),
        create_workspace=AsyncMock(),
        file_panel_cursor_codec=FilePanelCursorCodec(b"mcp-file-cursor-secret"),
        file_panel_snapshot=AsyncMock(
            return_value={
                "files": [],
                "next_cursor": None,
                "fetched_rows": 0,
            }
        ),
    )
    corpus_run = _run_record(run_kind="corpus_mutation")
    application.corpus_mutations = SimpleNamespace(
        create_ingest=AsyncMock(return_value=SimpleNamespace(run=corpus_run, replayed=False)),
        create_delete=AsyncMock(return_value=SimpleNamespace(run=corpus_run, replayed=False)),
        create_retry=AsyncMock(return_value=SimpleNamespace(run=corpus_run, replayed=False)),
        create_reset=AsyncMock(return_value=SimpleNamespace(run=corpus_run, replayed=False)),
    )
    application.retrieval = SimpleNamespace(
        create=AsyncMock(
            return_value=SimpleNamespace(run=_run_record(run_kind="retrieval"), replayed=False)
        ),
        project_stored=Mock(side_effect=_project_stored_retrieval),
    )
    capability_view = answer_capability_view()
    run_get = AsyncMock(return_value=_run_record())
    application.runs = SimpleNamespace(
        get=run_get,
        get_global=AsyncMock(
            side_effect=lambda **_kwargs: (
                RunView.from_runtime(run_get.return_value)
                if run_get.return_value is not None
                else None
            )
        ),
        list=AsyncMock(return_value=(_run_record(),)),
        cancel=AsyncMock(
            return_value=SimpleNamespace(outcome="cancelled", run=_run_record(status="cancelled"))
        ),
        resume_repair=AsyncMock(return_value=True),
    )
    application.answers = SimpleNamespace(
        create=AsyncMock(return_value=SimpleNamespace(run=_run_record(), replayed=False)),
        capabilities=capability_view.read,
        list_artifacts=AsyncMock(return_value=()),
        read_artifact=AsyncMock(return_value=None),
        steer=AsyncMock(return_value=None),
        continuation_workspaces=AsyncMock(return_value=None),
        follow_up=AsyncMock(return_value=None),
        fork=AsyncMock(return_value=None),
        transcript_tail=AsyncMock(return_value=None),
        children=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(mcp_server, "_ensure_application", AsyncMock(return_value=application))
    monkeypatch.setattr(mcp_server, "create_application", AsyncMock(return_value=application))
    return application


_RUN_ID = "019893f4-0000-7000-8000-000000000001"
_CREATED_AT = datetime.datetime(2026, 8, 12, tzinfo=datetime.UTC)
#: MCP with ``auth_mode="none"`` collapses callers into the deployment owner.
_EXPECTED_OWNER = owner_id_from_principal(auth_mode="none", user_id="anonymous")


def _run_record(
    *,
    status: str = "queued",
    result: dict[str, Any] | None = None,
    cancel_requested: bool = False,
    error_kind: str | None = None,
    error_message: str | None = None,
    run_kind: str = "answer",
    owner_id: str = _EXPECTED_OWNER,
    phase: str | None = None,
    checkpoint: dict[str, Any] | None = None,
) -> RunRecord:
    terminal = status in ("succeeded", "failed", "cancelled")
    return RunRecord(
        run_id=_RUN_ID,
        run_kind=run_kind,  # type: ignore[arg-type]
        lane="corpus_mutation" if run_kind == "corpus_mutation" else "query",
        submitted_by=owner_id,
        access_scope=(
            RunAccessScope(kind="workspace", scope_id="default")
            if run_kind == "corpus_mutation"
            else RunAccessScope(kind="owner", scope_id=owner_id)
        ),
        submission_key=None or str(_RUN_ID),
        request_fingerprint="test-fingerprint",
        prepared_input=(
            {"action": "ingest", "workspace": "default"}
            if run_kind == "corpus_mutation"
            else {"query": "Follow up", "workspaces": ["default"]}
        ),
        status=status,  # type: ignore[arg-type]
        phase=phase,
        stop_reason=None,
        cancel_requested_at=_CREATED_AT if cancel_requested else None,
        lease_owner=None,
        lease_expires_at=None,
        fencing_epoch=0,
        durable_progress_version=0,
        last_reclaim_progress_version=0,
        reclaims_without_progress=0,
        next_event_sequence=1,
        events_trimmed_at=None,
        result=result,
        error_kind=error_kind,
        error_message=error_message,
        created_at=_CREATED_AT,
        checkpoint=checkpoint,
        updated_at=_CREATED_AT,
        started_at=None,
        finished_at=_CREATED_AT if terminal else None,
    )


def _stored_result() -> dict[str, Any]:
    return {
        "answer": "Answer [1-1].",
        "contexts": {
            "chunks": [
                {
                    "chunk_id": "c1",
                    "reference_id": "1",
                    "file_path": "/private/report.pdf",
                    "content": "Evidence",
                    "image_data": "base64-payload",
                    "_workspace": "default",
                }
            ]
        },
        "sources": [
            {
                "id": "1",
                "title": "report.pdf",
                "type": "document",
                "source_uri": "local://default/report.pdf",
                "workspace": "default",
                "document_id": "doc-report",
                "chunks": [],
            }
        ],
        "evidence_images": [],
        "artifacts": [],
        "artifact_outcome": {"status": "complete", "issues": []},
        "trace": {},
        "image_descriptions": [],
    }


async def test_get_capabilities_reports_answer_image_capability(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.engine.answer.image_capability import AnswerImageCapability

    mock_mcp_application.answers.capabilities = answer_capability_view(
        AnswerImageCapability(
            status="supported",
            configured_ceiling=8,
            effective_max_images=6,
            provider="test",
            base_url=None,
            model="test-model",
            failure_kind=None,
        )
    ).read

    result = await mcp_server.mcp_app.call_tool("get_capabilities", {})

    cap = _tool_json(result)["answer_image_capability"]
    assert cap["status"] == "supported"
    assert cap["effective_max_images"] == 6
    assert cap["configured_ceiling"] == 8
    assert cap["model"] == "test-model"


async def test_list_answer_artifacts_uses_canonical_semantic_descriptors(
    mock_mcp_application: AsyncMock,
) -> None:
    stored = _stored_result()
    stored["artifacts"] = [
        {
            "resource_id": "artifact-report",
            "media_type": "text/markdown",
            "label": "Quarterly report",
            "filename": "report.md",
            "byte_size": 42,
            "digest": "a" * 64,
            "presentation": "markdown",
            "status": "available",
        }
    ]
    stored["artifact_outcome"] = {"status": "partial", "issues": []}
    mock_mcp_application.runs.get.return_value = _run_record(status="succeeded", result=stored)

    result = await mcp_server.mcp_app.call_tool("list_answer_artifacts", {"run_id": _RUN_ID})

    payload = _tool_json(result)
    descriptor = payload["artifacts"][0]
    assert descriptor == {
        "resource_id": "artifact-report",
        "media_type": "text/markdown",
        "label": "Quarterly report",
        "filename": "report.md",
        "byte_size": 42,
        "digest": "a" * 64,
        "presentation": "markdown",
        "status": "available",
        "uri": f"dlightrag://answer/{_RUN_ID}/artifacts/artifact-report",
        "width": None,
        "height": None,
        "issue": None,
        "data_url": None,
        "download_url": None,
        "presentation_url": None,
    }
    assert payload["artifact_outcome"] == {"status": "partial", "issues": []}
    assert "kind" not in descriptor


async def test_list_answer_artifacts_does_not_invent_an_in_flight_outcome(
    mock_mcp_application: AsyncMock,
) -> None:
    mock_mcp_application.runs.get.return_value = _run_record(status="running", result=None)

    result = await mcp_server.mcp_app.call_tool("list_answer_artifacts", {"run_id": _RUN_ID})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == (
        "Error: Answer artifacts are not available until the run has a stored result"
    )


async def test_mcp_v2_client_lists_and_calls_tools(mock_mcp_application: AsyncMock) -> None:
    async with Client(mcp_server.mcp_app) as client:
        listing = await client.list_tools()
        result = await client.call_tool("list_runs", {})

    tool_names = {tool.name for tool in listing.tools}
    assert {
        "retrieve",
        "ingest",
        "retry_files",
        "delete_files",
        "reset_corpus",
        "get_run",
    } <= tool_names
    assert "get_ingest_job" not in tool_names
    assert result.is_error is False
    assert _tool_json(result)["runs"][0]["run_id"] == _RUN_ID


async def test_mcp_internal_errors_do_not_leak_details(mock_mcp_application: AsyncMock) -> None:
    mock_mcp_application.corpora.list_workspace_records_page.side_effect = RuntimeError(
        "database-secret"
    )

    result = await mcp_server.mcp_app.call_tool("list_workspaces", {})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == "Error: internal tool failure"
    assert "database-secret" not in _tool_text(result)


async def test_mcp_protocol_errors_remain_protocol_errors() -> None:
    app = mcp_server.DlightRAGMCPServer("probe")

    @app.tool()
    async def reject() -> None:
        raise MCPError(INVALID_PARAMS, "invalid request")

    with pytest.raises(MCPError, match="invalid request"):
        await app.call_tool("reject", {})


async def test_mcp_lists_workspace_lifecycle_tools() -> None:
    tools = await mcp_server.mcp_app.list_tools()
    names = {tool.name for tool in tools}

    assert names == {
        "answer",
        "cancel_run",
        "create_workspace",
        "delete_files",
        "get_run",
        "get_capabilities",
        "get_workspace_storage_status",
        "ingest",
        "forget_memory",
        "remember_memory",
        "remove_model_catalogue_entry",
        "undo_memory_change",
        "upsert_model_catalogue_entry",
        "get_memory_settings",
        "get_model_catalogue",
        "set_memory_enabled",
        "clear_memory",
        "follow_up_answer_run",
        "fork_answer_run",
        "get_answer_transcript",
        "list_answer_artifacts",
        "list_answer_children",
        "list_runs",
        "list_memories",
        "list_files",
        "list_workspaces",
        "read_answer_artifact",
        "reset_corpus",
        "resume_corpus_run",
        "retrieve",
        "retry_files",
        "steer_answer_run",
    }
    answer_tool = next(tool for tool in tools if tool.name == "answer")
    answer_props = answer_tool.input_schema["properties"]
    assert {
        "query",
        "history",
        "mode",
        "attachments",
        "filters",
        "chunk_top_k",
        "idempotency_key",
    } <= answer_props.keys()
    assert "query_images" not in answer_props
    # Creation tools start durable work and return; common tools read and stop it.
    assert answer_tool.annotations is not None
    assert answer_tool.annotations.read_only_hint is False
    retrieve_tool = next(tool for tool in tools if tool.name == "retrieve")
    assert retrieve_tool.annotations is not None
    assert retrieve_tool.annotations.read_only_hint is False
    assert "idempotency_key" in retrieve_tool.input_schema["properties"]
    for name in ("get_run", "cancel_run"):
        tool = next(item for item in tools if item.name == name)
        assert set(tool.input_schema["properties"]) == {"run_id"}
        assert tool.input_schema["properties"]["run_id"]["description"]
        assert tool.description and "durable run" in tool.description
    ingest_tool = next(tool for tool in tools if tool.name == "ingest")
    ingest_props = ingest_tool.input_schema["properties"]
    assert {"source_type", "path", "url", "documents", "metadata"} <= ingest_props.keys()
    delete_files_tool = next(tool for tool in tools if tool.name == "delete_files")
    assert "idempotency_key" in delete_files_tool.input_schema["properties"]
    assert "document_ids" in delete_files_tool.input_schema["properties"]
    assert "dry_run" not in delete_files_tool.input_schema["properties"]
    retry_files_tool = next(tool for tool in tools if tool.name == "retry_files")
    assert {"document_ids", "selector", "workspace", "idempotency_key"} == set(
        retry_files_tool.input_schema["properties"]
    )


def test_mcp_security_defaults_are_loopback_only() -> None:
    cfg = cast(Any, DlightragConfig)()

    assert cfg.interfaces.mcp.allowed_hosts == ("127.0.0.1:*", "localhost:*", "[::1]:*")
    assert cfg.interfaces.mcp.allowed_origins == (
        "http://127.0.0.1:*",
        "http://localhost:*",
        "http://[::1]:*",
    )


async def test_mcp_rejects_unknown_mode_without_schema_wrapper(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x", "mode": "mix"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert "Error:" in _tool_text(result)
    assert "mode" in _tool_text(result)
    mock_mcp_application.answers.create.assert_not_awaited()


@pytest.mark.parametrize(
    ("tool_name", "payload", "error_fragment"),
    [
        (
            "retrieve",
            {
                "query": "x",
                "query_images": [_IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK, _IMAGE_BLOCK],
            },
            "query_images",
        ),
        (
            "retrieve",
            {"query": "x", "query_images": ["data:image/png;base64,abc"]},
            "valid dictionary",
        ),
    ],
)
async def test_mcp_rejects_invalid_query_image_payloads(
    mock_mcp_application,
    tool_name: str,
    payload: dict[str, Any],
    error_fragment: str,
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        tool_name,
        payload,
    )

    assert "Error:" in _tool_text(result)
    assert error_fragment in _tool_text(result)
    mock_mcp_application.retrieval.create.assert_not_awaited()


async def test_mcp_retrieve_forwards_chunk_top_k(mock_mcp_application) -> None:
    await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "top_k": 8, "chunk_top_k": 5},
    )

    await_args = mock_mcp_application.retrieval.create.await_args
    assert await_args is not None
    request = await_args.kwargs["request"]
    assert request.top_k == 8
    assert request.chunk_top_k == 5


async def test_mcp_retrieve_returns_run_descriptor(mock_mcp_application: AsyncMock) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "retrieve", {"query": "x", "idempotency_key": "retrieval-1"}
    )

    body = _tool_json(result)
    assert body["run_id"] == _RUN_ID
    assert body["run_kind"] == "retrieval"
    assert body["lane"] == "query"
    call = mock_mcp_application.retrieval.create.await_args
    assert call is not None
    assert call.kwargs["request"].workspaces == ("default",)
    assert call.kwargs["owner_id"] == _EXPECTED_OWNER
    assert call.kwargs["idempotency_key"] == "retrieval-1"


async def test_mcp_jwt_claims_access_control_denies_unmapped_workspace(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    mutate_config(test_config, "access.auth_mode", "jwt")
    mutate_config(test_config, "access.jwt_verification_key", "test-key")
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=["finance"],
                    actions=["workspace.query"],
                )
            ],
        ),
    )

    with request_scope_context(
        RequestScope(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": ["legal-rag-readers"]},
        )
    ):
        result = await mcp_server.mcp_app.call_tool(
            "retrieve",
            {"query": "x", "workspaces": ["finance"]},
        )

    assert "Access denied" in _tool_text(result)
    mock_mcp_application.retrieval.create.assert_not_awaited()


async def test_mcp_query_permission_does_not_imply_visual_asset_permission(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=["default"],
                    actions=["workspace.query"],
                )
            ],
        ),
    )
    mock_mcp_application.runs.get.return_value = _run_record(
        run_kind="retrieval",
        status="succeeded",
        owner_id=owner_id_from_principal(auth_mode="jwt", user_id="alice"),
        result={
            "contexts": {
                "chunks": [
                    {
                        "chunk_id": "figure-1",
                        "reference_id": "1",
                        "full_doc_id": "doc-report",
                        "file_path": "report.pdf",
                        "content": "Evidence",
                        "_has_visual_asset": True,
                        "_workspace": "default",
                        "metadata": {
                            "source_uri": "local://default/report.pdf",
                            "source_download_locator": "report.pdf",
                            "source_file_name": "report.pdf",
                        },
                    }
                ]
            },
            "trace": {},
            "image_descriptions": [],
        },
    )

    with request_scope_context(
        RequestScope(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": ["finance-rag-readers"]},
        )
    ):
        result = await mcp_server.mcp_app.call_tool("get_run", {"run_id": _RUN_ID})

    chunk = _tool_json(result)["result"]["contexts"]["chunks"][0]
    assert chunk["content"] == "Evidence"
    assert "image_url" not in chunk
    assert "thumbnail_url" not in chunk


async def test_mcp_retrieve_all_workspaces_uses_visible_records(mock_mcp_application) -> None:
    mock_mcp_application.corpora.alist_workspace_records.return_value = [
        {"workspace": "default"},
        {"workspace": "research_notes"},
    ]
    await mcp_server.mcp_app.call_tool(
        "retrieve",
        {"query": "x", "all_workspaces": True},
    )

    request = mock_mcp_application.retrieval.create.await_args.kwargs["request"]
    assert request.workspaces == ("default", "research_notes")


async def test_mcp_all_workspaces_rejects_empty_authorized_set(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    mutate_config(test_config, "access.control", AccessControlConfig(mode="jwt_claims", rules=[]))

    with request_scope_context(RequestScope(user_id="alice", auth_mode="jwt")):
        result = await mcp_server.mcp_app.call_tool(
            "answer",
            {"query": "x", "all_workspaces": True},
        )

    assert "No workspaces" in _tool_text(result)
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_all_workspaces_is_relative_to_query_authorization(
    mock_mcp_application,
    test_config: DlightragConfig,
) -> None:
    registered = [f"ws_{index:02d}" for index in range(14)]
    allowed = registered[:10]
    test_config = replace_config(
        test_config,
        "access.control",
        AccessControlConfig(
            mode="jwt_claims",
            rules=[
                AccessControlRuleConfig(
                    claim="groups",
                    value="finance-rag-readers",
                    workspaces=allowed,
                    actions=["workspace.query"],
                )
            ],
        ),
    )
    mock_mcp_application.corpora.alist_workspace_records.return_value = [
        {"workspace": workspace} for workspace in registered
    ]
    with request_scope_context(
        RequestScope(
            user_id="alice",
            auth_mode="jwt",
            claims={"groups": ["finance-rag-readers"]},
        )
    ):
        await mcp_server.mcp_app.call_tool(
            "retrieve",
            {"query": "x", "all_workspaces": True},
        )

    request = mock_mcp_application.retrieval.create.await_args.kwargs["request"]
    assert request.workspaces == tuple(allowed)


async def test_mcp_rejects_unknown_argument_without_echoing_the_url(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "url",
            "url": "https://fetch.example.com/file?signature=secret",
            "no_such_option": "loose",
        },
    )

    assert "Error:" in _tool_text(result)
    # A rejected call must not replay the caller's signed URL back to the model.
    assert "signature=secret" not in _tool_text(result)


async def test_mcp_rejects_mutually_exclusive_s3_key_and_prefix(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "s3", "bucket": "b", "s3_key": "a.pdf", "prefix": "docs/"},
    )

    assert "Error:" in _tool_text(result)
    assert "mutually exclusive" in _tool_text(result)


async def test_mcp_create_workspace_uses_corpus_catalog(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "create_workspace",
        {"workspace": "New Workspace", "display_name": "New Workspace"},
    )

    body = _tool_json(result)
    assert body == {
        "workspace": "new_workspace",
        "display_name": "New Workspace",
        "created": True,
    }
    mock_mcp_application.corpora.create_workspace.assert_awaited_once_with(
        "new_workspace",
        display_name="New Workspace",
    )


async def test_mcp_reset_corpus_accepts_a_durable_run(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "reset_corpus",
        {"workspace": "Old Workspace"},
    )

    body = _tool_json(result)
    assert body["run_kind"] == "corpus_mutation"
    mock_mcp_application.corpus_mutations.create_reset.assert_awaited_once_with(
        workspace="old_workspace",
        submitted_by=_EXPECTED_OWNER,
        supersedes_run_id=None,
        idempotency_key=None,
    )


async def test_mcp_corpus_mutation_projects_the_admission_limit(mock_mcp_application) -> None:
    from dlightrag.application.runs import RunAdmissionLimitExceededError

    mock_mcp_application.corpus_mutations.create_reset.side_effect = RunAdmissionLimitExceededError(
        "limit reached"
    )

    result = await mcp_server.mcp_app.call_tool("reset_corpus", {"workspace": "default"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == "Error: Deployment-wide nonterminal admission limit reached"


async def test_mcp_rejects_local_path_outside_input_dir(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "path": "/tmp/report.pdf"},
    )

    assert "relative to input_dir" in _tool_text(result)
    mock_mcp_application.corpus_mutations.create_ingest.assert_not_awaited()


async def test_mcp_rejects_local_path_traversal(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "local",
            "path": "../default/report.pdf",
            "workspace": "finance",
        },
    )

    assert "relative to input_dir" in _tool_text(result)
    mock_mcp_application.corpus_mutations.create_ingest.assert_not_awaited()


async def test_mcp_remote_prefix_ingest_accepts_a_run(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "ingest",
        {
            "source_type": "s3",
            "bucket": "bucket",
            "prefix": "docs/",
            "workspace": "default",
        },
    )

    assert _tool_json(result)["run_kind"] == "corpus_mutation"
    mock_mcp_application.corpus_mutations.create_ingest.assert_awaited_once_with(
        workspace="default",
        spec=IngestSpec(source_type="s3", bucket="bucket", prefix="docs/"),
        submitted_by=_EXPECTED_OWNER,
        idempotency_key=None,
    )


async def test_mcp_requests_stay_bound_to_running_application_config(
    mock_mcp_application,
    test_config: DlightragConfig,
    tmp_path,
) -> None:
    from dlightrag.application.config import set_config

    application_config = test_config.model_copy(
        update={
            "deployment": test_config.deployment.model_copy(
                update={
                    "workspace": "Application Workspace",
                    "working_dir": str((tmp_path / "application-storage").resolve()),
                }
            )
        }
    )
    global_config = test_config.model_copy(
        update={
            "deployment": test_config.deployment.model_copy(
                update={
                    "workspace": "Global Workspace",
                    "working_dir": str((tmp_path / "global-storage").resolve()),
                }
            )
        }
    )
    mock_mcp_application.config = application_config
    set_config(global_config)

    await mcp_server.mcp_app.call_tool(
        "ingest",
        {"source_type": "local", "path": "report.pdf"},
    )

    expected_path = str(
        (application_config.input_dir_path / "application_workspace" / "report.pdf").resolve()
    )
    mock_mcp_application.corpus_mutations.create_ingest.assert_awaited_once_with(
        workspace="application_workspace",
        spec=IngestSpec(source_type="local", path=expected_path),
        submitted_by=_EXPECTED_OWNER,
        idempotency_key=None,
    )


async def test_mcp_answer_returns_a_descriptor_without_waiting(
    mock_mcp_application: AsyncMock,
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {
            "query": "Follow up",
            "workspaces": ["default"],
            "top_k": 8,
            "chunk_top_k": 12,
            "attachments": [{"url": "https://example.com/report.pdf", "filename": "report.pdf"}],
            "filters": {"title": "Manual"},
            "semantic_highlights": True,
            "idempotency_key": "key-1",
        },
    )

    body = _tool_json(result)
    assert body == {
        "run_id": _RUN_ID,
        "run_kind": "answer",
        "lane": "query",
        "status": "queued",
        "cancel_requested": False,
        "parent_run_id": None,
        "continuation_kind": None,
        "created_at": _CREATED_AT.isoformat(),
    }
    # The tool call never holds the run open, so no answer text is returned here.
    assert "answer" not in body

    answer_request = mock_mcp_application.answers.create.await_args.kwargs["request"]
    call_kwargs = mock_mcp_application.answers.create.await_args.kwargs
    assert answer_request.workspaces == ("default",)
    assert answer_request.retrieval.top_k == 8
    assert answer_request.retrieval.chunk_top_k == 12
    assert answer_request.semantic_highlights is True
    assert call_kwargs["idempotency_key"] == "key-1"
    assert call_kwargs["owner_id"] == _EXPECTED_OWNER
    assert answer_request.filters is not None
    assert answer_request.filters.title == "Manual"
    resources = answer_request.resources
    assert [resource.url for resource in resources] == ["https://example.com/report.pdf"]
    assert resources[0].filename == "report.pdf"
    assert resources[0].content is None


async def test_mcp_answer_reports_a_reused_key_with_different_input(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.application.runs import IdempotencyKeyConflict

    mock_mcp_application.answers.create.side_effect = IdempotencyKeyConflict("reused")

    result = await mcp_server.mcp_app.call_tool(
        "answer", {"query": "x", "idempotency_key": "key-1"}
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert "idempotency_key" in _tool_text(result)


async def test_mcp_answer_projects_the_deployment_wide_admission_limit(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.application.runs import RunAdmissionLimitExceededError

    mock_mcp_application.answers.create.side_effect = RunAdmissionLimitExceededError(
        "limit reached"
    )

    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == "Error: Deployment-wide nonterminal admission limit reached"


async def test_mcp_status_returns_the_canonical_result_and_sanitizes_contexts(
    mock_mcp_application: AsyncMock,
) -> None:
    mock_mcp_application.runs.get.return_value = _run_record(
        status="succeeded",
        result=_stored_result(),
    )

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_run", {"run_id": _RUN_ID}))

    assert body["status"] == "succeeded"
    assert body["result"]["answer"] == "Answer [1-1]."
    assert body["result"]["contexts"]["chunks"][0]["image_url"] == "/images/default/c1?size=full"
    assert "image_data" not in body["result"]["contexts"]["chunks"][0]
    assert body["result"]["sources"][0]["source_uri"] == "local://default/report.pdf"
    assert body["result"]["sources"][0]["download_url"] is None
    assert {"workspace", "download_locator", "path", "url"}.isdisjoint(body["result"]["sources"][0])
    mock_mcp_application.runs.get_global.assert_awaited_once_with(run_id=_RUN_ID)


async def test_mcp_status_keeps_the_recorded_evidence_image_transport_state(
    mock_mcp_application: AsyncMock,
) -> None:
    """An image the answer model never received must not read as if it had."""
    stored = _stored_result()
    stored["evidence_images"] = [
        {
            "id": "c1",
            "chunk_id": "c1",
            "workspace": "default",
            "source_ref": "1-1",
            "label": "Figure 1",
            "answer_image_sent": False,
        }
    ]
    mock_mcp_application.runs.get.return_value = _run_record(status="succeeded", result=stored)

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_run", {"run_id": _RUN_ID}))

    assert body["result"]["evidence_images"][0]["answer_image_sent"] is False
    assert "answer_images" not in body["result"]


async def test_mcp_artifacts_use_stable_uris_without_browser_cookie_urls(
    mock_mcp_application: AsyncMock,
) -> None:
    stored = _stored_result()
    stored["answer"] = "[Notes](artifact:artifact-1)"
    stored["artifacts"] = [
        {
            "resource_id": "artifact-1",
            "role": "attachment",
            "media_type": "text/plain",
            "label": "Notes",
            "filename": "notes.txt",
            "byte_size": 5,
            "digest": "a" * 64,
            "presentation": "text",
            "status": "available",
        }
    ]
    stored["artifact_outcome"] = {"status": "complete", "issues": []}
    mock_mcp_application.runs.get.return_value = _run_record(status="succeeded", result=stored)

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_run", {"run_id": _RUN_ID}))

    artifact = body["result"]["artifacts"][0]
    assert artifact["uri"].startswith("dlightrag://answer/")
    assert artifact["data_url"] is None
    assert artifact["download_url"] is None
    assert body["result"]["parts"][0]["artifact"]["resource_id"] == "artifact-1"


async def test_mcp_status_reports_a_failed_run_with_its_public_error(
    mock_mcp_application: AsyncMock,
) -> None:
    mock_mcp_application.runs.get.return_value = _run_record(
        status="failed",
        error_kind="answer_stream_failed",
        error_message="Service error.",
    )

    body = _tool_json(await mcp_server.mcp_app.call_tool("get_run", {"run_id": _RUN_ID}))

    assert body["status"] == "failed"
    assert body["error_kind"] == "answer_stream_failed"
    assert body["error_message"] == "Service error."
    assert body["result"] is None


@pytest.mark.parametrize("tool", ["get_run", "cancel_run"])
async def test_mcp_never_reveals_another_owners_run(
    mock_mcp_application: AsyncMock, tool: str
) -> None:
    mock_mcp_application.runs.get.return_value = None
    mock_mcp_application.runs.cancel.return_value = SimpleNamespace(outcome="unknown", run=None)

    result = await mcp_server.mcp_app.call_tool(tool, {"run_id": _RUN_ID})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == f"Error: Run not found: {_RUN_ID}"


async def test_mcp_cancel_reports_the_pending_request(mock_mcp_application: AsyncMock) -> None:
    running = _run_record(status="running", cancel_requested=True)
    mock_mcp_application.runs.cancel.return_value = SimpleNamespace(outcome="pending", run=running)

    body = _tool_json(await mcp_server.mcp_app.call_tool("cancel_run", {"run_id": _RUN_ID}))

    assert body["status"] == "running"
    assert body["cancel_requested"] is True
    assert mock_mcp_application.runs.cancel.await_args.kwargs["owner_id"] == _EXPECTED_OWNER


async def test_mcp_resume_requeues_the_same_authorized_repair_run(
    mock_mcp_application: AsyncMock,
) -> None:
    waiting = RunView.from_runtime(
        _run_record(
            status="running",
            run_kind="corpus_mutation",
            phase="waiting_for_repair",
            checkpoint={
                "repair_reason": "Inspect upstream state.",
                "repair_remedy": "Repair it, then resume.",
            },
        )
    )
    queued = RunView.from_runtime(_run_record(run_kind="corpus_mutation"))
    mock_mcp_application.runs.get_global.side_effect = None
    mock_mcp_application.runs.get_global.return_value = waiting
    mock_mcp_application.runs.get.return_value = queued

    body = _tool_json(await mcp_server.mcp_app.call_tool("resume_corpus_run", {"run_id": _RUN_ID}))

    assert body["run_id"] == _RUN_ID
    assert body["status"] == "queued"
    mock_mcp_application.runs.resume_repair.assert_awaited_once_with(
        owner_id="default", run_id=_RUN_ID
    )


async def test_mcp_answer_preserves_answer_input_error_kind(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.engine.answer.errors import (
        ANSWER_INPUT_OVERFLOW,
        AnswerInputOverflowError,
    )

    mock_mcp_application.answers.create.side_effect = AnswerInputOverflowError(
        "The answer input is too large."
    )

    result = await mcp_server.mcp_app.call_tool("answer", {"query": "x"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == f"Error [{ANSWER_INPUT_OVERFLOW}]: The answer input is too large."


async def test_mcp_answer_reports_tool_misconfiguration_as_a_server_failure(
    mock_mcp_application: AsyncMock,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from dlightrag.engine.answer.errors import (
        INVALID_TOOL_CONFIGURATION,
        InvalidToolConfigurationError,
    )

    mock_mcp_application.answers.create.side_effect = InvalidToolConfigurationError(
        ("nonexistent_tool",)
    )

    with caplog.at_level(logging.WARNING):
        result = await mcp_server.mcp_app.call_tool("answer", {"query": "x"})

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert _tool_text(result) == (
        f"Error [{INVALID_TOOL_CONFIGURATION}]: Answer tooling is misconfigured."
    )
    assert "nonexistent_tool" not in _tool_text(result)
    assert [record for record in caplog.records if record.levelno >= logging.ERROR]


@pytest.mark.parametrize(
    "descriptor",
    [
        {"path": "/etc/passwd"},
        {"url": "https://example.com/x.pdf", "path": "/etc/passwd"},
        {"url": "https://example.com/x.pdf", "content": "aGVsbG8="},
        {"url": "ftp://example.com/x.pdf"},
    ],
)
async def test_mcp_answer_rejects_local_and_base64_attachments(
    mock_mcp_application, descriptor: dict[str, Any]
) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {"query": "x", "attachments": [descriptor]},
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_answer_enforces_link_count_limit(mock_mcp_application) -> None:
    mutate_config(mock_mcp_application.config, "answer.generation.max_attachments", 2)

    result = await mcp_server.mcp_app.call_tool(
        "answer",
        {
            "query": "x",
            "attachments": [{"url": f"https://example.com/{index}.pdf"} for index in range(3)],
        },
    )

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_answer_rejects_top_level_local_fields(mock_mcp_application) -> None:
    for field in ("path", "attachment_bytes", "attachment_base64"):
        result = await mcp_server.mcp_app.call_tool(
            "answer",
            {"query": "x", field: "value"},
        )
        assert isinstance(result, CallToolResult)
        assert result.is_error is True
    mock_mcp_application.answers.create.assert_not_awaited()


async def test_mcp_retry_files_accepts_an_exact_durable_cohort(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "retry_files",
        {"document_ids": ["doc-1", "doc-2"], "idempotency_key": "retry-docs"},
    )

    assert _tool_json(result)["run_kind"] == "corpus_mutation"
    mock_mcp_application.corpus_mutations.create_retry.assert_awaited_once_with(
        workspace="default",
        submitted_by=_EXPECTED_OWNER,
        document_ids=["doc-1", "doc-2"],
        selector=None,
        idempotency_key="retry-docs",
    )


async def test_mcp_delete_files_accepts_exact_document_ids(mock_mcp_application) -> None:
    result = await mcp_server.mcp_app.call_tool(
        "delete_files",
        {"document_ids": ["doc-report"], "idempotency_key": "delete-report"},
    )

    assert _tool_json(result)["run_kind"] == "corpus_mutation"
    mock_mcp_application.corpus_mutations.create_delete.assert_awaited_once_with(
        workspace="default",
        submitted_by=_EXPECTED_OWNER,
        filenames=(),
        file_paths=(),
        document_ids=["doc-report"],
        idempotency_key="delete-report",
    )


async def test_mcp_file_tools_canonicalize_display_workspace_before_access_and_manager(
    mock_mcp_application,
) -> None:
    mock_mcp_application.corpora.file_panel_snapshot.return_value = {
        "files": [],
        "next_cursor": None,
        "fetched_rows": 0,
    }
    listed = await mcp_server.mcp_app.call_tool(
        "list_files",
        {"workspace": "Finance Reports"},
    )
    deleted = await mcp_server.mcp_app.call_tool(
        "delete_files",
        {"workspace": "Finance Reports", "filenames": ["report.pdf"]},
    )

    assert _tool_json(listed)["workspace"] == "finance_reports"
    assert _tool_json(deleted)["run_kind"] == "corpus_mutation"
    mock_mcp_application.corpora.file_panel_snapshot.assert_awaited_once()
    assert mock_mcp_application.corpora.file_panel_snapshot.await_args.args == ("finance_reports",)
    mock_mcp_application.corpus_mutations.create_delete.assert_awaited_once_with(
        workspace="finance_reports",
        submitted_by=_EXPECTED_OWNER,
        filenames=["report.pdf"],
        file_paths=(),
        document_ids=(),
        idempotency_key=None,
    )


async def test_mcp_list_workspaces_returns_the_bounded_first_page(
    mock_mcp_application: AsyncMock,
) -> None:
    from dlightrag.application.corpus_admin import (
        WorkspaceCatalogCursor,
        WorkspaceCatalogPage,
    )

    mock_mcp_application.corpora.list_workspace_records_page = AsyncMock(
        return_value=WorkspaceCatalogPage(
            items=(
                {
                    "workspace": "default",
                    "display_name": "default",
                    "embedding_model": "voyage-multimodal-3.5",
                    "created_at": None,
                    "updated_at": None,
                },
            ),
            next_cursor=WorkspaceCatalogCursor(after_workspace="default"),
            fetched_rows=2,
        )
    )

    result = await mcp_server.mcp_app.call_tool("list_workspaces", {})

    assert isinstance(result, CallToolResult)
    assert result.is_error is False
    payload = _tool_json(result)
    assert payload["workspaces"] == ["default"]
    assert payload["has_more"] is True
    assert payload["records"][0]["display_name"] == "default"


@pytest.mark.parametrize(
    ("tool", "arguments", "expected"),
    [
        ("steer_answer_run", {"run_id": _RUN_ID, "instruction": "focus"}, "live Research"),
        (
            "follow_up_answer_run",
            {"run_id": _RUN_ID, "query": "next"},
            "terminal owned run",
        ),
        ("fork_answer_run", {"run_id": _RUN_ID, "query": "branch"}, "terminal owned run"),
        ("get_answer_transcript", {"run_id": _RUN_ID}, "Answer run not found"),
        ("list_answer_children", {"run_id": _RUN_ID}, "Answer run not found"),
        ("list_answer_artifacts", {"run_id": _RUN_ID}, "Answer run not found"),
        (
            "read_answer_artifact",
            {"run_id": _RUN_ID, "resource_id": "fabricated"},
            "artifact not found",
        ),
    ],
)
async def test_retrieval_run_id_is_unknown_to_answer_only_mcp_tools(
    mock_mcp_application: AsyncMock,
    tool: str,
    arguments: dict[str, Any],
    expected: str,
) -> None:
    retrieval = _run_record(
        run_kind="retrieval",
        status="succeeded",
        result={"answer": "fabricated"},
    )
    mock_mcp_application.runs.get.return_value = retrieval
    mock_mcp_application.answers.list_artifacts.return_value = None

    result = await mcp_server.mcp_app.call_tool(tool, arguments)

    assert isinstance(result, CallToolResult)
    assert result.is_error is True
    assert expected in _tool_text(result)
    mock_mcp_application.answers.create.assert_not_awaited()
