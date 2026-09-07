# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""MCP tools for durable Answer runs."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from mcp.types import ToolAnnotations
from pydantic import Field

from dlightrag.adapters.mcp import server as mcp_server
from dlightrag.adapters.mcp.contracts import (
    AnswerInput,
    AnswerRunInput,
)
from dlightrag.adapters.mcp.server import (
    AttachmentsParam,
    FederatedRerankParam,
    HistoryParam,
    IdempotencyKeyParam,
    mcp_app,
)
from dlightrag.application.access import (
    AccessAction,
    corpus_mutation_access_action,
    current_request_scope,
)
from dlightrag.application.answer_runs import AnswerRequest as ServiceAnswerRequest
from dlightrag.application.answer_runs.client_contracts import conversation_history_as_dicts
from dlightrag.application.answer_runs.resource_links import answer_link_resources
from dlightrag.application.answer_runs.results import project_answer_result
from dlightrag.application.retrieval import (
    MetadataFilter,
    RetrievalOptions,
    RetrieveProjection,
    retrieval_response_payload,
)
from dlightrag.application.runs import (
    IdempotencyKeyConflict,
    RunCapacityExceededError,
    RunView,
)


@mcp_app.tool(
    name="answer",
    description=(
        "Start an LLM-generated answer backed by retrieved context from the default or "
        "selected workspaces. Returns immediately with a run_id and its initial status; "
        "the answer itself is NOT returned here. Poll get_run with that run_id "
        "until status is succeeded, failed, or cancelled, and call cancel_run to "
        "stop a run you no longer need. A run survives this call, this connection, and a "
        "server restart."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False),
)
async def answer_tool(
    query: Annotated[str, Field(description="The question to answer")],
    top_k: Annotated[
        int | None,
        Field(default=None, description="Retrieval candidate count override for this answer"),
    ] = None,
    chunk_top_k: Annotated[
        int | None,
        Field(default=None, description="Vector chunk candidate count override for this answer"),
    ] = None,
    federated_rerank: FederatedRerankParam = False,
    workspaces: Annotated[
        list[str] | None,
        Field(default=None, description="Workspace names to search. Omit for default."),
    ] = None,
    all_workspaces: Annotated[
        bool,
        Field(
            default=False,
            description="Search all workspaces visible to the current caller.",
        ),
    ] = False,
    filters: Annotated[
        dict[str, Any] | None,
        Field(default=None, description="Metadata filters for structured queries."),
    ] = None,
    attachments: AttachmentsParam = Field(default_factory=list),
    semantic_highlights: Annotated[
        bool,
        Field(default=False, description="Include semantic highlight phrases in cited sources."),
    ] = False,
    history: HistoryParam = None,
    mode: Annotated[
        Literal["auto", "fast", "research"] | None,
        Field(default=None, description="Answer mode. Omit for auto."),
    ] = None,
    idempotency_key: IdempotencyKeyParam = None,
) -> dict[str, Any]:
    args = AnswerInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    max_attachments = application.config.answer.generation.max_attachments
    if len(args.attachments) > max_attachments:
        raise ValueError(f"Too many attachments; at most {max_attachments} are allowed")
    resolved_workspaces = await mcp_server._resolve_authorized_query_workspaces(
        application,
        workspaces=args.workspaces,
        all_workspaces=args.all_workspaces,
    )
    try:
        creation = await application.answers.create(
            request=ServiceAnswerRequest(
                query=args.query,
                workspaces=tuple(resolved_workspaces),
                retrieval=RetrievalOptions(
                    top_k=args.top_k,
                    chunk_top_k=args.chunk_top_k,
                    federated_rerank=args.federated_rerank,
                ),
                filters=MetadataFilter.model_validate(args.filters) if args.filters else None,
                semantic_highlights=args.semantic_highlights,
                history=tuple(conversation_history_as_dicts(args.history) or ()),
                resources=tuple(answer_link_resources(args.attachments)),
                mode=args.mode,
            ),
            idempotency_key=args.idempotency_key,
            owner_id=mcp_server._owner_id(),
            auth_mode=current_request_scope().auth_mode,
        )
    except IdempotencyKeyConflict:
        raise ValueError(
            "idempotency_key was already used for a different answer request"
        ) from None
    except RunCapacityExceededError:
        raise ValueError("Run admission capacity is full") from None
    return mcp_server._run_descriptor(creation.run)


@mcp_app.tool(
    name="get_run",
    description=(
        "Return the common lifecycle state of a durable run. status is queued, running, "
        "succeeded, failed, or cancelled; cancel_requested reports whether cancellation was "
        "asked for. A succeeded run carries its operation-owned result. A failed run carries "
        "error_kind and error_message. An unknown run id, or one owned by another caller, "
        "is reported as not found."
    ),
    annotations=ToolAnnotations(read_only_hint=True),
)
async def get_run_tool(
    run_id: Annotated[str, Field(description="Run id returned by a creation tool.")],
) -> dict[str, Any]:
    args = AnswerRunInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    record = await _authorized_run(application, args.run_id, cancel=False)
    result: dict[str, Any] | None = None
    if record.result is not None:
        if record.run_kind == "retrieval":
            result = retrieval_response_payload(
                application.retrieval.project_stored(
                    record.result,
                    RetrieveProjection(
                        downloadable_workspaces=frozenset(),
                        visual_workspaces=frozenset(
                            await mcp_server._authorized_workspace_names(
                                AccessAction.WORKSPACE_READ_VISUAL_ASSET,
                                [
                                    str(value)
                                    for value in record.request_input().get("workspaces") or ()
                                ],
                                application=application,
                            )
                        ),
                        include_download_links=False,
                    ),
                )
            )
        elif record.run_kind == "answer":
            result = project_answer_result(
                record.result,
                visual_workspaces=await mcp_server._authorized_workspace_names(
                    AccessAction.WORKSPACE_READ_VISUAL_ASSET,
                    [str(value) for value in record.request_input().get("workspaces") or ()],
                    application=application,
                ),
                run_id=record.run_id,
                artifact_url_prefix=None,
            )
        else:
            result = dict(record.result)
    return {
        **mcp_server._run_descriptor(record),
        "phase": record.phase,
        "durable_progress_version": record.durable_progress_version,
        "result": result,
        "error_kind": record.error_kind,
        "error_message": record.error_message,
        "repair_reason": record.repair_reason,
        "repair_remedy": record.repair_remedy,
        "finished_at": record.finished_at.isoformat() if record.finished_at else None,
    }


@mcp_app.tool(
    name="cancel_run",
    description=(
        "Request cancellation of a durable run and return its state. A queued run is "
        "cancelled immediately; a running one is cancelled once its "
        "worker observes the request, so status may still be running with cancel_requested "
        "true. Cancelling an already finished run changes nothing. An unknown run id, or "
        "one owned by another caller, is reported as not found."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def cancel_run_tool(
    run_id: Annotated[str, Field(description="Run id returned by a creation tool.")],
) -> dict[str, Any]:
    args = AnswerRunInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    record = await _authorized_run(application, args.run_id, cancel=True)
    outcome = await application.runs.cancel(owner_id=record.access_scope_id, run_id=args.run_id)
    if outcome.run is None:
        raise ValueError(f"Run not found: {args.run_id}")
    if outcome.outcome == "rejected":
        raise ValueError("Run cancellation is no longer safe after upstream handoff")
    return mcp_server._run_descriptor(outcome.run)


@mcp_app.tool(
    name="resume_corpus_run",
    description=(
        "Explicitly confirm operator repair and resume the same Corpus Mutation Run. "
        "The Run must be in phase waiting_for_repair and remains the lifecycle authority."
    ),
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=True),
)
async def resume_corpus_run_tool(
    run_id: Annotated[str, Field(description="Waiting-for-repair Corpus Mutation Run id.")],
) -> dict[str, Any]:
    args = AnswerRunInput.model_validate(locals())
    application = await mcp_server._ensure_application()
    record = await _authorized_run(application, args.run_id, cancel=True)
    if record.run_kind != "corpus_mutation" or record.phase != "waiting_for_repair":
        raise ValueError("Run is not waiting for repair")
    if not await application.runs.resume_repair(
        owner_id=record.access_scope_id,
        run_id=args.run_id,
    ):
        raise ValueError("Run could not be resumed")
    updated = await application.runs.get(
        owner_id=record.access_scope_id,
        run_id=args.run_id,
    )
    if updated is None:
        raise ValueError(f"Run not found: {args.run_id}")
    return mcp_server._run_descriptor(updated)


@mcp_app.tool(
    name="steer_answer_run",
    description="Queue one ordered steering instruction for a live Research run.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False),
)
async def steer_answer_run_tool(
    run_id: Annotated[str, Field(description="Run id")],
    instruction: Annotated[str, Field(min_length=1, max_length=20_000)],
) -> dict[str, Any]:
    receipt = await (await mcp_server._ensure_application()).answers.steer(
        owner_id=mcp_server._owner_id(), run_id=run_id, instruction=instruction
    )
    if receipt is None:
        raise ValueError("Run is not a live Research session")
    return {
        "run_id": receipt.run_id,
        "control_sequence": receipt.control_sequence,
        "kind": receipt.kind,
    }


async def _authorized_run(application: Any, run_id: str, *, cancel: bool) -> RunView:
    record = await application.runs.get_global(run_id=run_id)
    if record is None:
        raise ValueError(f"Run not found: {run_id}")
    if record.access_scope_kind == "owner":
        if record.access_scope_id != mcp_server._owner_id():
            raise ValueError(f"Run not found: {run_id}")
        return record
    action = AccessAction.WORKSPACE_LIST_FILES
    if cancel:
        action = corpus_mutation_access_action(record.request_input().get("action"))
    try:
        await mcp_server._enforce_access(action, record.access_scope_id, application=application)
    except Exception:
        raise ValueError(f"Run not found: {run_id}") from None
    return record


async def _mcp_continuation(
    run_id: str,
    query: str,
    *,
    fork: bool,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    owner_id = mcp_server._owner_id()
    parent_workspaces = await application.answers.continuation_workspaces(
        owner_id=owner_id, run_id=run_id
    )
    authorized_workspaces: list[str] | None = None
    if parent_workspaces is not None:
        authorized_workspaces = await mcp_server._resolve_authorized_query_workspaces(
            application,
            workspaces=list(parent_workspaces),
            all_workspaces=False,
        )
    method = application.answers.fork if fork else application.answers.follow_up
    try:
        creation = await method(
            owner_id=owner_id,
            run_id=run_id,
            query=query,
            idempotency_key=idempotency_key,
            auth_mode=current_request_scope().auth_mode,
            authorized_workspaces=authorized_workspaces,
        )
    except IdempotencyKeyConflict:
        raise ValueError("idempotency_key was already used for a different continuation") from None
    except RunCapacityExceededError:
        raise ValueError("Run admission capacity is full") from None
    if creation is None:
        raise ValueError("Continuation requires a terminal owned run")
    return mcp_server._run_descriptor(creation.run)


@mcp_app.tool(
    name="follow_up_answer_run",
    description="Start a continuation using one terminal run's answer as context.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False),
)
async def follow_up_answer_run_tool(
    run_id: Annotated[str, Field(description="Terminal run id")],
    query: Annotated[str, Field(min_length=1, max_length=20_000, description="Follow-up question")],
    idempotency_key: IdempotencyKeyParam = None,
) -> dict[str, Any]:
    return await _mcp_continuation(run_id, query, fork=False, idempotency_key=idempotency_key)


@mcp_app.tool(
    name="fork_answer_run",
    description="Start a sibling branch from one terminal run's accepted context.",
    annotations=ToolAnnotations(read_only_hint=False, idempotent_hint=False),
)
async def fork_answer_run_tool(
    run_id: Annotated[str, Field(description="Terminal run id")],
    query: Annotated[str, Field(min_length=1, max_length=20_000, description="Branch question")],
    idempotency_key: IdempotencyKeyParam = None,
) -> dict[str, Any]:
    return await _mcp_continuation(run_id, query, fork=True, idempotency_key=idempotency_key)


@mcp_app.tool(
    name="get_answer_transcript",
    description="Return the bounded transcript tail for one owned run.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def get_answer_transcript_tool(
    run_id: Annotated[str, Field(description="Run id")],
    limit: Annotated[int, Field(default=20, ge=1, le=100)] = 20,
) -> dict[str, Any]:
    transcript = await (await mcp_server._ensure_application()).answers.transcript_tail(
        owner_id=mcp_server._owner_id(), run_id=run_id, limit=limit
    )
    if transcript is None:
        raise ValueError(f"Answer run not found: {run_id}")
    return {
        "run_id": transcript.run_id,
        "status": transcript.status,
        "messages": list(transcript.messages),
    }


@mcp_app.tool(
    name="list_answer_children",
    description=(
        "Return at most the newest 50 children of one owned run's foreground "
        "roster. has_more=true marks additional older children that are only "
        "available through the Web/REST children endpoints with cursor paging."
    ),
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def list_answer_children_tool(
    run_id: Annotated[str, Field(description="Run id")],
) -> dict[str, Any]:
    page = await (await mcp_server._ensure_application()).answers.children(
        owner_id=mcp_server._owner_id(), run_id=run_id
    )
    if page is None:
        raise ValueError(f"Answer run not found: {run_id}")
    return {
        "run_id": run_id,
        "children": [dict(child) for child in page.children],
        "has_more": page.next_cursor is not None,
    }


@mcp_app.tool(
    name="list_runs",
    description="List this caller's durable runs, oldest first.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def list_runs_tool(
    after: Annotated[str | None, Field(default=None, description="Cursor run id")] = None,
    limit: Annotated[int, Field(default=50, description="Page size")] = 50,
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    rows = await application.runs.list(
        owner_id=mcp_server._owner_id(), after_run_id=after, limit=limit
    )
    return {"runs": [mcp_server._run_descriptor(record) for record in rows]}


@mcp_app.tool(
    name="list_answer_artifacts",
    description="List artifact descriptors for one owned answer run.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def list_answer_artifacts_tool(
    run_id: Annotated[str, Field(description="Run id")],
) -> dict[str, Any]:
    application = await mcp_server._ensure_application()
    owner_id = mcp_server._owner_id()
    if await application.answers.list_artifacts(owner_id=owner_id, run_id=run_id) is None:
        raise ValueError(f"Answer run not found: {run_id}")
    record = await application.runs.get(owner_id=owner_id, run_id=run_id)
    if record is None:
        raise ValueError(f"Answer run not found: {run_id}")
    if record.result is None:
        raise ValueError("Answer artifacts are not available until the run has a stored result")
    projected = project_answer_result(
        record.result,
        run_id=run_id,
        artifact_url_prefix=None,
    )
    return {
        "artifacts": projected["artifacts"],
        "artifact_outcome": projected["artifact_outcome"],
    }


@mcp_app.tool(
    name="read_answer_artifact",
    description="Read up to 1 MiB of one artifact as base64, returning the next offset.",
    annotations=ToolAnnotations(read_only_hint=True, idempotent_hint=True),
)
async def read_answer_artifact_tool(
    run_id: Annotated[str, Field(description="Run id")],
    resource_id: Annotated[str, Field(description="Artifact resource id")],
    offset: Annotated[int, Field(default=0, description="Byte offset")] = 0,
    length: Annotated[int, Field(default=1_048_576, description="Max bytes")] = 1_048_576,
) -> dict[str, Any]:
    import base64

    application = await mcp_server._ensure_application()
    start = max(0, offset)
    chunk = await application.answers.read_artifact(
        owner_id=mcp_server._owner_id(),
        run_id=run_id,
        resource_id=resource_id,
        offset=start,
        length=min(max(0, length), 1_048_576),
    )
    if chunk is None:
        raise ValueError("artifact not found")
    return {
        "data": base64.b64encode(chunk).decode("ascii"),
        "offset": start,
        "next_offset": start + len(chunk),
        "bytes": len(chunk),
    }
