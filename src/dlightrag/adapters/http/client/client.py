# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The async REST client for durable Answer runs."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import httpx

from dlightrag.adapters.http.client.attachments import AnswerAttachmentUpload
from dlightrag.application.runs import RunCancelledError, RunFailedError

logger = logging.getLogger(__name__)

STATUS_POLL_SECONDS = 1.0
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_BACKOFF_SECONDS = 0.5
EVENT_READ_IDLE_SECONDS = 30.0
_EVENT_STREAM_TIMEOUT = httpx.Timeout(
    connect=10.0, read=EVENT_READ_IDLE_SECONDS, write=10.0, pool=10.0
)

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


@dataclass(frozen=True, slots=True)
class RunDescriptor:
    """The accepted run one request created, with its owner-scoped URLs."""

    run_id: str
    run_kind: str
    lane: str
    status: str
    status_url: str
    events_url: str
    cancel_url: str
    parent_run_id: str | None = None
    continuation_kind: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RunDescriptor:
        run_id = str(payload["run_id"])
        return cls(
            run_id=run_id,
            run_kind=str(payload["run_kind"]),
            lane=str(payload["lane"]),
            status=str(payload["status"]),
            status_url=str(payload.get("status_url") or f"/runs/{run_id}"),
            events_url=str(payload.get("events_url") or f"/runs/{run_id}/events"),
            cancel_url=str(payload.get("cancel_url") or f"/runs/{run_id}"),
            parent_run_id=(str(payload["parent_run_id"]) if payload.get("parent_run_id") else None),
            continuation_kind=(
                str(payload["continuation_kind"]) if payload.get("continuation_kind") else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ProfileMemoryReceipt:
    """One settled direct Profile Memory operation."""

    action: str
    outcome: str
    change_id: str
    memory_ids: tuple[str, ...]
    kind: str | None = None
    body: str = ""
    supersedes_id: str | None = None
    target_change_id: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ProfileMemoryReceipt:
        return cls(
            action=str(payload["action"]),
            outcome=str(payload["outcome"]),
            change_id=str(payload["change_id"]),
            memory_ids=tuple(str(item) for item in payload.get("memory_ids") or ()),
            kind=str(payload["kind"]) if payload.get("kind") is not None else None,
            body=str(payload.get("body") or ""),
            supersedes_id=(
                str(payload["supersedes_id"]) if payload.get("supersedes_id") is not None else None
            ),
            target_change_id=(
                str(payload["target_change_id"])
                if payload.get("target_change_id") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ProfileMemorySettings:
    enabled: bool
    active_count: int | None


@dataclass(frozen=True, slots=True)
class AnswerStreamEvent:
    """One durable event replayed from the run's gap-free sequence."""

    sequence: int
    event_type: str
    payload: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AnswerArtifactIssue:
    kind: str
    description: str
    resource_id: str | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnswerArtifactIssue:
        return cls(
            kind=str(payload.get("kind") or "publication_failed"),
            description=str(payload.get("description") or "Artifact is unavailable."),
            resource_id=str(payload["resource_id"]) if payload.get("resource_id") else None,
        )


@dataclass(frozen=True, slots=True)
class ArtifactOutcome:
    status: Literal["complete", "partial", "failed"]
    issues: tuple[AnswerArtifactIssue, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> ArtifactOutcome:
        value = payload or {}
        status = value.get("status")
        return cls(
            status=status if status in {"complete", "partial", "failed"} else "complete",
            issues=tuple(
                AnswerArtifactIssue.from_payload(issue)
                for issue in value.get("issues") or ()
                if isinstance(issue, Mapping)
            ),
        )


@dataclass(frozen=True, slots=True)
class AnswerArtifact:
    resource_id: str
    media_type: str
    label: str
    filename: str
    byte_size: int
    digest: str
    presentation: str
    status: Literal["available", "unavailable"]
    uri: str
    width: int | None = None
    height: int | None = None
    data_url: str | None = None
    download_url: str | None = None
    presentation_url: str | None = None
    issue: AnswerArtifactIssue | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnswerArtifact:
        issue = payload.get("issue")
        status = payload.get("status")
        return cls(
            resource_id=str(payload["resource_id"]),
            media_type=str(payload.get("media_type") or "application/octet-stream"),
            label=str(payload.get("label") or payload.get("filename") or "Artifact"),
            filename=str(payload.get("filename") or "artifact"),
            byte_size=int(payload.get("byte_size") or 0),
            digest=str(payload.get("digest") or ""),
            presentation=str(payload.get("presentation") or "download"),
            status=status if status in {"available", "unavailable"} else "unavailable",
            uri=str(payload.get("uri") or ""),
            width=int(payload["width"]) if payload.get("width") is not None else None,
            height=int(payload["height"]) if payload.get("height") is not None else None,
            data_url=str(payload["data_url"]) if payload.get("data_url") else None,
            download_url=str(payload["download_url"]) if payload.get("download_url") else None,
            presentation_url=(
                str(payload["presentation_url"]) if payload.get("presentation_url") else None
            ),
            issue=AnswerArtifactIssue.from_payload(issue) if isinstance(issue, Mapping) else None,
        )


@dataclass(frozen=True, slots=True)
class EvidenceImage:
    id: str
    chunk_id: str
    source_ref: str
    url: str
    thumbnail_url: str
    label: str
    answer_image_sent: bool

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> EvidenceImage:
        return cls(
            id=str(payload.get("id") or ""),
            chunk_id=str(payload.get("chunk_id") or ""),
            source_ref=str(payload.get("source_ref") or ""),
            url=str(payload.get("url") or ""),
            thumbnail_url=str(payload.get("thumbnail_url") or ""),
            label=str(payload.get("label") or ""),
            answer_image_sent=payload.get("answer_image_sent") is not False,
        )


@dataclass(frozen=True, slots=True)
class AnswerPart:
    type: Literal["markdown", "artifact", "evidence_image"]
    text: str = ""
    artifact: AnswerArtifact | None = None
    evidence_image: EvidenceImage | None = None
    inline: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnswerPart:
        kind = payload.get("type")
        artifact = payload.get("artifact")
        image = payload.get("evidence_image")
        return cls(
            type=kind if kind in {"markdown", "artifact", "evidence_image"} else "markdown",
            text=str(payload.get("text") or ""),
            artifact=AnswerArtifact.from_payload(artifact)
            if isinstance(artifact, Mapping)
            else None,
            evidence_image=EvidenceImage.from_payload(image)
            if isinstance(image, Mapping)
            else None,
            inline=bool(payload.get("inline")),
        )


@dataclass(frozen=True, slots=True)
class AnswerResult:
    answer: str
    parts: tuple[AnswerPart, ...]
    sources: tuple[Mapping[str, Any], ...]
    evidence_images: tuple[EvidenceImage, ...]
    artifacts: tuple[AnswerArtifact, ...]
    artifact_outcome: ArtifactOutcome
    contexts: Mapping[str, Any]
    references: tuple[Mapping[str, Any], ...]
    usage: Mapping[str, Any]
    evidence: Mapping[str, Any]
    trace: Mapping[str, Any]
    image_descriptions: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnswerResult:
        return cls(
            answer=str(payload.get("answer") or ""),
            parts=tuple(
                AnswerPart.from_payload(part)
                for part in payload.get("parts") or ()
                if isinstance(part, Mapping)
            ),
            sources=tuple(dict(item) for item in payload.get("sources") or ()),
            evidence_images=tuple(
                EvidenceImage.from_payload(item)
                for item in payload.get("evidence_images") or ()
                if isinstance(item, Mapping)
            ),
            artifacts=tuple(
                AnswerArtifact.from_payload(item)
                for item in payload.get("artifacts") or ()
                if isinstance(item, Mapping)
            ),
            artifact_outcome=ArtifactOutcome.from_payload(payload.get("artifact_outcome")),
            contexts=dict(payload.get("contexts") or {}),
            references=tuple(dict(item) for item in payload.get("references") or ()),
            usage=dict(payload.get("usage") or {}),
            evidence=dict(payload.get("evidence") or {}),
            trace=dict(payload.get("trace") or {}),
            image_descriptions=tuple(str(item) for item in payload.get("image_descriptions") or ()),
        )


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """One authorized reader projection of a succeeded Retrieval run."""

    contexts: Mapping[str, Any]
    sources: tuple[Mapping[str, Any], ...]
    trace: Mapping[str, Any]
    image_descriptions: tuple[str, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> RetrievalResult:
        return cls(
            contexts=dict(payload.get("contexts") or {}),
            sources=tuple(dict(item) for item in payload.get("sources") or ()),
            trace=dict(payload.get("trace") or {}),
            image_descriptions=tuple(str(item) for item in payload.get("image_descriptions") or ()),
        )


def parse_sse_frames(chunk: str, *, buffer: str = "") -> tuple[list[AnswerStreamEvent], str]:
    """Decode complete SSE frames and return their incomplete tail."""
    text = buffer + chunk
    frames = text.split("\n\n")
    tail = frames.pop()
    events: list[AnswerStreamEvent] = []
    for frame in frames:
        sequence: int | None = None
        event_type = "message"
        data_lines: list[str] = []
        for line in frame.splitlines():
            if not line or line.startswith(":"):
                continue
            field, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if field == "id" and value.isdigit():
                sequence = int(value)
            elif field == "event":
                event_type = value
            elif field == "data":
                data_lines.append(value)
        if sequence is None or not data_lines:
            continue
        try:
            payload = json.loads("\n".join(data_lines))
        except json.JSONDecodeError:
            logger.warning("Discarding an answer event with undecodable data")
            continue
        events.append(
            AnswerStreamEvent(
                sequence=sequence,
                event_type=event_type,
                payload=payload if isinstance(payload, dict) else {"value": payload},
            )
        )
    return events, tail


class AnswerRunClient:
    """Internal REST helper for creating, following, and reading durable Runs."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = "",
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._headers = dict(headers or {})

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    async def create(
        self,
        payload: Mapping[str, Any],
        *,
        attachments: Sequence[AnswerAttachmentUpload] = (),
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        """Submit one Answer request and return its 202 descriptor."""
        headers = dict(self._headers)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if attachments:
            headers.pop("Content-Type", None)
            response = await self._client.post(
                self._url("/answer"),
                data={"request": json.dumps(payload)},
                files=[
                    ("attachments", (item.filename, item.content, item.content_type))
                    for item in attachments
                ],
                headers=headers,
            )
        else:
            response = await self._client.post(
                self._url("/answer"), json=dict(payload), headers=headers
            )
        response.raise_for_status()
        return RunDescriptor.from_payload(response.json())

    async def create_retrieve(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        """Submit one Retrieval request and return its 202 Run descriptor."""
        headers = dict(self._headers)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        response = await self._client.post(
            self._url("/retrieve"), json=dict(payload), headers=headers
        )
        response.raise_for_status()
        return RunDescriptor.from_payload(response.json())

    async def _create_corpus_action(
        self,
        action: Literal["ingest", "replace", "delete", "retry", "reset"],
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        headers = {
            **self._headers,
            "Idempotency-Key": idempotency_key or str(uuid4()),
        }
        response = await self._client.post(
            self._url(f"/runs/corpus/{action}"),
            json=dict(payload),
            headers=headers,
        )
        response.raise_for_status()
        return RunDescriptor.from_payload(response.json())

    async def create_corpus_ingest(
        self,
        payload: Mapping[str, Any],
        *,
        replace: bool = False,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        """Submit one durable ingest/replace mutation and return its Run descriptor."""
        return await self._create_corpus_action(
            "replace" if replace else "ingest",
            payload,
            idempotency_key=idempotency_key,
        )

    async def create_corpus_replace(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        return await self._create_corpus_action("replace", payload, idempotency_key=idempotency_key)

    async def create_corpus_upload(
        self,
        *,
        filename: str,
        content: bytes,
        workspace: str | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        content_sha256: str | None = None,
        replace: bool = False,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        """Upload one bounded source and return its durable mutation descriptor."""
        headers = {
            **self._headers,
            "Idempotency-Key": idempotency_key or str(uuid4()),
        }
        headers.pop("Content-Type", None)
        data = {
            key: value
            for key, value in {
                "workspace": workspace,
                "title": title,
                "author": author,
                "metadata": json.dumps(dict(metadata)) if metadata is not None else None,
                "content_sha256": content_sha256,
            }.items()
            if value is not None
        }
        action = "replace" if replace else "ingest"
        response = await self._client.post(
            self._url(f"/runs/corpus/{action}/upload"),
            data=data,
            files={"file": (filename, content)},
            headers=headers,
        )
        response.raise_for_status()
        return RunDescriptor.from_payload(response.json())

    async def create_corpus_delete(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        return await self._create_corpus_action("delete", payload, idempotency_key=idempotency_key)

    async def create_corpus_retry(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        return await self._create_corpus_action("retry", payload, idempotency_key=idempotency_key)

    async def create_corpus_reset(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RunDescriptor:
        return await self._create_corpus_action("reset", payload, idempotency_key=idempotency_key)

    async def resume_corpus_mutation(self, run_id: str) -> RunDescriptor:
        """Explicitly requeue the same waiting-for-repair mutation Run."""
        response = await self._client.post(
            self._url(f"/runs/{run_id}/resume"), headers=self._headers
        )
        response.raise_for_status()
        return RunDescriptor.from_payload(response.json())

    async def wait_corpus_mutation(self, run_id: str) -> dict[str, Any]:
        """Poll until terminal, or return a mutation awaiting operator repair."""
        while True:
            status = await self.status(run_id)
            state = str(status.get("status") or "")
            if state == "succeeded":
                result = status.get("result")
                return dict(result) if isinstance(result, Mapping) else {}
            if state == "running" and status.get("phase") == "waiting_for_repair":
                return status
            if state == "failed":
                raise RunFailedError(
                    str(status.get("error_kind") or "corpus_mutation_failed"),
                    str(status.get("error_message") or "Corpus Mutation failed."),
                )
            if state == "cancelled":
                raise RunCancelledError(run_id)
            await asyncio.sleep(STATUS_POLL_SECONDS)

    async def ingest(
        self,
        payload: Mapping[str, Any],
        *,
        replace: bool = False,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        """Create one durable ingest/replace mutation and wait for its result."""
        descriptor = await self.create_corpus_ingest(
            payload,
            replace=replace,
            idempotency_key=idempotency_key,
        )
        return await self.wait_corpus_mutation(descriptor.run_id)

    async def replace(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        descriptor = await self.create_corpus_replace(payload, idempotency_key=idempotency_key)
        return await self.wait_corpus_mutation(descriptor.run_id)

    async def upload(
        self,
        *,
        filename: str,
        content: bytes,
        workspace: str | None = None,
        title: str | None = None,
        author: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        content_sha256: str | None = None,
        replace: bool = False,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        descriptor = await self.create_corpus_upload(
            filename=filename,
            content=content,
            workspace=workspace,
            title=title,
            author=author,
            metadata=metadata,
            content_sha256=content_sha256,
            replace=replace,
            idempotency_key=idempotency_key,
        )
        return await self.wait_corpus_mutation(descriptor.run_id)

    async def delete(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        descriptor = await self.create_corpus_delete(payload, idempotency_key=idempotency_key)
        return await self.wait_corpus_mutation(descriptor.run_id)

    async def retry(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        descriptor = await self.create_corpus_retry(payload, idempotency_key=idempotency_key)
        return await self.wait_corpus_mutation(descriptor.run_id)

    async def reset(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        descriptor = await self.create_corpus_reset(payload, idempotency_key=idempotency_key)
        return await self.wait_corpus_mutation(descriptor.run_id)

    async def resume(self, run_id: str) -> dict[str, Any]:
        await self.resume_corpus_mutation(run_id)
        return await self.wait_corpus_mutation(run_id)

    async def status(self, run_id: str) -> dict[str, Any]:
        """Read one run's authoritative status and terminal result."""
        response = await self._client.get(self._url(f"/runs/{run_id}"), headers=self._headers)
        response.raise_for_status()
        return dict(response.json())

    async def steer(self, run_id: str, instruction: str) -> dict[str, Any]:
        """Queue one ordered steering instruction for a live Research run."""
        response = await self._client.post(
            self._url(f"/answer/{run_id}/steer"),
            json={"content": instruction},
            headers=self._headers,
        )
        response.raise_for_status()
        return dict(response.json())

    async def follow_up(
        self, run_id: str, query: str, *, idempotency_key: str | None = None
    ) -> RunDescriptor:
        return await self._continuation(run_id, "follow-up", query, idempotency_key)

    async def fork(
        self, run_id: str, query: str, *, idempotency_key: str | None = None
    ) -> RunDescriptor:
        return await self._continuation(run_id, "fork", query, idempotency_key)

    async def _continuation(
        self,
        run_id: str,
        operation: str,
        query: str,
        idempotency_key: str | None,
    ) -> RunDescriptor:
        headers = dict(self._headers)
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        response = await self._client.post(
            self._url(f"/answer/{run_id}/{operation}"),
            json={"content": query},
            headers=headers,
        )
        response.raise_for_status()
        return RunDescriptor.from_payload(response.json())

    async def transcript(self, run_id: str, *, limit: int = 20) -> dict[str, Any]:
        response = await self._client.get(
            self._url(f"/answer/{run_id}/transcript"),
            params={"limit": limit},
            headers=self._headers,
        )
        response.raise_for_status()
        return dict(response.json())

    async def children(
        self, run_id: str, *, cursor: str | None = None, limit: int = 50
    ) -> dict[str, Any]:
        """Return one bounded newest-first child-roster page for an owned run.

        The roster endpoint pages newest-first, so without ``cursor`` this reads
        only the newest ``limit`` children. The returned ``next_cursor`` is the
        opaque continuation, or ``None`` once the traversal is exhausted.
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._client.get(
            self._url(f"/answer/{run_id}/children"),
            params=params,
            headers=self._headers,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "children": list(payload.get("children") or []),
            "next_cursor": payload.get("next_cursor") or None,
        }

    async def cancel(self, run_id: str) -> dict[str, Any]:
        """Request cancellation; repeating it on a terminal run is a no-op."""
        response = await self._client.request(
            "DELETE", self._url(f"/runs/{run_id}"), headers=self._headers
        )
        response.raise_for_status()
        return dict(response.json())

    async def events(self, run_id: str, *, after: int = 0) -> AsyncGenerator[AnswerStreamEvent]:
        """Yield durable events after ``after``, reconnecting by sequence."""
        cursor = max(0, after)
        attempts = 0
        while True:
            try:
                async with aclosing(self._stream_once(run_id, cursor)) as stream:
                    async for event in stream:
                        cursor = event.sequence
                        attempts = 0
                        yield event
                        if event.event_type in {"done", "error"}:
                            return
            except httpx.HTTPStatusError:
                raise
            except httpx.HTTPError:
                attempts += 1
                if attempts >= MAX_RECONNECT_ATTEMPTS:
                    raise
                logger.info("Answer event stream dropped; resuming after sequence %d", cursor)
                await asyncio.sleep(RECONNECT_BACKOFF_SECONDS * attempts)
                continue
            if (await self.status(run_id))["status"] in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(STATUS_POLL_SECONDS)

    async def retrieve(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None = None,
    ) -> RetrievalResult:
        """Create one durable Retrieval and wait for its projected result."""
        descriptor = await self.create_retrieve(payload, idempotency_key=idempotency_key)
        return await self.wait_retrieve(descriptor.run_id)

    async def wait_retrieve(self, run_id: str) -> RetrievalResult:
        """Wait for an accepted Retrieval through events with status fallback."""
        try:
            async with aclosing(self.events(run_id)) as events:
                async for event in events:
                    if event.event_type == "done":
                        return self._terminal_retrieval_result(
                            run_id,
                            status=str(event.payload.get("status") or ""),
                            result=event.payload.get("result"),
                        )
                    if event.event_type == "error":
                        raise RunFailedError(
                            str(event.payload.get("kind") or "retrieval_failed"),
                            str(event.payload.get("message") or "Retrieval failed."),
                        )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 410:
                raise
        while True:
            status = await self.status(run_id)
            if status["status"] in _TERMINAL_STATUSES:
                return self._terminal_retrieval_result(
                    run_id,
                    status=str(status["status"]),
                    result=status.get("result"),
                    error_kind=status.get("error_kind"),
                    error_message=status.get("error_message"),
                )
            await asyncio.sleep(STATUS_POLL_SECONDS)

    async def answer(
        self,
        payload: Mapping[str, Any],
        *,
        attachments: Sequence[AnswerAttachmentUpload] = (),
        idempotency_key: str | None = None,
        on_token: Callable[[str], None] | None = None,
    ) -> AnswerResult:
        """Create one run and wait for its typed canonical result."""
        descriptor = await self.create(
            payload, attachments=attachments, idempotency_key=idempotency_key
        )
        try:
            async with aclosing(self.events(descriptor.run_id)) as events:
                async for event in events:
                    if event.event_type == "token" and on_token is not None:
                        on_token(str(event.payload.get("text") or ""))
                    elif event.event_type == "done":
                        return self._terminal_result(
                            descriptor.run_id,
                            status=str(event.payload.get("status") or ""),
                            result=event.payload.get("result"),
                        )
                    elif event.event_type == "error":
                        raise RunFailedError(
                            str(event.payload.get("kind") or "answer_stream_failed"),
                            str(event.payload.get("message") or "Answer run failed."),
                        )
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code != 410:
                raise
        return await self._await_terminal_status(descriptor.run_id)

    async def _stream_once(self, run_id: str, cursor: int) -> AsyncGenerator[AnswerStreamEvent]:
        headers = dict(self._headers)
        headers.pop("Content-Type", None)
        if cursor:
            headers["Last-Event-ID"] = str(cursor)
        async with self._client.stream(
            "GET",
            self._url(f"/runs/{run_id}/events"),
            headers=headers,
            timeout=_EVENT_STREAM_TIMEOUT,
        ) as response:
            response.raise_for_status()
            buffer = ""
            async for chunk in response.aiter_text():
                events, buffer = parse_sse_frames(chunk, buffer=buffer)
                for event in events:
                    yield event

    async def _await_terminal_status(self, run_id: str) -> AnswerResult:
        while True:
            status = await self.status(run_id)
            if status["status"] in _TERMINAL_STATUSES:
                return self._terminal_result(
                    run_id,
                    status=str(status["status"]),
                    result=status.get("result"),
                    error_kind=status.get("error_kind"),
                    error_message=status.get("error_message"),
                )
            await asyncio.sleep(STATUS_POLL_SECONDS)

    async def list_runs(self, *, after: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        response = await self._client.get(
            self._url("/runs"),
            params={"after": after, "limit": limit} if after else {"limit": limit},
            headers=self._headers,
        )
        response.raise_for_status()
        payload = response.json()
        return list(payload.get("runs") or [])

    async def list_memories(self, *, cursor: str | None = None, limit: int = 50) -> dict[str, Any]:
        """Return one bounded newest-first page of active Profile Memories.

        Without ``cursor`` this reads only the newest ``limit`` memories. The
        returned ``next_cursor`` is the opaque continuation, or ``None`` once
        the traversal is exhausted.
        """
        params: dict[str, Any] = {"limit": limit}
        if cursor is not None:
            params["cursor"] = cursor
        response = await self._client.get(
            self._url("/memory"), params=params, headers=self._headers
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "memories": list(payload.get("memories") or []),
            "next_cursor": payload.get("next_cursor") or None,
        }

    async def remember_memory(
        self,
        *,
        kind: str,
        body: str,
        supersedes_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> ProfileMemoryReceipt:
        key = idempotency_key or str(uuid4())
        response = await self._client.post(
            self._url("/memory"),
            json={"kind": kind, "body": body, "supersedes_id": supersedes_id},
            headers={**self._headers, "Idempotency-Key": key},
        )
        response.raise_for_status()
        return ProfileMemoryReceipt.from_payload(response.json())

    async def forget_memory(
        self, memory_id: str, *, idempotency_key: str | None = None
    ) -> ProfileMemoryReceipt:
        key = idempotency_key or str(uuid4())
        response = await self._client.delete(
            self._url(f"/memory/{memory_id}"),
            headers={**self._headers, "Idempotency-Key": key},
        )
        response.raise_for_status()
        return ProfileMemoryReceipt.from_payload(response.json())

    async def undo_memory_change(
        self, change_id: str, *, idempotency_key: str | None = None
    ) -> ProfileMemoryReceipt:
        key = idempotency_key or str(uuid4())
        response = await self._client.post(
            self._url(f"/memory/changes/{change_id}/undo"),
            headers={**self._headers, "Idempotency-Key": key},
        )
        response.raise_for_status()
        return ProfileMemoryReceipt.from_payload(response.json())

    async def memory_settings(self) -> ProfileMemorySettings:
        response = await self._client.get(self._url("/memory/settings"), headers=self._headers)
        response.raise_for_status()
        payload = response.json()
        count = payload.get("active_count")
        return ProfileMemorySettings(
            enabled=bool(payload["enabled"]),
            active_count=int(count) if count is not None else None,
        )

    async def set_memory_enabled(self, enabled: bool) -> ProfileMemorySettings:
        response = await self._client.put(
            self._url("/memory/settings"),
            json={"enabled": enabled},
            headers=self._headers,
        )
        response.raise_for_status()
        payload = response.json()
        count = payload.get("active_count")
        return ProfileMemorySettings(
            enabled=bool(payload["enabled"]),
            active_count=int(count) if count is not None else None,
        )

    async def clear_memory(self) -> None:
        response = await self._client.post(self._url("/memory/clear"), headers=self._headers)
        response.raise_for_status()

    async def list_artifacts(self, run_id: str) -> tuple[AnswerArtifact, ...]:
        response = await self._client.get(
            self._url(f"/answer/{run_id}/artifacts"), headers=self._headers
        )
        response.raise_for_status()
        payload = response.json()
        return tuple(
            AnswerArtifact.from_payload(item)
            for item in payload.get("artifacts") or ()
            if isinstance(item, Mapping)
        )

    async def read_artifact(
        self,
        run_id: str,
        resource_id: str,
        *,
        offset: int = 0,
        length: int = 1_048_576,
    ) -> bytes | None:
        """Read one bounded Artifact byte window without downloading the collection."""
        start = max(0, offset)
        end = start + max(1, length) - 1
        response = await self._client.get(
            self._url(f"/answer/{run_id}/artifacts/{resource_id}"),
            headers={**self._headers, "Range": f"bytes={start}-{end}"},
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.content

    async def download_artifact(
        self, run_id: str, artifact: AnswerArtifact, destination: str | Path
    ) -> Path:
        """Stream one explicitly selected Artifact to ``destination``."""
        target = Path(destination)
        with target.open("wb") as output:
            async for chunk in self.iter_artifact(run_id, artifact.resource_id):
                output.write(chunk)
        return target

    async def iter_artifact(
        self, run_id: str, resource_id: str, *, chunk_size: int = 1_048_576
    ) -> AsyncGenerator[bytes]:
        offset = 0
        while True:
            end = offset + max(1, chunk_size) - 1
            response = await self._client.get(
                self._url(f"/answer/{run_id}/artifacts/{resource_id}"),
                headers={**self._headers, "Range": f"bytes={offset}-{end}"},
            )
            if response.status_code in {404, 416}:
                return
            response.raise_for_status()
            chunk = response.content
            if not chunk:
                return
            yield chunk
            if len(chunk) < chunk_size:
                return
            offset += len(chunk)

    @staticmethod
    def _terminal_retrieval_result(
        run_id: str,
        *,
        status: str,
        result: Any,
        error_kind: Any = None,
        error_message: Any = None,
    ) -> RetrievalResult:
        if status == "succeeded" and isinstance(result, dict):
            return RetrievalResult.from_payload(result)
        if status == "cancelled":
            raise RunCancelledError(run_id)
        raise RunFailedError(
            str(error_kind or "retrieval_failed"),
            str(error_message or "Retrieval failed."),
        )

    @staticmethod
    def _terminal_result(
        run_id: str,
        *,
        status: str,
        result: Any,
        error_kind: Any = None,
        error_message: Any = None,
    ) -> AnswerResult:
        if status == "succeeded" and isinstance(result, dict):
            return AnswerResult.from_payload(result)
        if status == "cancelled":
            raise RunCancelledError(run_id)
        raise RunFailedError(
            str(error_kind or "answer_stream_failed"),
            str(error_message or "Answer run failed."),
        )


__all__ = [
    "EVENT_READ_IDLE_SECONDS",
    "MAX_RECONNECT_ATTEMPTS",
    "STATUS_POLL_SECONDS",
    "AnswerArtifact",
    "AnswerArtifactIssue",
    "AnswerPart",
    "AnswerResult",
    "AnswerRunClient",
    "RunDescriptor",
    "AnswerStreamEvent",
    "ArtifactOutcome",
    "EvidenceImage",
    "ProfileMemoryReceipt",
    "ProfileMemorySettings",
    "RetrievalResult",
    "parse_sse_frames",
]
