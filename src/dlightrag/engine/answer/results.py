# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Canonical durable Answer Result and authenticated transport projection."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

from pydantic import BaseModel, Field, field_validator

from dlightrag.engine.answer.citations.contracts import SourceReference, SourceReferencePayload
from dlightrag.engine.answer.citations.sources import (
    SourceDownloadLinkBuilder,
    can_project_workspace_visual,
    project_contexts_for_client,
    project_source_payloads,
)
from dlightrag.engine.answer.citations.utils import context_chunk_key
from dlightrag.engine.answer.runs.snapshots import dump_answer_snapshot, load_answer_snapshot
from dlightrag.engine.rag.retrieval import RetrievalContexts

IMAGE_URL_PREFIX = "/images"
_ARTIFACT_PART = re.compile(
    r"(?P<image>!)?\[(?P<label>[^\]]*)\]\(\s*<?artifact:(?P<resource>[^\s)>]+)>?(?:\s+[^)]*)?\)",
    re.IGNORECASE,
)
_EVIDENCE_PART = re.compile(
    r"!\[(?P<label>[^\]]*)\]\(\s*<?evidence:(?P<resource>[^\s)>]+)>?(?:\s+[^)]*)?\)",
    re.IGNORECASE,
)


class Reference(BaseModel):
    """One document-level reference returned with an Answer result."""

    id: str = Field(description="Reference id matching the answer context")
    title: str = Field(description="Document title or filename")

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_id(cls, value: object) -> str:
        return str(value)


@dataclass(frozen=True, slots=True)
class AnswerArtifactIssue:
    kind: str
    description: str
    resource_id: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactOutcome:
    status: Literal["complete", "partial", "failed"] = "complete"
    issues: tuple[AnswerArtifactIssue, ...] = ()


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


@dataclass(frozen=True, slots=True)
class EvidenceImage:
    id: str
    chunk_id: str
    source_ref: str
    url: str
    thumbnail_url: str
    label: str
    answer_image_sent: bool = True


@dataclass(frozen=True, slots=True)
class AnswerPart:
    type: Literal["markdown", "artifact", "evidence_image"]
    text: str = ""
    artifact: AnswerArtifact | None = None
    evidence_image: EvidenceImage | None = None
    inline: bool = False


@dataclass
class AnswerResult:
    """Typed in-process projection of the canonical Answer Result."""

    answer: str
    parts: list[AnswerPart] = field(default_factory=list)
    contexts: RetrievalContexts = field(default_factory=dict)
    references: list[Reference] = field(default_factory=list)
    sources: list[SourceReference] = field(default_factory=list)
    evidence_images: list[EvidenceImage] = field(default_factory=list)
    artifacts: list[AnswerArtifact] = field(default_factory=list)
    artifact_outcome: ArtifactOutcome = field(default_factory=ArtifactOutcome)
    trace: dict[str, Any] = field(default_factory=dict)
    image_descriptions: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)


def store_answer_result(
    *,
    answer: str,
    contexts: RetrievalContexts,
    sources: Sequence[SourceReference],
    evidence_images: Sequence[Mapping[str, Any]],
    trace: Mapping[str, Any],
    image_descriptions: Sequence[str],
    artifacts: Sequence[Mapping[str, Any]] = (),
    artifact_outcome: Mapping[str, Any] | None = None,
    artifact_sources: Mapping[str, Sequence[SourceReference]] | None = None,
) -> dict[str, Any]:
    """Store one canonical Markdown Answer and transport-neutral identities."""
    workspaces = _image_workspaces(sources)
    usage = trace.get("usage")
    return {
        "answer": answer,
        "contexts": dict(contexts),
        "sources": dump_answer_snapshot(list(sources))["sources"],
        "evidence_images": [
            _store_evidence_image(image, workspaces=workspaces) for image in evidence_images
        ],
        "trace": dict(trace),
        "usage": dict(usage) if isinstance(usage, Mapping) else {},
        "evidence": _evidence_summary(contexts, source_count=len(sources)),
        "image_descriptions": list(image_descriptions),
        "artifacts": [dict(item) for item in artifacts],
        "artifact_outcome": dict(artifact_outcome or {"status": "complete", "issues": []}),
        "artifact_sources": {
            str(resource_id): dump_answer_snapshot(list(source_values))["sources"]
            for resource_id, source_values in (artifact_sources or {}).items()
            if source_values
        },
    }


def restore_answer_result(stored: Mapping[str, Any]) -> AnswerResult:
    """Rebuild the typed internal result without authorization-sensitive URLs."""
    projected = project_answer_result(stored)
    sources = load_answer_snapshot(
        {"sources": stored.get("sources") or []}, image_url_prefix=IMAGE_URL_PREFIX
    )
    artifacts = [_artifact_model(item) for item in projected["artifacts"]]
    images = [_evidence_model(item) for item in projected["evidence_images"]]
    return AnswerResult(
        answer=str(projected["answer"]),
        parts=_part_models(projected["parts"]),
        contexts=dict(projected["contexts"]),
        references=[Reference(id=source.id, title=source.title or "Source") for source in sources],
        sources=sources,
        evidence_images=images,
        artifacts=artifacts,
        artifact_outcome=_outcome_model(projected["artifact_outcome"]),
        trace=dict(projected["trace"]),
        image_descriptions=list(projected["image_descriptions"]),
        usage=dict(projected["usage"]),
        evidence=dict(projected["evidence"]),
    )


def project_answer_result(
    stored: Mapping[str, Any],
    *,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_url_prefix: str = IMAGE_URL_PREFIX,
    run_id: str | None = None,
    artifact_url_prefix: str | None = None,
) -> dict[str, Any]:
    """Project a stored result for one authenticated reader.

    ``artifact_url_prefix=None`` deliberately omits browser-cookie URLs for MCP.
    REST/Web callers pass their owner-scoped route prefix and the owning run id.
    """
    sources = load_answer_snapshot(
        {"sources": stored.get("sources") or []}, image_url_prefix=image_url_prefix
    )
    source_payloads = project_source_payloads(
        sources,
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )
    answer = str(stored.get("answer") or "")
    images = [
        _public_evidence_image(image, image_url_prefix=image_url_prefix)
        for image in stored.get("evidence_images") or ()
        if image and can_project_workspace_visual(image.get("workspace"), visual_workspaces)
    ]
    artifacts = [
        _public_artifact(
            item,
            run_id=run_id,
            artifact_url_prefix=artifact_url_prefix,
        )
        for item in stored.get("artifacts") or ()
        if isinstance(item, Mapping)
    ]
    outcome = _public_outcome(stored.get("artifact_outcome"))
    return {
        "answer": answer,
        "parts": answer_parts_from_markdown(answer, artifacts=artifacts, evidence_images=images),
        "contexts": project_contexts_for_client(
            dict(stored.get("contexts") or {}),
            image_url_prefix=image_url_prefix,
            visual_workspaces=visual_workspaces,
        ),
        "references": [
            {"id": source.id, "title": source.title or "Source"} for source in source_payloads
        ],
        "sources": [source.model_dump() for source in source_payloads],
        "evidence_images": images,
        "trace": dict(stored.get("trace") or {}),
        "usage": dict(stored.get("usage") or {}),
        "evidence": dict(stored.get("evidence") or {}),
        "image_descriptions": list(stored.get("image_descriptions") or ()),
        "artifacts": artifacts,
        "artifact_outcome": outcome,
    }


def answer_parts_from_markdown(
    answer: str,
    *,
    artifacts: Sequence[Mapping[str, Any]],
    evidence_images: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Derive ordered semantic parts from canonical Markdown and stable ids."""
    artifacts_by_id = {str(item.get("resource_id") or ""): dict(item) for item in artifacts}
    images_by_id = {str(item.get("id") or ""): dict(item) for item in evidence_images}
    matches: list[tuple[int, int, str, re.Match[str]]] = [
        (match.start(), match.end(), "artifact", match) for match in _ARTIFACT_PART.finditer(answer)
    ]
    matches.extend(
        (match.start(), match.end(), "evidence_image", match)
        for match in _EVIDENCE_PART.finditer(answer)
    )
    matches.sort(key=lambda value: value[0])
    result: list[dict[str, Any]] = []
    cursor = 0
    for start, end, kind, match in matches:
        if start < cursor:
            continue
        if start > cursor:
            result.append({"type": "markdown", "text": answer[cursor:start]})
        resource = str(match.group("resource"))
        label = str(match.group("label") or "")
        if kind == "artifact":
            artifact = artifacts_by_id.get(resource)
            if artifact is None:
                result.append({"type": "markdown", "text": match.group(0)})
            else:
                artifact = {**artifact, "label": label or artifact.get("label")}
                result.append(
                    {
                        "type": "artifact",
                        "artifact": artifact,
                        "inline": bool(match.group("image")),
                    }
                )
        else:
            image = images_by_id.get(resource)
            if image is None:
                result.append({"type": "markdown", "text": match.group(0)})
            else:
                result.append(
                    {
                        "type": "evidence_image",
                        "evidence_image": {**image, "label": label or image.get("label")},
                        "inline": True,
                    }
                )
        cursor = end
    if cursor < len(answer) or not result:
        result.append({"type": "markdown", "text": answer[cursor:]})
    return [part for part in result if part.get("type") != "markdown" or part.get("text")]


def project_artifact_sources(
    stored: Mapping[str, Any],
    *,
    resource_id: str,
    source_link_builder: SourceDownloadLinkBuilder | None = None,
    downloadable_workspaces: set[str] | None = None,
    visual_workspaces: set[str] | None = None,
    image_url_prefix: str = IMAGE_URL_PREFIX,
) -> list[SourceReferencePayload]:
    """Project citation sources belonging to one Markdown Artifact."""
    snapshots = stored.get("artifact_sources")
    source_values = snapshots.get(resource_id, []) if isinstance(snapshots, Mapping) else []
    return project_source_payloads(
        load_answer_snapshot(
            {"sources": source_values},
            image_url_prefix=image_url_prefix,
        ),
        resolver=source_link_builder,
        downloadable_workspaces=downloadable_workspaces,
        visual_workspaces=visual_workspaces,
    )


def _public_artifact(
    item: Mapping[str, Any],
    *,
    run_id: str | None,
    artifact_url_prefix: str | None,
) -> dict[str, Any]:
    resource_id = str(item.get("resource_id") or "")
    issue = item.get("issue")
    value: dict[str, Any] = {
        "resource_id": resource_id,
        "media_type": str(item.get("media_type") or "application/octet-stream"),
        "label": str(item.get("label") or item.get("filename") or "Artifact"),
        "filename": str(item.get("filename") or "artifact"),
        "byte_size": int(item.get("byte_size") or 0),
        "digest": str(item.get("digest") or ""),
        "presentation": str(item.get("presentation") or "download"),
        "status": "available" if item.get("status") == "available" else "unavailable",
        "uri": f"dlightrag://answer/{run_id or 'run'}/artifacts/{resource_id}",
        "width": item.get("width"),
        "height": item.get("height"),
        "issue": dict(issue) if isinstance(issue, Mapping) else None,
    }
    if artifact_url_prefix is not None and run_id and value["status"] == "available":
        base = (
            f"{artifact_url_prefix.rstrip('/')}/{quote(run_id, safe='')}/artifacts/"
            f"{quote(resource_id, safe='')}"
        )
        value.update(
            data_url=base,
            download_url=f"{base}?download=1",
            presentation_url=(
                f"{base}/presentation" if value["presentation"] == "markdown" else None
            ),
        )
    else:
        value.update(data_url=None, download_url=None, presentation_url=None)
    return value


def _public_outcome(value: Any) -> dict[str, Any]:
    outcome = value if isinstance(value, Mapping) else {}
    status = outcome.get("status")
    return {
        "status": status if status in {"complete", "partial", "failed"} else "complete",
        "issues": [
            {
                "kind": str(issue.get("kind") or "publication_failed"),
                "description": str(issue.get("description") or "Artifact is unavailable."),
                "resource_id": (
                    str(issue["resource_id"]) if issue.get("resource_id") is not None else None
                ),
            }
            for issue in outcome.get("issues") or ()
            if isinstance(issue, Mapping)
        ],
    }


def _evidence_summary(
    contexts: Mapping[str, Sequence[Mapping[str, Any]]], *, source_count: int
) -> dict[str, int]:
    return {
        "chunks": len(contexts.get("chunks") or ()),
        "entities": len(contexts.get("entities") or ()),
        "relationships": len(contexts.get("relationships") or ()),
        "sources": source_count,
    }


def _image_workspaces(sources: Sequence[SourceReference]) -> dict[str, str]:
    workspaces: dict[str, str] = {}
    for source in sources:
        for chunk in source.chunks or []:
            workspaces.setdefault(chunk.chunk_id, source.workspace)
            workspaces[context_chunk_key(chunk.chunk_id, workspace=source.workspace)] = (
                source.workspace
            )
    return workspaces


def _store_evidence_image(
    image: Mapping[str, Any], *, workspaces: dict[str, str]
) -> dict[str, Any]:
    image_id = str(image.get("id") or "")
    chunk_id = str(image.get("chunk_id") or "")
    return {
        "id": image_id,
        "chunk_id": chunk_id,
        "workspace": workspaces.get(image_id) or workspaces.get(chunk_id) or "",
        "source_ref": str(image.get("source_ref") or ""),
        "label": str(image.get("label") or ""),
        "answer_image_sent": image.get("answer_image_sent") is not False,
    }


def _public_evidence_image(
    image: Mapping[str, Any], *, image_url_prefix: str = IMAGE_URL_PREFIX
) -> dict[str, Any]:
    base = (
        f"{image_url_prefix.rstrip('/')}/"
        f"{quote(str(image.get('workspace') or ''), safe='')}/"
        f"{quote(str(image.get('chunk_id') or ''), safe='')}"
    )
    return {
        "id": str(image.get("id") or ""),
        "chunk_id": str(image.get("chunk_id") or ""),
        "source_ref": str(image.get("source_ref") or ""),
        "url": f"{base}?size=full",
        "thumbnail_url": f"{base}?size=thumb",
        "label": str(image.get("label") or ""),
        "answer_image_sent": image.get("answer_image_sent") is not False,
    }


def _artifact_model(value: Mapping[str, Any]) -> AnswerArtifact:
    issue_value = value.get("issue")
    issue = (
        AnswerArtifactIssue(
            kind=str(issue_value.get("kind") or "publication_failed"),
            description=str(issue_value.get("description") or "Artifact is unavailable."),
            resource_id=(
                str(issue_value["resource_id"])
                if issue_value.get("resource_id") is not None
                else None
            ),
        )
        if isinstance(issue_value, Mapping)
        else None
    )
    return AnswerArtifact(
        resource_id=str(value["resource_id"]),
        media_type=str(value["media_type"]),
        label=str(value["label"]),
        filename=str(value["filename"]),
        byte_size=int(value["byte_size"]),
        digest=str(value["digest"]),
        presentation=str(value["presentation"]),
        status=value["status"],
        uri=str(value["uri"]),
        width=int(value["width"]) if value.get("width") is not None else None,
        height=int(value["height"]) if value.get("height") is not None else None,
        data_url=str(value["data_url"]) if value.get("data_url") else None,
        download_url=str(value["download_url"]) if value.get("download_url") else None,
        presentation_url=(
            str(value["presentation_url"]) if value.get("presentation_url") else None
        ),
        issue=issue,
    )


def _evidence_model(value: Mapping[str, Any]) -> EvidenceImage:
    return EvidenceImage(
        id=str(value["id"]),
        chunk_id=str(value["chunk_id"]),
        source_ref=str(value["source_ref"]),
        url=str(value["url"]),
        thumbnail_url=str(value["thumbnail_url"]),
        label=str(value["label"]),
        answer_image_sent=value.get("answer_image_sent") is not False,
    )


def _part_models(values: Sequence[Mapping[str, Any]]) -> list[AnswerPart]:
    result: list[AnswerPart] = []
    for value in values:
        result.append(
            AnswerPart(
                type=value["type"],
                text=str(value.get("text") or ""),
                artifact=(
                    _artifact_model(value["artifact"])
                    if isinstance(value.get("artifact"), Mapping)
                    else None
                ),
                evidence_image=(
                    _evidence_model(value["evidence_image"])
                    if isinstance(value.get("evidence_image"), Mapping)
                    else None
                ),
                inline=bool(value.get("inline")),
            )
        )
    return result


def _outcome_model(value: Mapping[str, Any]) -> ArtifactOutcome:
    return ArtifactOutcome(
        status=value["status"],
        issues=tuple(
            AnswerArtifactIssue(
                kind=str(issue["kind"]),
                description=str(issue["description"]),
                resource_id=str(issue["resource_id"]) if issue.get("resource_id") else None,
            )
            for issue in value.get("issues") or ()
        ),
    )


__all__ = [
    "AnswerArtifact",
    "AnswerArtifactIssue",
    "AnswerPart",
    "AnswerResult",
    "ArtifactOutcome",
    "EvidenceImage",
    "IMAGE_URL_PREFIX",
    "Reference",
    "answer_parts_from_markdown",
    "project_answer_result",
    "project_artifact_sources",
    "restore_answer_result",
    "store_answer_result",
]
