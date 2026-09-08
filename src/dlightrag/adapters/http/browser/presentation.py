# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Safe semantic projection shared by browser history, SSE, and Artifact views."""

import re
from typing import Any, Literal

import nh3
from pydantic import Field

from dlightrag.adapters.http.browser.markdown import (
    inject_highlights,
    normalize_chunk_source,
    render_chunk_content,
    render_markdown,
)
from dlightrag.adapters.http.browser.safe_html import sanitize_html_fragment
from dlightrag.application.corpus_admin import validate_public_web_url
from dlightrag.engine.answer.citations.contracts import (
    CITATION_PATTERN,
    DOC_CITATION_PATTERN,
    SourceReferencePayload,
)
from dlightrag.engine.answer.client_contracts import ClientContractModel
from dlightrag.engine.answer.results import answer_parts_from_markdown

_CHUNK_ALLOWED_TAGS = {
    "table",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "th",
    "td",
    "caption",
    "colgroup",
    "col",
    "p",
    "br",
    "hr",
    "b",
    "i",
    "em",
    "strong",
    "u",
    "s",
    "del",
    "sub",
    "sup",
    "mark",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "div",
    "span",
    "ul",
    "ol",
    "li",
    "dl",
    "dt",
    "dd",
    "pre",
    "code",
    "a",
    "blockquote",
    "abbr",
    "details",
    "summary",
}
_CHUNK_ALLOWED_ATTRS: dict[str, set[str]] = {
    "*": {"class"},
    "a": {"href", "title"},
    "td": {"colspan", "rowspan"},
    "th": {"colspan", "rowspan", "scope"},
    "col": {"span"},
    "colgroup": {"span"},
}


class PresentationImage(ClientContractModel):
    id: str = ""
    chunk_id: str = ""
    source_ref: str = ""
    url: str
    thumbnail_url: str
    label: str = ""
    answer_image_sent: bool = True


class PresentationArtifactIssue(ClientContractModel):
    kind: str
    description: str
    resource_id: str | None = None


class PresentationArtifact(ClientContractModel):
    resource_id: str
    media_type: str
    label: str
    filename: str
    byte_size: int
    digest: str
    presentation: Literal["image", "markdown", "html", "pdf", "text", "download"]
    status: Literal["available", "unavailable"]
    uri: str
    width: int | None = None
    height: int | None = None
    data_url: str | None = None
    download_url: str | None = None
    presentation_url: str | None = None
    issue: PresentationArtifactIssue | None = None


class PresentationArtifactOutcome(ClientContractModel):
    status: Literal["complete", "partial", "failed"] = "complete"
    issues: list[PresentationArtifactIssue] = Field(default_factory=list)


class PresentationPart(ClientContractModel):
    type: Literal["markdown", "artifact", "evidence_image"]
    text: str = ""
    html: str = ""
    artifact: PresentationArtifact | None = None
    evidence_image: PresentationImage | None = None
    inline: bool = False


class PresentationSourceChunk(ClientContractModel):
    chunk_idx: int | None = None
    page_number: int | None = None
    content_html: str = ""
    image_url: str | None = None
    thumbnail_url: str | None = None


class PresentationSource(ClientContractModel):
    id: str
    title: str
    source_url: str | None = None
    download_url: str | None = None
    chunks: list[PresentationSourceChunk]


class AnswerPresentation(ClientContractModel):
    answer_text: str
    parts: list[PresentationPart]
    sources: list[PresentationSource]
    evidence_images: list[PresentationImage]
    artifacts: list[PresentationArtifact]
    artifact_outcome: PresentationArtifactOutcome


def _protect_code_blocks(html: str) -> tuple[str, list[str]]:
    protected: list[str] = []

    def replace(match: re.Match[str]) -> str:
        index = len(protected)
        protected.append(match.group(0))
        return f"\x00CODE{index}\x00"

    html = re.sub(r"<pre[^>]*>.*?</pre>", replace, html, flags=re.DOTALL)
    html = re.sub(r"<code[^>]*>.*?</code>", replace, html, flags=re.DOTALL)
    return html, protected


def _restore_code_blocks(html: str, protected: list[str]) -> str:
    for index in range(len(protected) - 1, -1, -1):
        html = html.replace(f"\x00CODE{index}\x00", protected[index])
    return html


def _reference_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    ref = str(ref_id)
    return ref if chunk_idx is None or chunk_idx == "" else f"{ref}-{chunk_idx}"


def _reference_aria_label(ref_id: Any, chunk_idx: Any | None = None) -> str:
    ref = str(ref_id)
    return f"Source {ref}" if chunk_idx in {None, ""} else f"Source {ref}, chunk {chunk_idx}"


def render_answer_html(answer: str) -> str:
    """Render one Markdown segment with semantic citation controls."""
    html = render_markdown(answer)
    html, protected = _protect_code_blocks(html)

    def chunk_citation(match: re.Match[str]) -> str:
        ref_id, chunk_idx = match.group(1), match.group(2)
        return (
            f'<cite class="citation-badge" data-ref="{ref_id}" data-chunk="{chunk_idx}" '
            f'role="button" tabindex="0" aria-label="{_reference_aria_label(ref_id, chunk_idx)}">'
            f"{_reference_label(ref_id, chunk_idx)}</cite>"
        )

    def document_citation(match: re.Match[str]) -> str:
        ref_id = match.group(1)
        return (
            f'<cite class="citation-badge" data-ref="{ref_id}" role="button" tabindex="0" '
            f'aria-label="{_reference_aria_label(ref_id)}">{_reference_label(ref_id)}</cite>'
        )

    html = CITATION_PATTERN.sub(chunk_citation, html)
    html = DOC_CITATION_PATTERN.sub(document_citation, html)
    return sanitize_html_fragment(_restore_code_blocks(html, protected))


def render_source_chunk_html(content: str, phrases: list[str] | None = None) -> str:
    source = normalize_chunk_source(content)
    html = nh3.clean(
        render_chunk_content(source), tags=_CHUNK_ALLOWED_TAGS, attributes=_CHUNK_ALLOWED_ATTRS
    )
    return inject_highlights(html, source, phrases) if phrases else html


def _public_source_url(value: str) -> str | None:
    try:
        return validate_public_web_url(value.strip())
    except ValueError:
        return None


def _presentation_source(value: SourceReferencePayload | dict[str, Any]) -> PresentationSource:
    source = SourceReferencePayload.model_validate(value)
    return PresentationSource(
        id=source.id,
        title=source.title or "Source",
        source_url=_public_source_url(source.source_uri),
        download_url=source.download_url,
        chunks=[
            PresentationSourceChunk(
                chunk_idx=chunk.chunk_idx,
                page_number=chunk.page_number,
                content_html=render_source_chunk_html(chunk.content, chunk.highlight_phrases)
                if chunk.content
                else "",
                image_url=chunk.image_url,
                thumbnail_url=chunk.thumbnail_url,
            )
            for chunk in (source.chunks or [])
        ],
    )


def build_answer_presentation(
    *,
    answer: str,
    sources: list[SourceReferencePayload] | list[dict[str, Any]],
    evidence_images: list[dict[str, Any]],
    artifacts: list[dict[str, Any]] | None = None,
    artifact_outcome: dict[str, Any] | None = None,
) -> AnswerPresentation:
    """Build the bounded Web projection used identically by SSE and history."""
    artifact_values = artifacts or []
    image_values = evidence_images
    raw_parts = answer_parts_from_markdown(
        answer,
        artifacts=artifact_values,
        evidence_images=image_values,
    )
    parts = [
        PresentationPart(
            type=part["type"],
            text=str(part.get("text") or ""),
            html=render_answer_html(str(part.get("text") or ""))
            if part["type"] == "markdown"
            else "",
            artifact=(
                PresentationArtifact.model_validate(part["artifact"])
                if part.get("artifact")
                else None
            ),
            evidence_image=(
                PresentationImage.model_validate(part["evidence_image"])
                if part.get("evidence_image")
                else None
            ),
            inline=bool(part.get("inline")),
        )
        for part in raw_parts
    ]
    inline_evidence = {
        part.evidence_image.id
        for part in parts
        if part.type == "evidence_image" and part.evidence_image is not None
    }
    return AnswerPresentation(
        answer_text=answer,
        parts=parts,
        sources=[_presentation_source(source) for source in sources],
        evidence_images=[
            PresentationImage.model_validate(image)
            for image in image_values
            if str(image.get("id") or "") not in inline_evidence
        ],
        artifacts=[PresentationArtifact.model_validate(item) for item in artifact_values],
        artifact_outcome=PresentationArtifactOutcome.model_validate(
            artifact_outcome or {"status": "complete", "issues": []}
        ),
    )


__all__ = [
    "AnswerPresentation",
    "PresentationArtifact",
    "PresentationArtifactIssue",
    "PresentationArtifactOutcome",
    "PresentationImage",
    "PresentationPart",
    "PresentationSource",
    "PresentationSourceChunk",
    "build_answer_presentation",
    "render_answer_html",
    "render_source_chunk_html",
]
