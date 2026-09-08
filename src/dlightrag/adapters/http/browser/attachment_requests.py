# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Normalize JSON and multipart Web answer requests."""

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

from fastapi import HTTPException, Request
from pydantic import ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException as StarletteHTTPException

from dlightrag.adapters.http.browser.attachment_models import (
    ValidatedWebAttachment,
    classify_web_attachment,
    validate_web_attachments,
)
from dlightrag.adapters.http.browser.requests import WebAnswerRequest
from dlightrag.engine.answer.image_capability import (
    AnswerImageCapability,
    check_answer_image_count,
)

# Bound the multipart parse *before* buffering any bodies so a client cannot
# push Starlette's default 1000 parts into memory/disk ahead of the attachment
# caps. Allow one extra file part beyond the cap so an over-limit request
# surfaces a precise 413 here instead of Starlette's generic 400. ``max_fields``
# covers the handful of non-file form fields (query/workspaces/conversation_id/
# submission_id) plus slack.
_MAX_FORM_FIELDS = 16
_MAX_FORM_FIELD_BYTES = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ParsedWebAnswerRequest:
    query: str
    workspaces: list[str] | None
    conversation_id: UUID | None
    submission_id: UUID
    attachments: tuple[ValidatedWebAttachment, ...]
    mode: str | None = None
    requested_skill: str | None = None


def _optional_uuid(value: Any) -> UUID | None:
    if value in (None, ""):
        return None
    return UUID(str(value))


def _optional_skill(value: Any) -> str | None:
    """Normalize an optional requested skill name to a canonical identifier."""
    if value in (None, ""):
        return None
    return str(value).strip()


def _optional_mode(value: Any) -> Literal["auto", "fast", "research"] | None:
    if value in (None, ""):
        return None
    mode = str(value)
    if mode in {"auto", "fast", "research"}:
        return mode  # type: ignore[return-value]
    raise HTTPException(status_code=422, detail="Invalid mode")


def _json_list(value: Any, *, field: str) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    raise HTTPException(status_code=422, detail=f"Invalid {field}")


async def parse_web_answer_request(
    request: Request,
    *,
    max_attachments: int,
    max_attachment_bytes: int,
    max_total_attachment_bytes: int,
    image_max_pixels: int,
    answer_image_capability: AnswerImageCapability | None,
) -> ParsedWebAnswerRequest:
    content_type = request.headers.get("content-type", "").lower()
    if "multipart/form-data" not in content_type:
        try:
            body = WebAnswerRequest.model_validate_json(await request.body())
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=exc.errors()) from exc
        return ParsedWebAnswerRequest(
            query=body.query,
            workspaces=body.workspaces,
            conversation_id=body.conversation_id,
            submission_id=body.submission_id,
            attachments=(),
            mode=body.mode,
            requested_skill=_optional_skill(body.requested_skill),
        )

    try:
        form = await request.form(
            max_files=max_attachments + 1,
            max_fields=_MAX_FORM_FIELDS,
            max_part_size=_MAX_FORM_FIELD_BYTES,
        )
    except StarletteHTTPException as exc:
        detail = str(exc.detail)
        if exc.status_code == 400 and detail.startswith(
            ("Too many files.", "Too many fields.", "Part exceeded maximum size")
        ):
            raise HTTPException(status_code=413, detail=detail) from exc
        raise
    try:
        files = form.getlist("attachments")
        attachment_parts = [
            item for item in files if isinstance(item, UploadFile) and item.filename
        ]
        if len(attachment_parts) > max_attachments:
            raise HTTPException(
                status_code=413,
                detail=f"Web answer accepts at most {max_attachments} attachments per message",
            )
        attachment_inputs: list[tuple[str, str | None, bytes]] = []
        total_bytes = 0
        for item in attachment_parts:
            if item.size is not None and item.size > max_attachment_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Attachment exceeds the {max_attachment_bytes}-byte limit",
                )
            payload = await item.read(max_attachment_bytes + 1)
            if len(payload) > max_attachment_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Attachment exceeds the {max_attachment_bytes}-byte limit",
                )
            total_bytes += len(payload)
            if total_bytes > max_total_attachment_bytes:
                raise HTTPException(
                    status_code=413,
                    detail="Attachments exceed the total size limit",
                )
            attachment_inputs.append((str(item.filename), item.content_type, payload))
        check_answer_image_count(
            image_count=sum(
                classify_web_attachment(filename, mime_type) == "image"
                for filename, mime_type, _payload in attachment_inputs
            ),
            configured_ceiling=(
                answer_image_capability.configured_ceiling
                if answer_image_capability is not None
                else 0
            ),
        )
        try:
            attachments = await asyncio.to_thread(
                validate_web_attachments,
                attachment_inputs,
                max_attachments=max_attachments,
                max_attachment_bytes=max_attachment_bytes,
                max_total_attachment_bytes=max_total_attachment_bytes,
                image_max_pixels=image_max_pixels,
            )
            workspaces_raw = _json_list(form.get("workspaces"), field="workspaces")
            body = WebAnswerRequest(
                query=str(form.get("query") or ""),
                workspaces=[str(item) for item in workspaces_raw] or None,
                conversation_id=_optional_uuid(form.get("conversation_id")),
                submission_id=UUID(str(form.get("submission_id"))),
                mode=_optional_mode(form.get("mode")),
                requested_skill=_optional_skill(form.get("requested_skill")),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ParsedWebAnswerRequest(
            query=body.query,
            workspaces=body.workspaces,
            conversation_id=body.conversation_id,
            submission_id=body.submission_id,
            attachments=attachments,
            mode=body.mode,
            requested_skill=body.requested_skill,
        )
    finally:
        await form.close()


__all__ = ["ParsedWebAnswerRequest", "parse_web_answer_request"]
