# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable Answer run event streaming shared by every HTTP transport.

Resuming from a cursor, keeping a quiet run's connection alive, and detaching a
subscriber are properties of the durable event log, not of any one transport, so
they live here once. A transport contributes only how it renders one event: REST
serves the canonical result, the browser serves rendered presentation.

Nothing here writes durable state. Closing the returned generator detaches one
subscriber and never cancels the run.
"""

import asyncio
import contextlib
import json
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from typing import Any

from fastapi import HTTPException, Request

from dlightrag.application.runs import RunEvent
from dlightrag.engine.answer.client_contracts import model_dump_json_safe

#: A queued or quiet run keeps its connection alive with comments, not events.
SSE_KEEPALIVE_SECONDS = 10.0
KEEPALIVE_FRAME = ": keepalive\n\n"

#: How one transport turns a durable event into the frame its clients read.
type EventRenderer = Callable[[RunEvent], str]


def resume_cursor(request: Request) -> int:
    """Resolve the durable sequence this subscriber resumes after.

    An empty ``Last-Event-ID`` is how a browser reports "no cursor yet"; an
    explicitly supplied ``after`` is always parsed strictly.
    """
    header = request.headers.get("Last-Event-ID")
    query = request.query_params.get("after")
    from_header = _parse_cursor(header) if header else None
    from_query = _parse_cursor(query) if query is not None else None
    if from_header is not None and from_query is not None and from_header != from_query:
        raise HTTPException(
            status_code=400, detail="Last-Event-ID and 'after' request different cursors"
        )
    if from_query is not None:
        return from_query
    return from_header or 0


def _parse_cursor(value: str | None) -> int:
    if value is None or not value.isdigit():
        raise HTTPException(status_code=400, detail="Event cursor must be a non-negative integer")
    return int(value)


def sse_frame(*, sequence: int, event_type: str, payload: Any) -> str:
    """Serialize one durable event, keyed by the sequence a reconnect resumes from."""
    data = json.dumps(model_dump_json_safe(payload), ensure_ascii=False)
    return f"id: {sequence}\nevent: {event_type}\ndata: {data}\n\n"


async def follow_run_frames(
    events: AsyncIterator[RunEvent], render: EventRenderer
) -> AsyncGenerator[str]:
    """Replay and follow one run's durable events as this transport's frames.

    The pending read survives every keepalive, so a comment written while the run
    is quiet neither restarts nor loses the event that arrives next.
    """
    iterator = events.__aiter__()
    pending: asyncio.Task[RunEvent] | None = None
    try:
        while True:
            if pending is None:
                pending = asyncio.ensure_future(anext(iterator))
            try:
                event = await asyncio.wait_for(asyncio.shield(pending), SSE_KEEPALIVE_SECONDS)
            except TimeoutError:
                yield KEEPALIVE_FRAME
                continue
            except StopAsyncIteration:
                pending = None
                return
            pending = None
            yield render(event)
    finally:
        if pending is not None:
            pending.cancel()
            # A cancelled child is returned as a value; cancellation of this
            # enclosing subscriber still propagates instead of being swallowed.
            await asyncio.gather(pending, return_exceptions=True)
        with contextlib.suppress(Exception):
            await events.aclose()  # pyright: ignore[reportAttributeAccessIssue]


__all__ = [
    "KEEPALIVE_FRAME",
    "SSE_KEEPALIVE_SECONDS",
    "EventRenderer",
    "follow_run_frames",
    "resume_cursor",
    "sse_frame",
]
