# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Durable run event subscriptions.

A subscriber replays every committed event from its cursor and then follows the
run until its terminal event. PostgreSQL is authoritative: local notification
only shortens the wait between polls, so a subscriber in another process sees
exactly the same gap-free sequence. Detaching closes one subscriber and never
touches run state.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator
from typing import Protocol

from dlightrag.engine.runtime.records import RunEvent

#: Poll cadence for a quiet run; a local commit wakes its subscribers sooner.
EVENT_POLL_SECONDS = 1.0

_TERMINAL_EVENTS = ("done", "error")


class EventReader(Protocol):
    """The owner-scoped, bounded event replay a subscriber may use."""

    async def read_event_page(
        self, *, owner_id: str, run_id: str, after_sequence: int = 0
    ) -> tuple[RunEvent, ...]: ...


class RunEventBroker:
    """Wake local subscribers as soon as this process commits an event."""

    def __init__(self) -> None:
        self._waiters: dict[tuple[str, str], set[asyncio.Event]] = {}

    def notify(self, owner_id: str, run_id: str) -> None:
        for waiter in self._waiters.get((owner_id, run_id), ()):
            waiter.set()

    @contextlib.contextmanager
    def waiter(self, owner_id: str, run_id: str) -> Iterator[asyncio.Event]:
        key = (owner_id, run_id)
        event = asyncio.Event()
        self._waiters.setdefault(key, set()).add(event)
        try:
            yield event
        finally:
            waiters = self._waiters.get(key)
            if waiters is not None:
                waiters.discard(event)
                if not waiters:
                    del self._waiters[key]


async def follow_run_events(
    store: EventReader,
    broker: RunEventBroker,
    *,
    owner_id: str,
    run_id: str,
    after_sequence: int = 0,
    is_finished: Callable[[], Awaitable[bool]],
) -> AsyncGenerator[RunEvent]:
    """Replay committed events after ``after_sequence``, then follow the run.

    The waiter is armed before every read, so an event committed while a page is
    being read wakes the next wait instead of being missed, and a page is always
    drained completely before waiting again. A run that reports finished is
    re-read once, which closes the race between the last page read and the
    terminal transition; an unknown run therefore ends immediately.
    """
    cursor = max(0, int(after_sequence))
    closing = False
    with broker.waiter(owner_id, run_id) as woken:
        while True:
            woken.clear()
            page = await store.read_event_page(
                owner_id=owner_id, run_id=run_id, after_sequence=cursor
            )
            for event in page:
                cursor = event.sequence
                yield event
                if event.event_type in _TERMINAL_EVENTS:
                    return
            if page:
                closing = False
                continue
            if closing:
                return
            if await is_finished():
                closing = True
                continue
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(woken.wait(), timeout=EVENT_POLL_SECONDS)


__all__ = [
    "EVENT_POLL_SECONDS",
    "EventReader",
    "RunEventBroker",
    "follow_run_events",
]
