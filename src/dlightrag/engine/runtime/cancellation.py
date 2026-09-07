# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Immediate durable-run cancellation across processes.

One dedicated reconnecting LISTEN connection per Runtime process watches a
fixed channel. A NOTIFY payload is only a wake digest — ``sha256(owner_id +
NUL + run_id)`` — never authorization: the listener re-reads the authoritative
owner/run/cancellation/lease/epoch row before signaling a local task. On every
reconnect the listener re-establishes LISTEN and repeats the locally leased
cancel-pending rescan, so a cancellation missed while disconnected is still
observed.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, cast

CANCEL_CHANNEL = "dlightrag_run_cancel"
_RECONNECT_BASE_SECONDS = 1.0
_RECONNECT_MAX_SECONDS = 30.0
_RESYNC_SECONDS = 30.0

logger = logging.getLogger(__name__)


def cancellation_notify_key(owner_id: str, run_id: str) -> str:
    """Return the wake digest payload for one cancellation notification."""
    return hashlib.sha256(f"{owner_id}\0{run_id}".encode()).hexdigest()


class RunCancellationListener:
    """Own the Runtime process's dedicated LISTEN connection and rescan loop."""

    def __init__(
        self,
        *,
        open_connection: Callable[[], Awaitable[object]],
        rescan: Callable[[], AsyncIterator[tuple[str, str]]],
        on_cancel: Callable[[str, str], Awaitable[None]],
    ) -> None:
        self._open_connection = open_connection
        self._rescan = rescan
        self._on_cancel = on_cancel
        self._ready = asyncio.Event()
        self._closing = False
        self._task: asyncio.Task[None] | None = None

    @property
    def ready(self) -> asyncio.Event:
        return self._ready

    @property
    def is_started(self) -> bool:
        return self._task is not None and not self._task.done()

    async def start(self) -> None:
        """Begin the LISTEN/reconnect loop; callers await :attr:`ready`."""
        if self._task is not None:
            return
        self._closing = False
        self._task = asyncio.create_task(self._run_forever())

    async def aclose(self) -> None:
        self._closing = True
        task = self._task
        if task is not None:
            task.cancel()
            with __import__("contextlib").suppress(asyncio.CancelledError):
                await task
        self._task = None

    async def _run_forever(self) -> None:
        backoff = _RECONNECT_BASE_SECONDS
        while not self._closing:
            try:
                await self._listen_once()
                backoff = _RECONNECT_BASE_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning(
                    "Run cancellation listener failed; retrying in %.1fs",
                    backoff,
                    exc_info=True,
                )
                self._ready.clear()
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX_SECONDS)

    async def _listen_once(self) -> None:
        self._ready.clear()
        connection: Any = await self._open_connection()
        try:
            execute = connection.execute
            await execute(f"LISTEN {CANCEL_CHANNEL}")
            # Initial and reconnect scans are part of readiness: both the scan
            # and every local signal must succeed before this listener is safe.
            await self._rescan_cancel_pending(best_effort=False)
            self._ready.set()
            while not self._closing:
                notification = await self._wait_for_notification(connection)
                if notification is None:
                    break  # connection lost; reconnect and rescan
                await self._handle_notification(str(notification))
        finally:
            close = getattr(connection, "close", None)
            if close is not None:
                await close()

    async def _wait_for_notification(self, connection: Any) -> object | None:
        """Block for one channel notification; None means the connection died."""
        waiter = getattr(connection, "add_listener", None)
        queue: asyncio.Queue[object | None] = asyncio.Queue()
        if waiter is None:
            return None

        def _enqueue(conn: object, pid: object, channel: str, payload: str) -> None:
            if channel == CANCEL_CHANNEL:
                queue.put_nowait((conn, pid, payload))

        def _terminated(_conn: object) -> None:
            self._ready.clear()
            queue.put_nowait(None)

        add_termination = getattr(connection, "add_termination_listener", None)
        if add_termination is not None:
            result = add_termination(_terminated)
            if hasattr(result, "__await__"):
                await result
        registered = waiter(CANCEL_CHANNEL, _enqueue)
        if hasattr(registered, "__await__"):
            await registered
        try:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=_RESYNC_SECONDS)
            except TimeoutError:
                # Once ready, periodic rescans remain best-effort: the next
                # notification, rescan, or reconnect can retry a missed signal.
                await self._rescan_cancel_pending(best_effort=True)
                return b""  # sentinel: keep listening, nothing to handle
            if item is None:
                return None  # connection terminated; reconnect and rescan
            _conn, _pid, payload = cast(tuple[Any, Any, str], item)
            return payload
        finally:
            try:
                remove_listener = connection.remove_listener
                removed = remove_listener(CANCEL_CHANNEL, _enqueue)
                if hasattr(removed, "__await__"):
                    await removed
            except Exception:
                logger.debug("Listener removal failed during shutdown", exc_info=True)

    async def _handle_notification(self, payload: str) -> None:
        if not payload or len(payload) != 64:
            return  # a wake digest is a 64-char hex; anything else is noise
        # The payload never cancels a task directly: only the authoritative
        # rescan decides which locally leased runs to signal.
        await self._rescan_cancel_pending(best_effort=True)

    async def _rescan_cancel_pending(self, *, best_effort: bool) -> None:
        try:
            async for owner_id, run_id in self._rescan():
                await self._on_cancel(owner_id, run_id)
        except Exception:
            if not best_effort:
                raise
            logger.warning("Run cancel-pending rescan failed", exc_info=True)


__all__ = [
    "CANCEL_CHANNEL",
    "RunCancellationListener",
    "cancellation_notify_key",
]
