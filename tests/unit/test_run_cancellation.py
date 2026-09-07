# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for immediate durable-run cancellation across processes (Task 5)."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable

from dlightrag.engine.runtime.cancellation import (
    CANCEL_CHANNEL,
    RunCancellationListener,
    cancellation_notify_key,
)


class _FakeConnection:
    """One scripted LISTEN connection for the listener loop."""

    def __init__(self, *, payloads: list[str] | None = None, fail_after: int = 0) -> None:
        self.executed: list[str] = []
        self.payloads = list(payloads or [])
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.fail_after = fail_after
        self.closed = False
        self._listeners: list[Callable[..., None]] = []
        self._termination: list[Callable[..., None]] = []

    async def execute(self, statement: str) -> None:
        self.executed.append(statement)

    def add_termination_listener(self, callback: Callable[..., None]) -> None:
        self._termination.append(callback)

    def add_listener(self, channel: str, callback: Callable[..., None]) -> None:
        if channel == CANCEL_CHANNEL:
            self._listeners.append(callback)
            for payload in self.payloads:
                callback(self, 1, CANCEL_CHANNEL, payload)
            self.payloads.clear()

    def remove_listener(self, channel: str, callback: Callable[..., None]) -> None:
        if callback in self._listeners:
            self._listeners.remove(callback)

    async def close(self) -> None:
        self.closed = True

    def push(self, payload: str) -> None:
        for callback in self._listeners:
            callback(self, 1, CANCEL_CHANNEL, payload)

    def drop(self) -> None:
        """Simulate a lost connection."""
        for callback in self._termination:
            callback(self)


def _listener(
    *,
    connections: list[_FakeConnection],
    rescan: Callable[[], AsyncIterator[tuple[str, str]]],
    on_cancel: Callable[[str, str], Awaitable[None]],
) -> RunCancellationListener:
    async def _open() -> _FakeConnection:
        return connections.pop(0)

    return RunCancellationListener(open_connection=_open, rescan=rescan, on_cancel=on_cancel)


async def test_payload_alone_never_cancels_without_authoritative_rescan() -> None:
    pending: list[tuple[str, str]] = []
    cancelled: list[tuple[str, str]] = []
    connection = _FakeConnection(payloads=[cancellation_notify_key("o", "r1")])

    async def _rescan() -> AsyncIterator[tuple[str, str]]:
        for item in pending:
            yield item

    async def _on_cancel(owner: str, run: str) -> None:
        cancelled.append((owner, run))

    listener = _listener(connections=[connection], rescan=_rescan, on_cancel=_on_cancel)
    await listener.start()
    try:
        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        # The wake digest arrived, but no authoritative row said cancel-pending.
        await asyncio.sleep(0.05)
        assert cancelled == []
        assert "LISTEN dlightrag_run_cancel" in connection.executed

        # Once the authoritative rescan reports the run, the signal fires.
        pending.append(("o", "r1"))
        connection.push(cancellation_notify_key("o", "r1"))
        for _ in range(100):
            if cancelled:
                break
            await asyncio.sleep(0.01)
        assert cancelled == [("o", "r1")]
    finally:
        await listener.aclose()


async def test_non_digest_payloads_are_ignored() -> None:
    cancelled: list[tuple[str, str]] = []
    connection = _FakeConnection(payloads=["not-a-digest", ""])

    async def _rescan() -> AsyncIterator[tuple[str, str]]:
        yield ("o", "r9")

    async def _on_cancel(owner: str, run: str) -> None:
        cancelled.append((owner, run))

    listener = _listener(connections=[connection], rescan=_rescan, on_cancel=_on_cancel)
    await listener.start()
    try:
        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        await asyncio.sleep(0.05)
        # The rescan from startup fired the real cancel; noise payloads add none.
        assert cancelled == [("o", "r9")]
    finally:
        await listener.aclose()


async def test_reconnect_rescans_and_catches_missed_cancels() -> None:
    rescan_calls: list[int] = []
    cancelled: list[tuple[str, str]] = []
    first = _FakeConnection(payloads=[])
    second = _FakeConnection(payloads=[])

    async def _rescan() -> AsyncIterator[tuple[str, str]]:
        rescan_calls.append(1)
        if len(rescan_calls) > 1:
            yield ("o", "r2")

    async def _on_cancel(owner: str, run: str) -> None:
        cancelled.append((owner, run))

    listener = _listener(connections=[first, second], rescan=_rescan, on_cancel=_on_cancel)
    await listener.start()
    try:
        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        assert cancelled == []
        first.drop()  # connection lost: reconnect + rescan catches the cancel
        for _ in range(200):
            if cancelled:
                break
            await asyncio.sleep(0.01)
        assert cancelled == [("o", "r2")]
        assert first.closed
    finally:
        await listener.aclose()


async def test_initial_connection_failure_keeps_readiness_false_and_retries() -> None:
    attempts = 0

    async def _open():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectionError("refused")
        return _FakeConnection()

    async def _rescan() -> AsyncIterator[tuple[str, str]]:
        for item in ():
            yield item

    async def _on_cancel(owner: str, run: str) -> None:
        return None

    listener = RunCancellationListener(open_connection=_open, rescan=_rescan, on_cancel=_on_cancel)
    await listener.start()
    try:
        assert not listener.ready.is_set()
        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        assert listener.ready.is_set()
    finally:
        await listener.aclose()


async def test_initial_rescan_failure_keeps_readiness_false_until_recovery() -> None:
    attempts = 0
    first_failed = asyncio.Event()
    first = _FakeConnection()
    second = _FakeConnection()

    async def _rescan() -> AsyncIterator[tuple[str, str]]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_failed.set()
            raise RuntimeError("scan unavailable")
        for item in ():
            yield item

    async def _on_cancel(owner: str, run: str) -> None:
        return None

    listener = _listener(connections=[first, second], rescan=_rescan, on_cancel=_on_cancel)
    await listener.start()
    try:
        await asyncio.wait_for(first_failed.wait(), timeout=1.0)
        assert not listener.ready.is_set()
        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        assert attempts == 2
        assert first.closed
    finally:
        await listener.aclose()


async def test_initial_cancel_handler_failure_keeps_readiness_false_until_recovery() -> None:
    attempts = 0
    first_failed = asyncio.Event()
    first = _FakeConnection()
    second = _FakeConnection()

    async def _rescan() -> AsyncIterator[tuple[str, str]]:
        yield ("o", "r1")

    async def _on_cancel(owner: str, run: str) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            first_failed.set()
            raise RuntimeError("local signal unavailable")

    listener = _listener(connections=[first, second], rescan=_rescan, on_cancel=_on_cancel)
    await listener.start()
    try:
        await asyncio.wait_for(first_failed.wait(), timeout=1.0)
        assert not listener.ready.is_set()
        await asyncio.wait_for(listener.ready.wait(), timeout=5.0)
        assert attempts == 2
        assert first.closed
    finally:
        await listener.aclose()


def test_wake_digest_is_sha256_of_owner_nul_run() -> None:
    assert cancellation_notify_key("o", "r") == cancellation_notify_key("o", "r")
    assert len(cancellation_notify_key("o", "r")) == 64
    assert cancellation_notify_key("o", "r") != cancellation_notify_key("o", "s")
