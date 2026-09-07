# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Behavioral contract for the RAG-owned workspace runtime pool."""

import asyncio
from typing import Any, cast
from unittest.mock import AsyncMock

import pytest

from dlightrag.engine.rag.workspace.pool import WorkspacePool, WorkspaceUnavailableError
from dlightrag.engine.rag.workspace.ports import CorpusSchemaError
from dlightrag.engine.rag.workspace.workspace_rag import WorkspaceRag


def _pool(build, *, clock=lambda: 0.0) -> WorkspacePool:
    return WorkspacePool(build=build, clock=clock)


async def test_concurrent_acquire_creates_one_runtime_and_clears_backoff() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    calls = 0

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0)
        return runtime

    pool = _pool(build)
    first, second, third = await asyncio.gather(
        pool.acquire("research"),
        pool.acquire("research"),
        pool.acquire("research"),
    )

    assert (first, second, third) == (runtime, runtime, runtime)
    assert calls == 1


async def test_schema_error_passes_through_without_backoff() -> None:
    async def build(*_args: Any) -> WorkspaceRag:
        raise CorpusSchemaError("missing schema")

    pool = _pool(build)

    with pytest.raises(CorpusSchemaError, match="missing schema"):
        await pool.acquire("research")
    with pytest.raises(CorpusSchemaError, match="missing schema"):
        await pool.acquire("research")


async def test_retryable_failure_backs_off_then_success_clears_it() -> None:
    now = 0.0
    runtime = cast(WorkspaceRag, AsyncMock())
    calls = 0

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise ConnectionError("down")
        return runtime

    pool = _pool(build, clock=lambda: now)
    with pytest.raises(WorkspaceUnavailableError, match="ConnectionError"):
        await pool.acquire("research")
    with pytest.raises(WorkspaceUnavailableError, match="backoff"):
        await pool.acquire("research")

    now = 16.0
    assert await pool.acquire("research") is runtime
    assert await pool.acquire("research") is runtime
    assert calls == 2


async def test_retry_callback_clears_bounded_workspace_degradation() -> None:
    now = 0.0
    runtime = cast(WorkspaceRag, AsyncMock())
    outcomes = [ConnectionError("down"), runtime]
    unavailable: list[str] = []
    available: list[str] = []

    async def build(*_args: Any) -> WorkspaceRag:
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    pool = WorkspacePool(
        build=build,
        clock=lambda: now,
        initial_backoff_seconds=1,
        max_backoff_seconds=1,
        on_workspace_unavailable=unavailable.append,
        on_workspace_available=available.append,
    )
    with pytest.raises(WorkspaceUnavailableError):
        await pool.acquire("default")
    now = 1.0

    assert await pool.acquire("default") is runtime
    assert unavailable == ["default"]
    assert available == ["default"]


async def test_retryable_failure_backoff_grows_and_caps_at_five_minutes() -> None:
    now = 0.0
    calls = 0

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal calls
        calls += 1
        raise ConnectionError("down")

    pool = _pool(build, clock=lambda: now)
    for expected_interval in (15, 30, 60, 120, 240, 300, 300):
        with pytest.raises(WorkspaceUnavailableError, match="ConnectionError"):
            await pool.acquire("research")
        calls_after_failure = calls
        with pytest.raises(
            WorkspaceUnavailableError,
            match=rf"retry in {expected_interval}s",
        ):
            await pool.acquire("research")
        assert calls == calls_after_failure
        now += expected_interval + 1


async def test_stale_flight_callback_cannot_remove_a_retry_flight() -> None:
    now = 0.0
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    second_started = asyncio.Event()
    release_second = asyncio.Event()
    runtime = cast(WorkspaceRag, AsyncMock())
    calls = 0

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            await release_first.wait()
            raise ConnectionError("down")
        second_started.set()
        await release_second.wait()
        return runtime

    pool = _pool(build, clock=lambda: now)
    first_waiter = asyncio.create_task(pool.acquire("research"))
    await first_started.wait()
    first_flight = pool._workspace_flights["research"]
    release_first.set()
    with pytest.raises(WorkspaceUnavailableError, match="ConnectionError"):
        await first_waiter

    now = 16.0
    second_waiter = asyncio.create_task(pool.acquire("research"))
    await second_started.wait()
    second_flight = pool._workspace_flights["research"]
    assert second_flight is not first_flight

    pool._finish_workspace_flight("research", first_flight)
    assert pool._workspace_flights["research"] is second_flight

    release_second.set()
    assert await second_waiter is runtime


async def test_evict_closes_before_recreating_runtime() -> None:
    runtimes = [cast(WorkspaceRag, AsyncMock()), cast(WorkspaceRag, AsyncMock())]

    async def build(*_args: Any) -> WorkspaceRag:
        return runtimes.pop(0)

    pool = _pool(build)
    first = await pool.acquire("research")
    await pool.evict("research")
    second = await pool.acquire("research")

    cast(AsyncMock, first.aclose).assert_awaited_once()
    assert second is not first


async def test_close_is_idempotent_and_rejects_future_acquire() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())

    async def build(*_args: Any) -> WorkspaceRag:
        return runtime

    pool = _pool(build)
    await pool.acquire("research")
    await pool.aclose()
    await pool.aclose()

    cast(AsyncMock, runtime.aclose).assert_awaited_once()
    with pytest.raises(WorkspaceUnavailableError, match="closed"):
        await pool.acquire("research")


async def test_one_thousand_warm_callers_share_one_workspace_flight() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    calls = 0

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal calls
        calls += 1
        build_started.set()
        await release_build.wait()
        return runtime

    pool = _pool(build)
    warmups = [asyncio.create_task(pool.warm(["research", "research"])) for _ in range(1000)]
    await asyncio.wait_for(build_started.wait(), timeout=1)

    assert calls == 1
    assert len(pool._workspace_flights) == 1

    release_build.set()
    await asyncio.gather(*warmups)
    assert await pool.acquire("research") is runtime
    assert calls == 1
    assert not pool._workspace_flights


async def test_overlapping_warm_callers_create_distinct_workspace_flights_only() -> None:
    active = 0
    peak = 0
    calls: dict[str, int] = {}
    first_wave_started = asyncio.Event()
    release = asyncio.Event()
    workspaces = tuple(f"workspace_{index}" for index in range(12))

    async def build(workspace: str) -> WorkspaceRag:
        nonlocal active, peak
        calls[workspace] = calls.get(workspace, 0) + 1
        active += 1
        peak = max(peak, active)
        if active == 8:
            first_wave_started.set()
        try:
            await release.wait()
        finally:
            active -= 1
        return cast(WorkspaceRag, AsyncMock())

    pool = WorkspacePool(build=build, warm_concurrency=8)
    warmups = [
        asyncio.create_task(pool.warm(workspaces if index % 2 else tuple(reversed(workspaces))))
        for index in range(1000)
    ]
    await asyncio.wait_for(first_wave_started.wait(), timeout=1)

    assert peak == 8
    assert len(calls) == 8
    assert len(pool._workspace_flights) == len(workspaces)

    release.set()
    await asyncio.gather(*warmups)
    assert calls == dict.fromkeys(workspaces, 1)


async def test_cancelling_warm_waiters_does_not_cancel_the_shared_flight() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    build_started = asyncio.Event()
    release_build = asyncio.Event()
    build_cancelled = False
    calls = 0

    async def build(*_args: Any) -> WorkspaceRag:
        nonlocal build_cancelled, calls
        calls += 1
        build_started.set()
        try:
            await release_build.wait()
        except asyncio.CancelledError:
            build_cancelled = True
            raise
        return runtime

    pool = _pool(build)
    waiters = [asyncio.create_task(pool.warm(["research"])) for _ in range(1000)]
    await asyncio.wait_for(build_started.wait(), timeout=1)

    for waiter in waiters[:-1]:
        waiter.cancel()
    cancelled = await asyncio.gather(*waiters[:-1], return_exceptions=True)
    assert all(isinstance(result, asyncio.CancelledError) for result in cancelled)
    assert not build_cancelled
    assert not waiters[-1].done()

    release_build.set()
    await waiters[-1]
    assert calls == 1
    assert await pool.acquire("research") is runtime


async def test_warm_limits_concurrency_to_eight() -> None:
    active = 0
    peak = 0
    started: set[str] = set()
    first_wave_started = asyncio.Event()
    release = asyncio.Event()

    async def build(workspace: str, *_args: Any) -> WorkspaceRag:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        started.add(workspace)
        if len(started) == 8:
            first_wave_started.set()
        await release.wait()
        active -= 1
        return cast(WorkspaceRag, AsyncMock())

    pool = _pool(build)
    warmups = [
        asyncio.create_task(pool.warm([f"workspace_{index}" for index in range(5)])),
        asyncio.create_task(pool.warm([f"workspace_{index}" for index in range(5, 10)])),
    ]
    await asyncio.wait_for(first_wave_started.wait(), timeout=1)

    assert peak == 8
    assert len(started) == 8

    release.set()
    await asyncio.gather(*warmups)
    assert len(started) == 10


async def test_warm_failure_does_not_cancel_an_independent_workspace_flight() -> None:
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()

    async def build(workspace: str, *_args: Any) -> WorkspaceRag:
        if workspace == "failed":
            await sibling_started.wait()
            raise ConnectionError("down")
        sibling_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            sibling_cancelled.set()
        raise AssertionError("unreachable")

    pool = _pool(build)
    with pytest.raises(WorkspaceUnavailableError, match="ConnectionError"):
        await pool.warm(["failed", "sibling"])

    assert not sibling_cancelled.is_set()
    await pool.aclose()
    assert sibling_cancelled.is_set()


async def test_close_cancels_and_joins_active_warmup() -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def build(*_args: Any) -> WorkspaceRag:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        raise AssertionError("unreachable")

    pool = _pool(build)
    warmup = asyncio.create_task(pool.warm(["research"]))
    await asyncio.wait_for(started.wait(), timeout=1)

    await pool.aclose()

    assert warmup.done()
    assert cancelled.is_set()
    await asyncio.gather(warmup, return_exceptions=True)


async def test_close_cancels_inflight_build_and_rejects_its_acquire() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    build_started = asyncio.Event()
    build_cancelled = asyncio.Event()

    async def build(*_args: Any) -> WorkspaceRag:
        build_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            build_cancelled.set()
        return runtime

    pool = _pool(build)
    acquire = asyncio.create_task(pool.acquire("research"))
    await asyncio.wait_for(build_started.wait(), timeout=1)
    await pool.aclose()

    with pytest.raises(WorkspaceUnavailableError, match="closed"):
        await acquire
    assert build_cancelled.is_set()
    cast(AsyncMock, runtime.aclose).assert_not_awaited()
    assert await pool.is_loaded("research") is False


async def test_close_settles_eviction_waiting_on_a_cancelled_flight() -> None:
    build_started = asyncio.Event()

    async def build(*_args: Any) -> WorkspaceRag:
        build_started.set()
        await asyncio.Event().wait()
        raise AssertionError("unreachable")

    pool = _pool(build)
    acquire = asyncio.create_task(pool.acquire("research"))
    await build_started.wait()
    eviction = asyncio.create_task(pool.evict("research"))
    await asyncio.sleep(0)

    await pool.aclose()
    await eviction
    with pytest.raises(WorkspaceUnavailableError, match="closed"):
        await acquire
    assert not pool._workspace_flights


async def test_close_closes_runtime_returned_by_a_cancellation_suppressing_builder() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    build_started = asyncio.Event()

    async def build(*_args: Any) -> WorkspaceRag:
        build_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            return runtime
        raise AssertionError("unreachable")

    pool = _pool(build)
    acquire = asyncio.create_task(pool.acquire("research"))
    await asyncio.wait_for(build_started.wait(), timeout=1)
    await pool.aclose()

    with pytest.raises(WorkspaceUnavailableError, match="closed"):
        await acquire
    cast(AsyncMock, runtime.aclose).assert_awaited_once()
    assert not pool._workspace_flights
    assert await pool.is_loaded("research") is False


async def test_cancelled_close_finishes_cleanup_before_propagating() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    cold_started = asyncio.Event()
    cancellation_cleanup_started = asyncio.Event()
    hold_cancellation_cleanup = asyncio.Event()

    async def build(workspace: str, *_args: Any) -> WorkspaceRag:
        if workspace == "loaded":
            return runtime
        cold_started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancellation_cleanup_started.set()
            await hold_cancellation_cleanup.wait()
        raise AssertionError("unreachable")

    pool = _pool(build)
    await pool.acquire("loaded")
    warmup = asyncio.create_task(pool.warm(["cold"]))
    await asyncio.wait_for(cold_started.wait(), timeout=1)
    close = asyncio.create_task(pool.aclose())
    await asyncio.wait_for(cancellation_cleanup_started.wait(), timeout=1)

    close.cancel()
    hold_cancellation_cleanup.set()
    with pytest.raises(asyncio.CancelledError):
        await close

    assert warmup.done()
    cast(AsyncMock, runtime.aclose).assert_awaited_once()
    await asyncio.gather(warmup, return_exceptions=True)
    await pool.aclose()
    cast(AsyncMock, runtime.aclose).assert_awaited_once()


async def test_cancelled_concurrent_close_waits_for_shared_cleanup() -> None:
    runtime = cast(WorkspaceRag, AsyncMock())
    close_started = asyncio.Event()
    release_close = asyncio.Event()

    async def build(*_args: Any) -> WorkspaceRag:
        return runtime

    async def close_runtime() -> None:
        close_started.set()
        await release_close.wait()

    runtime.aclose = AsyncMock(side_effect=close_runtime)  # type: ignore[method-assign]
    pool = _pool(build)
    await pool.acquire("research")
    first = asyncio.create_task(pool.aclose())
    await asyncio.wait_for(close_started.wait(), timeout=1)
    second = asyncio.create_task(pool.aclose())
    await asyncio.sleep(0)

    second.cancel()
    await asyncio.sleep(0)
    assert not second.done()

    release_close.set()
    await first
    with pytest.raises(asyncio.CancelledError):
        await second
    cast(AsyncMock, runtime.aclose).assert_awaited_once()


async def test_acquire_waits_while_evict_closes_loaded_runtime() -> None:
    first = cast(WorkspaceRag, AsyncMock())
    second = cast(WorkspaceRag, AsyncMock())
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    runtimes = [first, second]

    async def build(*_args: Any) -> WorkspaceRag:
        return runtimes.pop(0)

    async def close_first() -> None:
        close_started.set()
        await release_close.wait()

    first.aclose = AsyncMock(side_effect=close_first)  # type: ignore[method-assign]
    pool = _pool(build)
    assert await pool.acquire("research") is first
    eviction = asyncio.create_task(pool.evict("research"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    acquisition = asyncio.create_task(pool.acquire("research"))
    await asyncio.sleep(0)

    assert not acquisition.done()

    release_close.set()
    await eviction
    assert await acquisition is second


async def test_warm_waits_for_eviction_and_replaces_the_closing_runtime() -> None:
    first = cast(WorkspaceRag, AsyncMock())
    second = cast(WorkspaceRag, AsyncMock())
    close_started = asyncio.Event()
    release_close = asyncio.Event()
    runtimes = [first, second]

    async def build(*_args: Any) -> WorkspaceRag:
        return runtimes.pop(0)

    async def close_first() -> None:
        close_started.set()
        await release_close.wait()

    first.aclose = AsyncMock(side_effect=close_first)  # type: ignore[method-assign]
    pool = _pool(build)
    assert await pool.acquire("research") is first
    eviction = asyncio.create_task(pool.evict("research"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    warmup = asyncio.create_task(pool.warm(["research"]))
    await asyncio.sleep(0)

    assert not warmup.done()

    release_close.set()
    await eviction
    await warmup
    assert await pool.acquire("research") is second
