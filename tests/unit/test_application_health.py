# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Process-health state, readiness aggregation, and cache transitions."""

import asyncio

from dlightrag.application.health import ApplicationHealth


async def test_readiness_is_single_flight_and_invalidated_by_state_transitions() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    calls = 0

    async def probe() -> str | None:
        nonlocal calls
        calls += 1
        started.set()
        await release.wait()
        return None

    health = ApplicationHealth(readiness_probe=probe, readiness_cache_seconds=60.0)
    assert await health.readiness_detail() == "RAG service is not ready"

    health.mark_ready()
    first = asyncio.create_task(health.readiness_detail())
    second = asyncio.create_task(health.readiness_detail())
    await started.wait()
    release.set()
    assert await asyncio.gather(first, second) == [None, None]
    assert calls == 1

    health.mark_component_degraded("corpus_storage")
    assert health.is_ready is True
    assert health.is_degraded is True
    assert health.warnings == ("Corpus storage unavailable",)
    assert await health.readiness_detail() is None
    assert calls == 1

    health.mark_not_ready()
    assert await health.readiness_detail() == "RAG service is not ready"
    health.mark_ready()
    assert await health.readiness_detail() is None
    assert calls == 2

    health.mark_closed()
    assert await health.readiness_detail() == "RAG service is not ready"


def test_health_owns_image_capability_summary() -> None:
    health = ApplicationHealth(readiness_probe=None)
    summary = {"status": "supported", "effective_max_images": 8}

    health.set_answer_image_capability(summary)

    assert health.answer_image_capability == summary


def test_health_owns_bounded_search_tool_provenance() -> None:
    health = ApplicationHealth(readiness_probe=None)
    health.set_search_toolchain(
        {
            "fd": {
                "path": "/opt/bin/fd",
                "version": "10.5.0",
                "sha256": "a" * 64,
                "ignored": "value",
            },
            "unknown": {"path": "/tmp/unknown"},
        }
    )

    assert health.search_toolchain == {
        "fd": {"path": "/opt/bin/fd", "version": "10.5.0", "sha256": "a" * 64}
    }


def test_health_defaults_image_capability_to_fail_closed_unknown() -> None:
    health = ApplicationHealth(readiness_probe=None)

    assert health.answer_image_capability == {
        "status": "unknown",
        "effective_max_images": 0,
        "configured_ceiling": 0,
        "model": None,
    }


def test_closing_preserves_degraded_diagnostics() -> None:
    health = ApplicationHealth(readiness_probe=None)
    health.mark_degraded("startup failed")

    health.mark_closed()
    health.mark_ready()
    health.mark_degraded("late transition")

    assert health.is_closed is True
    assert health.is_ready is False
    assert health.is_degraded is True
    assert health.warnings == ("Corpus storage unavailable",)


def test_component_view_is_bounded_and_recovery_clears_stale_warning() -> None:
    health = ApplicationHealth(readiness_probe=None)
    health.mark_ready()
    health.mark_component_degraded("providers")

    assert health.components["providers"] == {
        "status": "degraded",
        "detail": "Model providers unavailable",
    }
    assert health.is_ready is True

    health.mark_component_healthy("providers")

    assert health.components["providers"] == {"status": "healthy"}
    assert "Model providers unavailable" not in health.warnings
    assert len(health.components) == 7


async def test_transition_discards_an_inflight_pre_transition_verdict() -> None:
    old_started = asyncio.Event()
    release_old = asyncio.Event()
    calls = 0

    async def probe() -> str | None:
        nonlocal calls
        calls += 1
        if calls == 1:
            old_started.set()
            await release_old.wait()
            return None
        return "new failure"

    health = ApplicationHealth(readiness_probe=probe, readiness_cache_seconds=60.0)
    health.mark_ready()
    old_waiter = asyncio.create_task(health.readiness_detail())
    await old_started.wait()

    health.mark_ready()
    assert await health.readiness_detail() == "new failure"
    release_old.set()
    assert await old_waiter is None

    assert await health.readiness_detail() == "new failure"
    assert calls == 2
