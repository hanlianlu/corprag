# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Bounded process/component health and Operational State readiness."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Literal

type ReadinessProbe = Callable[[], Awaitable[str | None]]
type HealthComponentName = Literal[
    "process",
    "operational_state",
    "run_coordinator",
    "cancellation_listener",
    "corpus_storage",
    "parser",
    "providers",
]
type HealthComponentStatus = Literal["healthy", "degraded", "starting", "unknown", "stopped"]

_COMPONENT_ORDER: tuple[HealthComponentName, ...] = (
    "process",
    "operational_state",
    "run_coordinator",
    "cancellation_listener",
    "corpus_storage",
    "parser",
    "providers",
)
_COMPONENT_DETAILS: dict[HealthComponentName, str] = {
    "process": "Process is stopping",
    "operational_state": "Operational State unavailable",
    "run_coordinator": "Run coordinator unavailable",
    "cancellation_listener": "Run cancellation listener unavailable",
    "corpus_storage": "Corpus storage unavailable",
    "parser": "Document parser unavailable",
    "providers": "Model providers unavailable",
}


class _ReadinessCache:
    def __init__(self, ttl_seconds: float) -> None:
        self._ttl = max(0.0, float(ttl_seconds))
        self._deadline = 0.0
        self._detail: str | None = None
        self._probe: asyncio.Task[str | None] | None = None
        self._generation = 0

    async def detail(self, probe: ReadinessProbe) -> str | None:
        if time.monotonic() < self._deadline:
            return self._detail
        probing = self._probe
        if probing is None:
            generation = self._generation
            probing = self._probe = asyncio.ensure_future(probe())
            probing.add_done_callback(
                lambda task, generation=generation: self._memoize(task, generation)
            )
        return await asyncio.shield(probing)

    def _memoize(self, probing: asyncio.Task[str | None], generation: int) -> None:
        if self._probe is probing:
            self._probe = None
        if probing.cancelled() or probing.exception() is not None:
            return
        if generation != self._generation:
            return
        self._detail = probing.result()
        self._deadline = time.monotonic() + self._ttl

    def invalidate(self) -> None:
        self._generation += 1
        self._deadline = 0.0
        self._probe = None


class ApplicationHealth:
    """Own liveness, a fixed component view, and control-plane readiness."""

    def __init__(
        self,
        *,
        readiness_probe: ReadinessProbe | None,
        readiness_cache_seconds: float = 2.0,
    ) -> None:
        self._readiness_probe = readiness_probe
        self._readiness = _ReadinessCache(readiness_cache_seconds)
        self._ready = False
        self._closed = False
        self._components: dict[HealthComponentName, HealthComponentStatus] = {
            "process": "healthy",
            "operational_state": "starting",
            "run_coordinator": "starting",
            "cancellation_listener": "starting",
            "corpus_storage": "unknown",
            "parser": "unknown",
            "providers": "unknown",
        }
        self._answer_image_capability: dict[str, object] = {
            "status": "unknown",
            "effective_max_images": 0,
            "configured_ceiling": 0,
            "model": None,
        }

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def is_degraded(self) -> bool:
        return any(status == "degraded" for status in self._components.values())

    @property
    def is_closed(self) -> bool:
        return self._closed

    @property
    def warnings(self) -> tuple[str, ...]:
        """Return at most one fixed public detail for each known component."""
        return tuple(
            _COMPONENT_DETAILS[name]
            for name in _COMPONENT_ORDER
            if self._components[name] == "degraded"
        )

    @property
    def components(self) -> Mapping[str, Mapping[str, str]]:
        """Return the bounded, I/O-free public component projection."""
        return {
            name: {
                "status": status,
                **(
                    {"detail": _COMPONENT_DETAILS[name]}
                    if status in {"degraded", "stopped"}
                    else {}
                ),
            }
            for name in _COMPONENT_ORDER
            if (status := self._components[name])
        }

    @property
    def answer_image_capability(self) -> Mapping[str, object]:
        return dict(self._answer_image_capability)

    def add_warning(self, warning: str) -> None:
        """Compatibility shim: record a bounded generic dependency warning."""
        if warning:
            self.mark_component_degraded("corpus_storage")

    def mark_component_degraded(self, component: HealthComponentName) -> None:
        if self._closed or component == "process":
            return
        self._components[component] = "degraded"
        if component == "operational_state":
            self._ready = False
            self._readiness.invalidate()

    def mark_component_healthy(self, component: HealthComponentName) -> None:
        if self._closed:
            return
        self._components[component] = "healthy"

    def mark_ready(self) -> None:
        """Mark only Operational State admission ready; keep dependency degradation."""
        if self._closed:
            return
        self._ready = True
        self._components["operational_state"] = "healthy"
        self._readiness.invalidate()

    def mark_not_ready(self) -> None:
        if self._closed:
            return
        self._ready = False
        self._components["operational_state"] = "degraded"
        self._readiness.invalidate()

    def mark_degraded(self, warning: str | None = None) -> None:
        """Compatibility transition for a non-authoritative dependency outage."""
        if self._closed:
            return
        self.mark_component_degraded("corpus_storage")

    def mark_closed(self) -> None:
        self._ready = False
        self._closed = True
        self._components["process"] = "stopped"
        self._readiness.invalidate()

    def set_answer_image_capability(self, summary: Mapping[str, object]) -> None:
        self._answer_image_capability = dict(summary)

    async def readiness_detail(self) -> str | None:
        if not self._ready or self._closed:
            self._readiness.invalidate()
            return "RAG service is not ready"
        if self._readiness_probe is None:
            return None
        return await self._readiness.detail(self._readiness_probe)


__all__ = [
    "ApplicationHealth",
    "HealthComponentName",
    "HealthComponentStatus",
    "ReadinessProbe",
]
