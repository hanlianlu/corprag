# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Root Profile Memory hard capability gate and epoch semantics."""

import pytest
from dlightrag_memory import MemoryProvenance
from dlightrag_memory.store import InMemoryMemoryStore

from dlightrag.application.memory import (
    InMemoryMemorySettingsStore,
    MemoryDisabledError,
    MemoryService,
)
from dlightrag.engine.answer.memory import MemoryCapability


def _service() -> MemoryService:
    return MemoryService(
        InMemoryMemoryStore(),
        settings_store=InMemoryMemorySettingsStore(),
        memory_list_cursor_secret=b"memory-service-list-test",
    )


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(origin_kind="management", origin_id="request-1")


async def test_disabled_owner_has_only_the_settings_control_plane() -> None:
    service = _service()
    owner = {"owner_id": "alpha", "auth_mode": "jwt"}
    disabled = await service.set_enabled(**owner, enabled=False)

    assert disabled.enabled is False
    assert disabled.active_count is None
    assert disabled.epoch == 1
    assert (await service.settings(**owner)).active_count is None
    with pytest.raises(MemoryDisabledError):
        await service.list_active_page(owner_id="alpha", auth_mode="jwt")
    with pytest.raises(MemoryDisabledError):
        await service.remember(
            **owner,
            kind="fact",
            body="Stable.",
            provenance=_provenance(),
            idempotency_key="request-1",
        )
    with pytest.raises(MemoryDisabledError):
        await service.clear(**owner)


async def test_mutation_rechecks_activation_inside_the_store_settlement() -> None:
    class DeactivatingSettings(InMemoryMemorySettingsStore):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        async def state(self, *, owner_id: str) -> MemoryCapability:
            self.reads += 1
            return MemoryCapability(enabled=self.reads == 1, epoch=int(self.reads > 1))

    store = InMemoryMemoryStore()
    service = MemoryService(
        store,
        settings_store=DeactivatingSettings(),
        memory_list_cursor_secret=b"memory-service-list-test",
    )
    with pytest.raises(MemoryDisabledError):
        await service.remember(
            owner_id="alpha",
            auth_mode="jwt",
            kind="fact",
            body="Stable.",
            provenance=_provenance(),
            idempotency_key="request-1",
        )
    records, cursor = await store.list_active_page(owner_id="alpha", limit=100)
    assert records == ()
    assert cursor is None


async def test_deactivation_and_clear_invalidate_existing_run_epochs() -> None:
    service = _service()
    owner = {"owner_id": "alpha", "auth_mode": "jwt"}
    initial = await service.settings(**owner)
    assert initial.enabled and initial.epoch == 0
    assert await service.capability_current(owner_id="alpha", epoch=0)

    disabled = await service.set_enabled(**owner, enabled=False)
    assert disabled.epoch == 1
    reenabled = await service.set_enabled(**owner, enabled=True)
    assert reenabled.epoch == 1
    assert not await service.capability_current(owner_id="alpha", epoch=0)
    assert await service.capability_current(owner_id="alpha", epoch=1)

    await service.remember(
        **owner,
        kind="preference",
        body="Use Chinese.",
        provenance=_provenance(),
        idempotency_key="request-1",
    )
    before_clear = await service.settings(**owner)
    assert before_clear.active_count == 1
    removed = await service.clear(**owner)
    assert removed == 1  # public count reports Profile Memory records, not journal rows
    after_clear = await service.settings(**owner)
    assert after_clear.enabled is True
    assert after_clear.epoch == 2
    assert after_clear.active_count == 0
    assert not await service.capability_current(owner_id="alpha", epoch=1)
