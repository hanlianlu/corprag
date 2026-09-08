# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""In-memory Profile Memory operation, receipt, and lifecycle semantics."""

import asyncio
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime

import pytest
from dlightrag_memory import Memory, MemoryOperation, MemoryProvenance, MemoryRecord
from dlightrag_memory.errors import MemoryWriteRejectedError
from dlightrag_memory.store import (
    InMemoryMemoryStore,
    operation_change_id,
    operation_record_id,
)


def _provenance(run_id: str = "run-1") -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run",
        origin_id=run_id,
        run_id=run_id,
        session_id="session-1",
    )


def _record(
    *, owner: str = "alpha", memory_id: str = "m1", body: str = "No email."
) -> MemoryRecord:
    now = datetime.now(UTC)
    return MemoryRecord(
        owner_id=owner,
        memory_id=memory_id,
        kind="preference",
        body=body,
        provenance=_provenance(),
        created_at=now,
        updated_at=now,
    )


async def _remember(
    memory: Memory,
    *,
    key: str,
    body: str = "No email.",
    kind: str = "preference",
    supersedes_id: str | None = None,
    scope: str | None = None,
    limit: int | None = None,
):
    return await memory.remember(
        owner_id="alpha",
        kind=kind,  # type: ignore[arg-type]
        body=body,
        provenance=_provenance(),
        idempotency_key=key,
        supersedes_id=supersedes_id,
        mutation_scope=scope,
        mutation_limit=limit,
    )


async def _active(memory: Memory, *, owner_id: str = "alpha") -> tuple[MemoryRecord, ...]:
    records, _ = await memory.browse(owner_id=owner_id, limit=100)
    return records


async def _store_active(
    store: InMemoryMemoryStore, *, owner_id: str = "alpha"
) -> tuple[MemoryRecord, ...]:
    records, _ = await store.list_active_page(owner_id=owner_id, limit=100)
    return records


async def test_owners_cannot_read_each_other() -> None:
    store = InMemoryMemoryStore()
    await store.insert(_record(owner="alpha", memory_id="m1"))
    await store.insert(_record(owner="beta", memory_id="m1", body="Other."))
    assert [row.body for row in await _store_active(store, owner_id="alpha")] == ["No email."]
    assert [row.body for row in await _store_active(store, owner_id="beta")] == ["Other."]


async def test_operation_replay_returns_the_original_receipt() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1")
    replay = await _remember(memory, key="call-1")

    assert first.outcome == "changed"
    assert replay == first
    assert len(await _active(memory)) == 1


async def test_owner_guard_rejects_before_journal_or_record_settlement() -> None:
    memory = Memory(InMemoryMemoryStore())

    async def reject(_settlement: object | None) -> None:
        raise MemoryWriteRejectedError("capability changed")

    with pytest.raises(MemoryWriteRejectedError, match="capability changed"):
        await memory.remember(
            owner_id="alpha",
            kind="fact",
            body="Lives in Gothenburg.",
            provenance=_provenance(),
            idempotency_key="call-1",
            guard=reject,
        )

    assert await _active(memory) == ()
    settled = await _remember(memory, key="call-1", body="Lives in Gothenburg.", kind="fact")
    assert settled.outcome == "changed"


async def test_reusing_an_idempotency_key_with_different_input_rejects() -> None:
    memory = Memory(InMemoryMemoryStore())
    await _remember(memory, key="call-1")

    with pytest.raises(MemoryWriteRejectedError, match="different input"):
        await _remember(memory, key="call-1", body="Use chat.")


async def test_semantic_duplicate_is_unchanged_without_another_record() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1", body="Use Chinese.")
    duplicate = await _remember(memory, key="call-2", body="  use chinese.  ")

    assert first.outcome == "changed"
    assert duplicate.outcome == "unchanged"
    assert duplicate.memory_id == first.memory_id
    assert await memory.count_active(owner_id="alpha") == 1


async def test_supersede_and_compensating_undo_preserve_history() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    old = await _remember(memory, key="call-1", body="Lives in Beijing.", kind="fact")
    replacement = await _remember(
        memory,
        key="call-2",
        body="Lives in Gothenburg.",
        kind="fact",
        supersedes_id=old.memory_id,
    )

    assert replacement.outcome == "changed"
    assert [row.body for row in await _active(memory)] == ["Lives in Gothenburg."]

    undone = await memory.undo(
        owner_id="alpha",
        change_id=replacement.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )

    assert undone.outcome == "changed"
    assert [row.body for row in await _active(memory)] == ["Lives in Beijing."]
    current = await store.get(owner_id="alpha", memory_id=replacement.memory_id or "")
    assert current is not None and current.status == "superseded"


async def test_forget_and_undo_restore_as_a_new_active_record() -> None:
    memory = Memory(InMemoryMemoryStore())
    remembered = await _remember(memory, key="call-1", body="Prefers concise answers.")
    forgotten = await memory.forget(
        owner_id="alpha",
        memory_id=remembered.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    assert forgotten.outcome == "changed"
    assert await _active(memory) == ()

    undone = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    assert undone.outcome == "changed"
    (restored,) = await _active(memory)
    assert restored.body == "Prefers concise answers."
    assert restored.memory_id != remembered.memory_id


async def test_stale_or_repeated_undo_conflicts() -> None:
    memory = Memory(InMemoryMemoryStore())
    remembered = await _remember(memory, key="call-1")
    first = await memory.undo(
        owner_id="alpha",
        change_id=remembered.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
        idempotency_key="undo-1",
    )
    second = await memory.undo(
        owner_id="alpha",
        change_id=remembered.change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-2"),
        idempotency_key="undo-2",
    )
    assert first.outcome == "changed"
    assert second.outcome == "conflict"


async def test_per_run_cap_counts_only_changed_mutations() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1", body="One.", scope="run-1", limit=2)
    duplicate = await _remember(memory, key="call-2", body="one.", scope="run-1", limit=2)
    await _remember(memory, key="call-3", body="Two.", scope="run-1", limit=2)

    assert first.changed
    assert duplicate.outcome == "unchanged"
    with pytest.raises(MemoryWriteRejectedError, match="mutation limit"):
        await _remember(memory, key="call-4", body="Three.", scope="run-1", limit=2)
    assert await memory.count_active(owner_id="alpha") == 2


async def test_clear_physically_removes_records_and_operation_replay() -> None:
    memory = Memory(InMemoryMemoryStore())
    first = await _remember(memory, key="call-1")
    assert await memory.clear(owner_id="alpha") == 1
    assert await _active(memory) == ()

    replay_after_clear = await _remember(memory, key="call-1")
    assert replay_after_clear.changed
    assert replay_after_clear.change_id == first.change_id
    assert await memory.count_active(owner_id="alpha") == 1


async def test_recall_searches_with_query() -> None:
    memory = Memory(InMemoryMemoryStore())
    await _remember(memory, key="call-1", body="No email.")
    await _remember(memory, key="call-2", body="Unrelated fact about trains.", kind="fact")

    result = await memory.recall(owner_id="alpha", query="email", top_k=5)

    assert result.strategy == "query_search"
    assert [record.body for record in result.records] == ["No email."]
    assert result.content_chars == len("No email.")


async def _undo(memory: Memory, change_id: str, *, key: str):
    return await memory.undo(
        owner_id="alpha",
        change_id=change_id,
        provenance=MemoryProvenance(origin_kind="undo", origin_id=f"undo-{key}"),
        idempotency_key=key,
    )


async def _forget_duplicate_rows(memory: Memory, store: InMemoryMemoryStore):
    await store.insert(_record(memory_id="r1", body="Prefers tea."))
    await store.insert(_record(memory_id="r2", body="  prefers tea.  "))
    await store.insert(_record(memory_id="r3", body="Likes trains."))
    forgotten = await memory.forget(
        owner_id="alpha",
        body="Prefers tea.",
        provenance=_provenance(),
        idempotency_key="forget-1",
    )
    assert forgotten.outcome == "changed"
    assert forgotten.memory_ids == ("r1", "r2")
    return forgotten


async def test_multi_row_forget_undo_restores_all_rows_in_order() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "changed"
    assert undone.memory_ids == (
        operation_record_id("alpha", undone.change_id, index=0),
        operation_record_id("alpha", undone.change_id, index=1),
    )
    restored = [
        record
        for memory_id in undone.memory_ids
        if (record := await store.get(owner_id="alpha", memory_id=memory_id)) is not None
    ]
    assert len(restored) == 2
    assert [record.supersedes_id for record in restored] == ["r1", "r2"]
    assert all(record.status == "active" for record in restored)
    assert all(record.provenance.origin_kind == "undo" for record in restored)
    assert all(record.created_at == undone.created_at for record in restored)
    assert all(record.updated_at == undone.created_at for record in restored)
    assert {record.body for record in restored} == {"Prefers tea.", "  prefers tea.  "}
    for old_id in ("r1", "r2"):
        old = await store.get(owner_id="alpha", memory_id=old_id)
        assert old is not None and old.status == "forgotten"
    assert sorted(record.body for record in await _active(memory)) == [
        "  prefers tea.  ",
        "Likes trains.",
        "Prefers tea.",
    ]


async def test_multi_row_forget_undo_late_wrong_state_row_conflicts_cleanly() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)

    drifted = await store.get(owner_id="alpha", memory_id="r2")
    assert drifted is not None
    store._rows[("alpha", "r2")] = replace(drifted, status="active")

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 2  # r3 plus the drifted r2
    reactivated = await store.get(owner_id="alpha", memory_id="r2")
    first_row = await store.get(owner_id="alpha", memory_id="r1")
    assert reactivated is not None and reactivated.status == "active"
    assert first_row is not None and first_row.status == "forgotten"
    assert ("alpha", forgotten.change_id) not in store._undone_by


async def test_multi_row_forget_undo_late_missing_row_conflicts_cleanly() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)

    del store._rows[("alpha", "r2")]

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    first_row = await store.get(owner_id="alpha", memory_id="r1")
    assert first_row is not None and first_row.status == "forgotten"
    assert ("alpha", forgotten.change_id) not in store._undone_by


async def test_multi_row_forget_undo_external_duplicate_conflicts_cleanly() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)

    external = await _remember(memory, key="call-2", body="PREFERS TEA.")
    assert external.outcome == "changed"

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 2  # r3 plus the external duplicate
    first_row = await store.get(owner_id="alpha", memory_id="r1")
    second_row = await store.get(owner_id="alpha", memory_id="r2")
    assert first_row is not None and first_row.status == "forgotten"
    assert second_row is not None and second_row.status == "forgotten"
    assert ("alpha", forgotten.change_id) not in store._undone_by

    # The target stays undoable once the external duplicate is gone.
    await memory.forget(
        owner_id="alpha",
        memory_id=external.memory_id,
        provenance=_provenance(),
        idempotency_key="forget-2",
    )
    retry = await _undo(memory, forgotten.change_id, key="undo-2")
    assert retry.outcome == "changed"
    assert await store.count_active(owner_id="alpha") == 3


async def test_multi_row_forget_undo_deterministic_id_collision_changes_nothing() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)

    undo_change_id = operation_change_id(
        MemoryOperation(
            owner_id="alpha",
            idempotency_key="undo-1",
            action="undo",
            provenance=MemoryProvenance(origin_kind="undo", origin_id="undo-1"),
            target_change_id=forgotten.change_id,
        )
    )
    squatter_id = operation_record_id("alpha", undo_change_id, index=0)
    await store.insert(_record(memory_id=squatter_id, body="Squatter."))

    with pytest.raises(ValueError, match="already exists"):
        await _undo(memory, forgotten.change_id, key="undo-1")

    squatter = await store.get(owner_id="alpha", memory_id=squatter_id)
    assert squatter is not None and squatter.body == "Squatter."
    assert await store.count_active(owner_id="alpha") == 2  # r3 plus the squatter
    assert ("alpha", forgotten.change_id) not in store._undone_by

    # Clearing the collision leaves the same undo idempotency key settleable.
    del store._rows[("alpha", squatter_id)]
    retry = await _undo(memory, forgotten.change_id, key="undo-1")
    assert retry.outcome == "changed"
    assert retry.change_id == undo_change_id
    assert await store.count_active(owner_id="alpha") == 3


async def test_multi_row_forget_undo_malformed_journal_owner_conflicts_cleanly() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)
    stored = store._operations[("alpha", forgotten.change_id)]
    foreign = replace(stored.before_records[1], owner_id="beta")
    store._operations[("alpha", forgotten.change_id)] = replace(
        stored, before_records=(stored.before_records[0], foreign)
    )

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    assert ("alpha", forgotten.change_id) not in store._undone_by
    assert await store.count_active(owner_id="beta") == 0


async def test_multi_row_forget_undo_malformed_journal_duplicate_ids_conflict_cleanly() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)
    stored = store._operations[("alpha", forgotten.change_id)]
    duplicated = replace(stored.before_records[0], memory_id=stored.before_records[1].memory_id)
    store._operations[("alpha", forgotten.change_id)] = replace(
        stored, before_records=(duplicated, stored.before_records[1])
    )

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    assert ("alpha", forgotten.change_id) not in store._undone_by


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda before: (),
        lambda before: (before[0],),
        lambda before: (before[1], before[0]),
        lambda before: (*before, replace(before[0], memory_id="extra-1")),
    ],
    ids=["empty", "truncated", "reordered", "extra"],
)
async def test_multi_row_forget_undo_malformed_journal_batch_conflicts_cleanly(
    corrupt: Callable[[tuple[MemoryRecord, ...]], tuple[MemoryRecord, ...]],
) -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)
    key = ("alpha", forgotten.change_id)
    stored = store._operations[key]
    original = stored.before_records
    store._operations[key] = replace(stored, before_records=corrupt(original))

    undone = await _undo(memory, forgotten.change_id, key="undo-1")

    assert undone.outcome == "conflict"
    assert await store.count_active(owner_id="alpha") == 1  # only r3 remains active
    assert key not in store._undone_by

    # Repairing the journal leaves the same target settleable.
    store._operations[key] = replace(store._operations[key], before_records=original)
    retry = await _undo(memory, forgotten.change_id, key="undo-2")
    assert retry.outcome == "changed"
    assert await store.count_active(owner_id="alpha") == 3
    assert key in store._undone_by


@pytest.mark.parametrize(
    "run_id, session_id",
    [("", None), (None, "")],
    ids=["empty-run-id", "empty-session-id"],
)
async def test_multi_row_forget_undo_preserves_exact_provenance(
    run_id: str | None, session_id: str | None
) -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)
    provenance = MemoryProvenance(
        origin_kind="undo", origin_id="undo-1", run_id=run_id, session_id=session_id
    )

    undone = await memory.undo(
        owner_id="alpha",
        change_id=forgotten.change_id,
        provenance=provenance,
        idempotency_key="undo-1",
    )

    assert undone.outcome == "changed"
    restored = [
        record
        for memory_id in undone.memory_ids
        if (record := await store.get(owner_id="alpha", memory_id=memory_id)) is not None
    ]
    assert len(restored) == 2
    assert all(record.provenance == provenance for record in restored)
    assert all(record.provenance.origin_kind == "undo" for record in restored)


async def test_concurrent_multi_row_undo_has_one_winner() -> None:
    store = InMemoryMemoryStore()
    memory = Memory(store)
    forgotten = await _forget_duplicate_rows(memory, store)

    first, second = await asyncio.gather(
        _undo(memory, forgotten.change_id, key="undo-1"),
        _undo(memory, forgotten.change_id, key="undo-2"),
    )

    assert sorted((first.outcome, second.outcome)) == ["changed", "conflict"]
    assert await store.count_active(owner_id="alpha") == 3  # r3 plus exactly one restored pair
    assert ("alpha", forgotten.change_id) in store._undone_by

    # A later repeated undo still conflicts via the undone_by mark.
    third = await _undo(memory, forgotten.change_id, key="undo-3")
    assert third.outcome == "conflict"
