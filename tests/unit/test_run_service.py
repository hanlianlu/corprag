# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""The common Run Application service owns lifecycle reads and cancellation."""

import datetime
from unittest.mock import AsyncMock, Mock

from dlightrag.application.runs import RunService, RunView
from dlightrag.engine.runtime import CancellationOutcome, RunEvent

_NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def _runtime_record() -> Mock:
    record = Mock()
    record.run_id = "run"
    record.run_kind = "answer"
    record.lane = "query"
    record.status = "running"
    record.phase = "searching"
    record.durable_progress_version = 2
    record.next_event_sequence = 4
    record.events_trimmed_at = None
    record.cancel_requested = True
    record.result = None
    record.error_kind = None
    record.error_message = None
    record.created_at = _NOW
    record.started_at = _NOW
    record.finished_at = None
    record.request_input.return_value = {"query": "why"}
    return record


async def test_run_view_surfaces_only_bounded_repair_guidance() -> None:
    record = _runtime_record()
    record.phase = "waiting_for_repair"
    record.checkpoint = {
        "repair_reason": "r" * 600,
        "repair_remedy": "Inspect and repair, then resume.",
        "private_phase_state": {"path": "/private/source"},
    }

    view = RunView.from_runtime(record)

    assert view.repair_reason == "r" * 512
    assert view.repair_remedy == "Inspect and repair, then resume."
    assert not hasattr(view, "checkpoint")


async def test_pending_cancellation_commits_before_local_signal() -> None:
    run = _runtime_record()
    repository = AsyncMock()
    repository.request_cancellation.return_value = CancellationOutcome(outcome="pending", run=run)
    scheduler = Mock()
    service = RunService(store=repository, scheduler=scheduler)

    outcome = await service.cancel(owner_id="owner", run_id="run")

    assert isinstance(outcome.run, RunView)
    assert outcome.run.run_id == "run"
    repository.request_cancellation.assert_awaited_once_with(owner_id="owner", run_id="run")
    scheduler.cancel_local.assert_called_once_with("owner", "run")


async def test_terminal_cancellation_does_not_signal_a_worker() -> None:
    repository = AsyncMock()
    repository.request_cancellation.return_value = CancellationOutcome(
        outcome="already_terminal", run=_runtime_record()
    )
    scheduler = Mock()
    service = RunService(store=repository, scheduler=scheduler)

    await service.cancel(owner_id="owner", run_id="run")

    scheduler.cancel_local.assert_not_called()


async def test_reads_and_events_drop_runtime_worker_state() -> None:
    record = _runtime_record()
    repository = AsyncMock()
    repository.get_run.return_value = record
    repository.list_runs.return_value = (record,)

    async def _events():
        yield RunEvent(sequence=3, event_type="token", payload={"text": "hi"}, created_at=_NOW)

    scheduler = Mock()
    scheduler.subscribe.return_value = _events()
    service = RunService(store=repository, scheduler=scheduler)

    view = await service.get(owner_id="owner", run_id="run")
    listed = await service.list(owner_id="owner")
    events = [event async for event in service.subscribe(owner_id="owner", run_id="run")]

    assert view == listed[0]
    assert view is not None
    assert view.request_input() == {"query": "why"}
    assert not hasattr(view, "lease_owner")
    assert [(event.sequence, event.event_type, event.payload) for event in events] == [
        (3, "token", {"text": "hi"})
    ]
