# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Tests for the Run-native Corpus Workspace reset command."""

import asyncio
import importlib.util
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from dlightrag.application.access import AccessDeniedError, AccessGate, AllowAllAccessControl

_reset_path = Path(__file__).resolve().parents[2] / "scripts" / "reset_workspace.py"
_spec = importlib.util.spec_from_file_location("reset_workspace_cli", _reset_path)
assert _spec is not None and _spec.loader is not None
_reset = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reset)


def _allow_all_gate() -> AccessGate:
    return AccessGate(AllowAllAccessControl(), None)


def _run(workspace: str, *, status: str = "succeeded") -> Any:
    return SimpleNamespace(
        run_id=f"run-{workspace}",
        run_kind="corpus_mutation",
        lane="corpus_mutation",
        status=status,
        access_scope_id=workspace,
        error_message=None,
        result={"documents": [{"documents_deleted": 2}]},
    )


def test_reset_scope_options_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit):
        _reset.build_parser().parse_args(["--all", "--workspace", "demo"])


def test_removed_direct_reset_options_are_not_accepted() -> None:
    with pytest.raises(SystemExit):
        _reset.build_parser().parse_args(["--dry-run"])
    with pytest.raises(SystemExit):
        _reset.build_parser().parse_args(["--keep-files"])


class _ConcreteResetHarness:
    """Small stateful Run facade: wrong owner lookup never reaches terminal."""

    def __init__(self) -> None:
        self.accepted: dict[str, Any] = {}
        self.observed_owners: list[str] = []

    async def alist_workspace_records(self) -> list[dict[str, str]]:
        return [{"workspace": "finance"}]

    async def create_reset(self, *, workspace: str, **_kwargs: Any) -> Any:
        run = _run(workspace, status="queued")
        self.accepted[run.run_id] = run
        return SimpleNamespace(run=run)

    async def get(self, *, owner_id: str, run_id: str) -> Any | None:
        self.observed_owners.append(owner_id)
        run = self.accepted[run_id]
        return _run(run.access_scope_id) if owner_id == run.access_scope_id else None


async def test_submitted_reset_is_observed_to_terminal_through_its_workspace_owner() -> None:
    harness = _ConcreteResetHarness()
    accepted = await _reset._submit_reset_runs(
        harness,
        harness,
        _allow_all_gate(),
        workspace="finance",
        reset_all=False,
        default_workspace="default",
    )

    async with asyncio.timeout(0.1):
        completed = await _reset._wait_for_runs(harness, accepted)

    assert [run.status for run in completed] == ["succeeded"]
    assert harness.observed_owners == ["finance"]


async def test_all_submits_one_reset_run_per_authorized_workspace() -> None:
    corpora = AsyncMock()
    corpora.alist_workspace_records.return_value = [
        {"workspace": "default"},
        {"workspace": "finance"},
    ]
    mutations = AsyncMock()
    mutations.create_reset.side_effect = [
        SimpleNamespace(run=_run("default")),
        SimpleNamespace(run=_run("finance")),
    ]

    accepted = await _reset._submit_reset_runs(
        corpora,
        mutations,
        _allow_all_gate(),
        workspace=None,
        reset_all=True,
        default_workspace="default",
    )

    assert [run.run_id for run in accepted] == ["run-default", "run-finance"]
    assert mutations.create_reset.await_count == 2
    for call in mutations.create_reset.await_args_list:
        assert call.kwargs["submitted_by"] == "operator:reset-workspace"
        assert call.kwargs["idempotency_key"].startswith("reset:")


@pytest.mark.parametrize("workspace", ["", "   ", "*"])
async def test_explicit_scope_rejects_policy_or_empty_selector(workspace: str) -> None:
    corpora = AsyncMock()
    mutations = AsyncMock()
    with pytest.raises(ValueError, match="concrete workspace"):
        await _reset._submit_reset_runs(
            corpora,
            mutations,
            _allow_all_gate(),
            workspace=workspace,
            reset_all=False,
            default_workspace="default",
        )
    mutations.create_reset.assert_not_awaited()


async def test_explicit_scope_must_pass_the_injected_access_gate() -> None:
    access_control = AsyncMock()
    access_control.check.side_effect = AccessDeniedError("denied")
    mutations = AsyncMock()
    with pytest.raises(AccessDeniedError, match="denied"):
        await _reset._submit_reset_runs(
            AsyncMock(),
            mutations,
            AccessGate(access_control, None),
            workspace="finance",
            reset_all=False,
            default_workspace="default",
        )
    mutations.create_reset.assert_not_awaited()


def test_verbose_enables_debug_logging(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def fake_run(**_kwargs: Any) -> tuple[Any, ...]:
        return (_run("default"),)

    def fake_basic_config(*, level: int) -> None:
        captured["level"] = level

    monkeypatch.setattr(_reset, "_run", fake_run)
    monkeypatch.setattr(_reset.logging, "basicConfig", fake_basic_config)
    monkeypatch.setattr(sys, "argv", ["reset_workspace.py", "--verbose", "--yes"])

    assert _reset.main() == 0
    assert captured["level"] == logging.DEBUG
