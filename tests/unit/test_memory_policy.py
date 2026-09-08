# Copyright 2025-2026 Hanlian Lu. SPDX-License-Identifier: Apache-2.0
"""Closed Profile Memory operation checklist and standing-block bounds."""

import pytest
from dlightrag_memory.errors import MemoryWriteRejectedError

from dlightrag.engine.answer.memory import (
    MEMORY_BODY_LIMIT,
    RECALL_CHAR_BUDGET,
    MemoryOperation,
    MemoryProvenance,
    MemoryRecord,
    evaluate_memory_operation,
    render_auto_recall,
    reserved_auto_recall_text,
    standing_memory_for_acceptance,
)


def _provenance() -> MemoryProvenance:
    return MemoryProvenance(
        origin_kind="answer_run", origin_id="run-1", run_id="run-1", session_id="sess-1"
    )


def _operation(**overrides: object) -> MemoryOperation:
    payload: dict[str, object] = {
        "owner_id": "owner",
        "idempotency_key": "call-1",
        "action": "remember",
        "kind": "preference",
        "body": "Do not use email.",
        "provenance": _provenance(),
    }
    payload.update(overrides)
    return MemoryOperation(**payload)  # type: ignore[arg-type]


def test_remember_passes() -> None:
    evaluate_memory_operation(_operation())


def test_owner_eligibility_is_root_policy() -> None:
    from dlightrag.engine.answer.memory import memory_owner_allowed

    assert memory_owner_allowed("jwt")
    assert memory_owner_allowed("none")
    assert not memory_owner_allowed("simple")


def test_empty_oversized_cited_and_credential_bodies_are_rejected() -> None:
    bodies = (
        "   ",
        "See [1] for the filing.",
        "x" * (MEMORY_BODY_LIMIT + 1),
        "-----BEGIN PRIVATE KEY-----\nsecret",
        "github_pat_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
    )
    for body in bodies:
        with pytest.raises(MemoryWriteRejectedError):
            evaluate_memory_operation(_operation(body=body))


def test_provenance_and_idempotency_are_enforced() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_operation(_operation(idempotency_key=""))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_operation(
            _operation(provenance=MemoryProvenance(origin_kind="management", origin_id=""))
        )


def test_forget_and_undo_require_exactly_their_target() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_operation(_operation(action="forget", kind=None, body="", memory_id=None))
    evaluate_memory_operation(_operation(action="forget", kind=None, body="", memory_id="mem-1"))
    evaluate_memory_operation(
        _operation(action="undo", kind=None, body="", target_change_id="change-1")
    )


def test_mutation_scope_and_limit_are_paired() -> None:
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_operation(_operation(mutation_scope="run-1"))
    with pytest.raises(MemoryWriteRejectedError):
        evaluate_memory_operation(_operation(mutation_limit=10))


def test_render_auto_recall_keeps_the_framing() -> None:
    records = (
        MemoryRecord(
            owner_id="o",
            memory_id="1",
            kind="preference",
            body="No email.",
            provenance=_provenance(),
        ),
    )
    text = render_auto_recall(records)
    assert "not citable" in text
    assert "the current request takes priority" in text
    assert "[1]" not in text
    assert render_auto_recall(()) == ""


def test_acceptance_reserves_full_recall_for_personal_and_local_identity() -> None:
    reserved = reserved_auto_recall_text()
    assert reserved.count("- (") == 8
    assert len(reserved) >= RECALL_CHAR_BUDGET
    assert standing_memory_for_acceptance("jwt") == reserved
    assert standing_memory_for_acceptance("none") == reserved
    assert standing_memory_for_acceptance("simple") == ""
